#!/usr/bin/env python3
"""Materialize Skill2Env Harbor tasks as Molt RL prompt records for asynchronous DPPO training.

Every record pairs the task instruction with its complete Polar task specification: the
Apptainer image, the pi harness against the Molt gateway, the ``prefix_merging`` trajectory
builder (one token stream per rollout, tool results masked out), and the plain Harbor
evaluator run in the same runtime. No rubric or judge model is configured.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from polar.rollout.models import TaskSpec

PI_INSTALL = (
    # Task clock shims can invalidate package-server certificates during setup.
    'unset LD_PRELOAD; '
    'export PATH="$HOME/.local/node/bin:$HOME/.local/bin:$PATH"; '
    'mkdir -p "$HOME/.cache"; '
    'flock "$HOME/.cache/skill2env-pi.lock" bash -c \''
    'set -e; export PATH="$HOME/.local/node/bin:$HOME/.local/bin:$PATH"; '
    # pi 0.84.2 requires Node >=22.19.
    'if ! node -e "const [major, minor] = process.versions.node.split(/\\./).map(Number); '
    'process.exit(major > 22 || (major === 22 && minor >= 19) ? 0 : 1)" >/dev/null 2>&1; then '
    # Alpine needs its musl build; the upstream tarball requires glibc.
    'if [ -f /etc/alpine-release ]; then /sbin/apk add --no-cache nodejs npm; else '
    'mkdir -p "$HOME/.local/node" "$HOME/.local/bin"; '
    "/usr/bin/curl -LsSf https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.gz | "
    'tar -xz --strip-components=1 -C "$HOME/.local/node"; '
    'for name in node npm npx; do ln -sf "$HOME/.local/node/bin/$name" "$HOME/.local/bin/$name"; done; '
    'fi; fi; if [ "$(pi --version 2>/dev/null || true)" != "0.84.2" ]; then '
    # Drop cached pre-upgrade Node paths and bypass task npm wrappers for global installs.
    'hash -r; npm_cli="$(dirname "$(command -v node)")/../lib/node_modules/npm/bin/npm-cli.js"; '
    'node "$npm_cli" install -g --offline=false --registry=https://registry.npmjs.org '
    "@earendil-works/pi-coding-agent@0.84.2; fi'"
)
RUNTIME_LAYER = """

USER root
RUN unset LD_PRELOAD FAKETIME FAKETIME_DONT_FAKE_MONOTONIC FAKETIME_NO_CACHE \\
    && if ! test -x /usr/bin/curl || ! test -x /usr/bin/git || ! command -v flock >/dev/null; then \\
         export DEBIAN_FRONTEND=noninteractive; \\
         apt-get update; \\
         apt-get install -y --no-install-recommends ca-certificates curl git util-linux; \\
         rm -rf /var/lib/apt/lists/*; \\
       fi
ENV PATH="/root/.local/bin:${PATH}"
"""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=Path("/raid/binfeng/data/s2e/s2ev2_terminal_coding"))
    parser.add_argument("--image-dir", type=Path, default=Path("/raid/binfeng/data/s2e/s2ev2_sif"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-tasks", type=int, default=-1, help="Task cap in stable path order; -1 selects all.")
    parser.add_argument("--task", action="append", default=[], help="Select a task directory name; repeatable.")
    parser.add_argument("--model-name", default="openai/Qwen/Qwen3.8-27B", help="pi provider/model id (display only).")
    parser.add_argument(
        "--context-window",
        type=int,
        default=65536,
        help="pi context window (98304 - 32768). Keep it at data.max_len - rollout.max_new_tokens so pi compacts its history "
        "before vLLM would left-truncate the prompt (the gateway sets truncate_prompt_tokens to that difference).",
    )
    parser.add_argument(
        "--thinking",
        default="high",
        help="pi thinking level (off|minimal|low|medium|high|xhigh|max). pi's own default is xhigh, at which one turn in "
        "five ran its reasoning into max_new_tokens with no tool call and no answer, ending the session at reward 0.",
    )
    parser.add_argument("--build-missing", action="store_true", help="Build missing SIFs with Docker and Apptainer.")
    parser.add_argument("--force", action="store_true", help="Rebuild selected SIFs.")
    parser.add_argument("--skip-image-check", action="store_true", help="Skip the Apptainer workspace preflight.")
    parser.add_argument(
        "--memory-mb",
        type=int,
        default=16384,
        help="Host-RAM cap per task container (cgroup); one uncapped agent process once grew to 1.4 TB.",
    )
    args = parser.parse_args()

    dataset_dir = args.dataset_dir.expanduser().resolve()
    image_dir = args.image_dir.expanduser().resolve()
    if not dataset_dir.is_dir():
        raise SystemExit(f"Skill2Env dataset directory not found: {dataset_dir}")

    task_files = sorted(dataset_dir.glob("task_*/task.toml"))
    if args.task:
        wanted = set(args.task)
        task_files = [path for path in task_files if path.parent.name in wanted]
        missing = sorted(wanted - {path.parent.name for path in task_files})
        if missing:
            raise SystemExit(f"Unknown task(s): {', '.join(missing)}")
    elif args.max_tasks > 0:
        task_files = task_files[: args.max_tasks]
    if not task_files:
        raise SystemExit(f"No Skill2Env tasks selected under {dataset_dir}")

    docker = shutil.which("docker")
    apptainer = shutil.which("apptainer") or shutil.which("singularity")
    if (args.build_missing or not args.skip_image_check) and not apptainer:
        raise SystemExit("Apptainer is required to build or preflight Skill2Env images")
    if args.build_missing and not docker:
        raise SystemExit("Docker is required to build missing Skill2Env images")
    image_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for index, task_file in enumerate(task_files, 1):
        task_dir = task_file.parent
        name = task_dir.name
        instruction_file = task_dir / "instruction.md"
        tests_dir = task_dir / "tests"
        dockerfile = task_dir / "environment" / "Dockerfile"
        for required in (instruction_file, tests_dir / "test.sh", dockerfile):
            if not required.exists():
                raise SystemExit(f"Incomplete Skill2Env task {name}: missing {required}")

        dockerfile_source = dockerfile.read_text()
        workdirs = re.findall(r"^\s*WORKDIR\s+(.+?)\s*$", dockerfile_source, re.IGNORECASE | re.MULTILINE)
        workdir = workdirs[-1].strip("\"'") if workdirs else "/workspace"
        metadata = tomllib.loads(task_file.read_text())
        normalized = re.sub(r"[^a-z0-9_.-]+", "-", name.lower().replace("__", "--")).strip("-")
        image = image_dir / f"{normalized}.sif"
        if args.force or not image.is_file() or image.stat().st_size == 0:
            if not args.build_missing:
                raise SystemExit(f"Missing image {image}; rerun with --build-missing")
            image_ref = f"linex-skill2env:{normalized}"
            source = dockerfile_source.rstrip() + "\n" + RUNTIME_LAYER.lstrip()
            print(f"[{index}/{len(task_files)}] build {name}", flush=True)
            subprocess.run(
                [docker, "build", "--tag", image_ref, "--file", "-", str(dockerfile.parent)],
                input=source,
                text=True,
                check=True,
            )
            temporary = image.with_name(f".{image.name}.tmp-{os.getpid()}")
            temporary.unlink(missing_ok=True)
            try:
                subprocess.run(
                    [apptainer, "build", "--force", str(temporary), f"docker-daemon://{image_ref}"],
                    check=True,
                )
                temporary.replace(image)
                subprocess.run([docker, "image", "rm", image_ref], check=False)
            finally:
                temporary.unlink(missing_ok=True)
        else:
            print(f"[{index}/{len(task_files)}] reuse {image}")

        if not args.skip_image_check:
            subprocess.run(
                [
                    apptainer,
                    "exec",
                    str(image),
                    "sh",
                    "-lc",
                    f"test -d {shlex.quote(workdir)} && test -x /usr/bin/curl && command -v flock >/dev/null",
                ],
                check=True,
            )

        verifier = metadata.get("verifier", {})
        spec = TaskSpec.model_validate(
            {
                "runtime": {
                    "backend": "apptainer",
                    "image": str(image.resolve()),
                    "prepare": [{"type": "exec", "command": PI_INSTALL}],
                    "env": {"HOME": "/root"},
                    "network": "host",
                    "allow_internet": True,
                    "workdir": workdir,
                    "memory_mb": args.memory_mb,
                },
                "agent": {
                    "harness": "pi",
                    "model_name": args.model_name,
                    "settings": {"context_window": args.context_window, "thinking": args.thinking},
                },
                "builder": {"strategy": "prefix_merging"},
                "evaluator": {
                    "strategy": "harbor",
                    "config": {
                        "tests_dir": str(tests_dir.resolve()),
                        "verifier_timeout": float(verifier.get("timeout_sec", 600.0)),
                    },
                    "refresh_runtime": False,
                },
                "metadata": {
                    "skill2env_task": name,
                    "harbor_task": metadata.get("task", {}),
                    "harbor_metadata": metadata.get("metadata", {}),
                },
            }
        )
        rows.append({"prompt": instruction_file.read_text().strip(), "task": spec.model_dump(mode="json")})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_name(f".{args.output.name}.tmp-{os.getpid()}")
    with temporary_output.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary_output.replace(args.output)
    print(f"Wrote {len(rows)} Skill2Env RL records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
