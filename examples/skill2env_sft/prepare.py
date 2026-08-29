#!/usr/bin/env python3
"""Prepare Skill2Env SFT tasks for the Pi teacher harness."""

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

DEFAULT_DATASET = Path("/raid/binfeng/data/s2e/s2ev2_sft_1k")
DEFAULT_IMAGES = Path("/raid/binfeng/data/s2e/s2ev2_sif")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "teacher_tasks.jsonl"
PI_INSTALL = (
    'export PATH="$HOME/.local/node/bin:$HOME/.local/bin:$PATH"; '
    'mkdir -p "$HOME/.cache"; '
    'flock "$HOME/.cache/skill2env-pi.lock" bash -c \''
    'set -e; export PATH="$HOME/.local/node/bin:$HOME/.local/bin:$PATH"; '
    "if ! node --version >/dev/null 2>&1; then "
    'mkdir -p "$HOME/.local/node" "$HOME/.local/bin"; '
    "curl -LsSf https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.gz | "
    'tar -xz --strip-components=1 -C "$HOME/.local/node"; '
    'for name in node npm npx; do ln -sf "$HOME/.local/node/bin/$name" "$HOME/.local/bin/$name"; done; '
    'fi; if [ "$(pi --version 2>/dev/null || true)" != "0.84.2" ]; then '
    "npm install -g @earendil-works/pi-coding-agent@0.84.2; fi'"
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
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-tasks", type=int, default=-1, help="Task count; -1 selects all tasks.")
    parser.add_argument("--task", action="append", default=[], help="Select a task directory name; repeatable.")
    parser.add_argument("--model-name", default="openai/Inferact/GLM-5.3-NVFP4")
    parser.add_argument("--context-window", type=int, default=131072)
    parser.add_argument("--build-missing", action="store_true", help="Build missing SIFs with Docker and Apptainer.")
    parser.add_argument("--force", action="store_true", help="Rebuild selected SIFs.")
    parser.add_argument("--skip-image-check", action="store_true", help="Skip the Apptainer workspace preflight.")
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

        environment = metadata.get("environment", {})
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
                },
                "agent": {
                    "harness": "pi",
                    "model_name": args.model_name,
                    "settings": {"context_window": args.context_window},
                },
                "builder": {"strategy": "per_request"},
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
                    "harbor_environment": environment,
                    "artifacts": metadata.get("artifacts", []),
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
    print(f"Wrote {len(rows)} validated Pi teacher records to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
