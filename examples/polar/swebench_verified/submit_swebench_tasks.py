"""Materialize SWE-bench Verified rows for Molt training.

Each JSONL row contains the plain instruction and one complete Polar task
shape. Molt owns sample count, timeout, sampling, and task identities.

    uv run python examples/polar/swebench_verified/submit_swebench_tasks.py --harness claude_code --max-tasks 10
    uv run python examples/polar/swebench_verified/submit_swebench_tasks.py \
        --harness claude_code --instance-id django__django-15098
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from dataset import (
    SUPPORTED_HARNESSES,
    load_swebench_verified,
    runtime_image_for_instance,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXAMPLE_DIR / "training.jsonl"

# Pinned versions keep the quickstart stable. Bump intentionally.
HARNESS_NPM_PACKAGE: dict[str, str] = {
    "codex": "@openai/codex@0.125.0",
    "opencode": "opencode-ai@1.4.6",
    "claude_code": "@anthropic-ai/claude-code@2.1.111",
    "qwen_code": "@qwen-code/qwen-code@0.14.5",
}

# INIT stage: install the harness CLI, then stage the repo into the workspace.
_PREPARE_BASE = (
    "rm -rf /polar/session/workspace && "
    'mkdir -p /polar/session/logs/agent /polar/session/workspace "$HOME/.venv/bin" && '
    "cp -a /testbed/. /polar/session/workspace/ && "
    'ln -sf /opt/miniconda3/envs/testbed/bin/python "$HOME/.venv/bin/python" && '
    'ln -sf /opt/miniconda3/envs/testbed/bin/python "$HOME/.venv/bin/python3" && '
    "git config --global core.pager '' && "
    "cd /polar/session/workspace && git reset --hard; true"
)


def prepare_command_for_harness(harness: str) -> str:
    return f"npm install -g {HARNESS_NPM_PACKAGE[harness]} && {_PREPARE_BASE}"


def runtime_env_for_harness(harness: str) -> dict[str, str]:
    return {"OPENCODE_FAKE_VCS": "git"} if harness == "opencode" else {}


def evaluator_exclude_patterns_for_harness(harness: str) -> list[str]:
    patterns: list[str] = []
    if harness == "claude_code":
        patterns += [".claude/**", "**/.claude/**"]
    if harness == "qwen_code":
        patterns += [".qwen/**", "**/.qwen/**"]
    return patterns


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--harness", required=True, choices=SUPPORTED_HARNESSES)
    parser.add_argument("--max-tasks", type=int, default=-1, help="Maximum rows to write. -1 = all 500.")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--runtime-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument(
        "--model-name",
        default="gpt-5.4",
        help="Model name the harness sends; the gateway rewrites it to the served model.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def runtime_image_for_backend(image: str, backend: str) -> str:
    if backend == "apptainer" and not image.startswith(("docker-daemon:", "docker://", "oras://")):
        return f"docker-daemon:{image}"
    return image


def select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    instances = load_swebench_verified()
    if args.instance_id:
        wanted = set(args.instance_id)
        selected = [i for i in instances if str(i.get("instance_id")) in wanted]
        missing = sorted(wanted - {str(i.get("instance_id")) for i in selected})
        if missing:
            raise SystemExit(f"Unknown instance_id(s): {', '.join(missing)}")
        return selected
    if args.max_tasks > 0:
        return instances[: args.max_tasks]
    return instances


def main() -> int:
    args = parse_args()
    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    from polar.rollout.models import TaskSpec

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for instance in instances:
            image = runtime_image_for_instance(str(instance["instance_id"]))
            task = TaskSpec.model_validate(
                {
                    "runtime": {
                        "backend": args.runtime_backend,
                        "image": runtime_image_for_backend(image, args.runtime_backend),
                        "prepare": [{"type": "exec", "command": prepare_command_for_harness(args.harness)}],
                        "env": runtime_env_for_harness(args.harness),
                        "network": "host",
                        "workdir": "/polar/session/workspace",
                    },
                    "agent": {"harness": args.harness, "model_name": args.model_name},
                    "builder": {"strategy": "prefix_merging"},
                    "evaluator": {
                        "strategy": "swebench_harness",
                        "config": {
                            "repo_dir": "/testbed",
                            "patch_command": (
                                "cd /polar/session/workspace && git add -A && git diff --cached --binary"
                            ),
                            "instance": instance,
                            "exclude_patterns": evaluator_exclude_patterns_for_harness(args.harness),
                        },
                        "refresh_runtime": True,
                    },
                }
            )
            row = {
                "prompt": str(instance["problem_statement"]).strip(),
                "task": task.model_dump(mode="json"),
            }
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Wrote {len(instances)} validated training rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
