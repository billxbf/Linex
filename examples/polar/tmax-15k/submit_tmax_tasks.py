"""Materialize TMax-15K-Harbor rows for Molt training.

Each JSONL row contains the task instruction and a complete Polar runtime,
agent, builder, and evaluator shape. Molt owns sampling and session settings.

    uv run python examples/polar/tmax-15k/submit_tmax_tasks.py --dataset-dir <dir> --harness codex --max-tasks 10
    uv run python examples/polar/tmax-15k/submit_tmax_tasks.py \
        --dataset-dir <dir> --harness codex --task task_000123_ab12cd34
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dataset import (
    SUPPORTED_HARNESSES,
    TmaxTask,
    load_tasks,
    runtime_image_for,
    sif_filename_for,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT = EXAMPLE_DIR / "training.jsonl"

# Per-harness INIT install command. The Node CLIs install globally. hermes and
# mini-swe-agent are PyPI packages that need Python >=3.11, but TMax task images
# ship whatever Python they were built with (this set includes 3.10), so installing
# against the image's system Python fails. We install them with uv against a managed
# 3.12 interpreter; `uv tool install` drops the entry point in $HOME/.local/bin,
# which both presets' PATH already includes.
# codex must match presets/codex.py DEFAULT_CODEX_VERSION (the preset hard-fails
# on a version mismatch). Bump versions intentionally.
HARNESS_INSTALL: dict[str, str] = {
    "codex": "npm install -g @openai/codex@0.125.0",
    "claude_code": "npm install -g @anthropic-ai/claude-code@2.1.111",
    "opencode": "npm install -g opencode-ai@1.4.6",
    "qwen_code": "npm install -g @qwen-code/qwen-code@0.14.5",
    "pi": "npm install -g @mariozechner/pi-coding-agent@0.67.68",
    "hermes": (
        "curl -LsSf https://astral.sh/uv/install.sh | sh "
        '&& export PATH="$HOME/.local/bin:$PATH" '
        "&& uv tool install --python 3.12 hermes-agent==0.15.1"
    ),
    "mini_swe_agent": (
        "curl -LsSf https://astral.sh/uv/install.sh | sh "
        '&& export PATH="$HOME/.local/bin:$PATH" '
        "&& uv tool install --python 3.12 mini-swe-agent==2.4.2"
    ),
}


def model_name_for(harness: str, model_name: str) -> str:
    """The pi preset requires ``provider/model`` form; the gateway rewrites the
    model id to the served model, so the provider prefix is all that matters."""
    if harness == "pi" and "/" not in model_name:
        return f"openai/{model_name}"
    return model_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", required=True, help="Exported Harbor dataset directory.")
    parser.add_argument("--harness", required=True, choices=SUPPORTED_HARNESSES)
    parser.add_argument("--max-tasks", type=int, default=-1, help="Max rows to write. -1 = all.")
    parser.add_argument("--task", action="append", default=[], help="Only materialize these task(s). Repeatable.")
    parser.add_argument(
        "--workdir",
        default="/root",
        help="Agent working dir in the container. TMax instructions use absolute paths; "
        "pass '/' to match Harbor exactly for tasks with relative-path instructions.",
    )
    parser.add_argument(
        "--model-name",
        default="gpt-5.4",
        help="Model name the harness sends; the gateway rewrites it to the served model.",
    )
    parser.add_argument("--runtime-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument(
        "--apptainer-image-dir",
        default=None,
        help="Dir of prebuilt .sif files for a docker-free Slurm flow. With "
        "--runtime-backend apptainer, launch <dir>/<task>.sif directly instead of "
        "reading from a local docker daemon. Build them with prepare_apptainer_images.py.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def resolve_runtime_image(args: argparse.Namespace, task: TmaxTask) -> str:
    """The image reference Polar's runtime launches for *task*.

    ``.sif`` path (docker-free apptainer) > ``docker-daemon:`` (apptainer reading
    the local docker daemon) > the plain docker tag.
    """
    if args.runtime_backend == "apptainer" and args.apptainer_image_dir:
        return str(Path(args.apptainer_image_dir).expanduser() / sif_filename_for(task.name))
    image = runtime_image_for(task.name)
    if args.runtime_backend == "apptainer" and not image.startswith(("docker-daemon:", "docker://", "oras://")):
        return f"docker-daemon:{image}"
    return image


def main() -> int:
    args = parse_args()
    tasks = load_tasks(args.dataset_dir, max_tasks=args.max_tasks, names=args.task or None)

    from polar.rollout.models import TaskSpec

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as output:
        for task in tasks:
            spec = TaskSpec.model_validate(
                {
                    "runtime": {
                        "backend": args.runtime_backend,
                        "image": resolve_runtime_image(args, task),
                        "prepare": [{"type": "exec", "command": HARNESS_INSTALL[args.harness]}],
                        "env": {"HOME": args.workdir} if args.workdir == "/root" else {},
                        "network": "host",
                        "workdir": task.workdir or args.workdir,
                    },
                    "agent": {
                        "harness": args.harness,
                        "model_name": model_name_for(args.harness, args.model_name),
                    },
                    "builder": {"strategy": "prefix_merging"},
                    "evaluator": {
                        "strategy": "harbor",
                        "config": {
                            "tests_dir": str(task.tests_dir.resolve()),
                            "verifier_timeout": task.verifier_timeout,
                        },
                        "refresh_runtime": False,
                    },
                    "metadata": {"tmax_task": task.name},
                }
            )
            row = {"prompt": task.instruction, "task": spec.model_dump(mode="json")}
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    print(f"Wrote {len(tasks)} validated training rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
