#!/usr/bin/env python3
"""Submit SWE-bench Verified tasks to the Polar rollout server.

Each task runs an agent in a per-instance container and is graded by the
official `swebench` harness. Tasks are submitted at once; live progress and
per-session detail are visible in the dashboard
(`polar dashboard -c examples/swebench_verified/topology.vllm.yaml`).

    uv run python examples/swebench_verified/submit_swebench_tasks.py --harness claude_code --max-tasks 10
    uv run python examples/swebench_verified/submit_swebench_tasks.py --harness codex --max-tasks 50 --num-samples 4
    uv run python examples/swebench_verified/submit_swebench_tasks.py --harness claude_code --instance-id django__django-15098
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx

from dataset import (
    SUPPORTED_HARNESSES,
    load_swebench_verified,
    runtime_image_for_instance,
    sanitize_instance_id,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.vllm.yaml"
POLL_INTERVAL_SECONDS = 15.0

# Pinned versions keep the quickstart stable. Bump intentionally.
HARNESS_NPM_PACKAGE: dict[str, str] = {
    "codex": "@openai/codex@0.121.0",
    "opencode": "opencode-ai@1.4.6",
    "claude_code": "@anthropic-ai/claude-code@2.1.111",
    "qwen_code": "@qwen-code/qwen-code@0.14.5",
}

# INIT stage: install the harness CLI, then stage the repo into the workspace.
_PREPARE_BASE = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/logs/agent /polar/session/workspace \"$HOME/.venv/bin\" && "
    "cp -a /testbed/. /polar/session/workspace/ && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python \"$HOME/.venv/bin/python\" && "
    "ln -sf /opt/miniconda3/envs/testbed/bin/python \"$HOME/.venv/bin/python3\" && "
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
    parser.add_argument("--num-samples", type=int, default=1, help="Samples per task (pass@k).")
    parser.add_argument("--max-tasks", type=int, default=-1, help="Maximum tasks to submit. -1 = all 500.")
    parser.add_argument("--instance-id", action="append", default=[])
    parser.add_argument("--timeout-seconds", type=float, default=3600.0)
    parser.add_argument("--runtime-backend", choices=["docker", "apptainer"], default="docker")
    parser.add_argument(
        "--model-name",
        default="gpt-5.4",
        help="Model name the harness sends; the gateway rewrites it to the served model.",
    )
    return parser.parse_args()


def runtime_image_for_backend(image: str, backend: str) -> str:
    if backend == "apptainer" and not image.startswith(("docker-daemon:", "docker://", "oras://")):
        return f"docker-daemon:{image}"
    return image


def docker_image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


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


def build_task_request(args: argparse.Namespace, instance: dict[str, Any], batch_id: str) -> dict[str, Any]:
    instance_id = str(instance["instance_id"])
    image = runtime_image_for_instance(instance_id)
    return {
        "task_id": f"swebench-{args.harness}-{sanitize_instance_id(instance_id)}-{batch_id}",
        "instruction": str(instance["problem_statement"]).strip(),
        "num_samples": args.num_samples,
        "timeout_seconds": args.timeout_seconds,
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
                "patch_command": "cd /polar/session/workspace && git add -A && git diff --cached --binary",
                "instance": instance,
                "exclude_patterns": evaluator_exclude_patterns_for_harness(args.harness),
            },
            "refresh_runtime": True,
        },
    }


def task_stats(result: dict[str, Any]) -> tuple[int, int]:
    """Return (sessions with reward==1, total sessions) for one finished task."""
    sessions = result.get("results") or []
    reward_one = 0
    for session in sessions:
        traces = (session.get("trajectory") or {}).get("traces") or []
        if traces and traces[-1].get("reward") == 1.0:
            reward_one += 1
    return reward_one, len(sessions)


def print_summary(stats: dict[str, tuple[int, int]], elapsed: float) -> None:
    total_tasks = len(stats)
    resolved = sum(1 for r1, _ in stats.values() if r1 > 0)
    total_sessions = sum(total for _, total in stats.values())
    reward_one = sum(r1 for r1, _ in stats.values())

    print("\n" + "=" * 72)
    print("  SWE-bench Verified — Reward Summary")
    print("=" * 72)
    print(f"  Tasks resolved (>=1):  {resolved}/{total_tasks}  ({100 * resolved / max(total_tasks, 1):.1f}%)")
    print(f"  Sessions reward=1:     {reward_one}/{total_sessions}  ({100 * reward_one / max(total_sessions, 1):.1f}%)")
    print(f"  Wall time:             {elapsed:.0f}s")
    print("=" * 72)
    print(f"\n  {'Instance ID':<45} {'Resolved':>12}")
    print("  " + "-" * 59)
    for iid in sorted(stats):
        r1, total = stats[iid]
        print(f"  {iid:<45} {f'{r1}/{total}':>12}")
    print("\n  Per-session detail: polar dashboard -c examples/swebench_verified/topology.vllm.yaml")


def main() -> int:
    args = parse_args()
    batch_id = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    instances = select_instances(args)
    if not instances:
        raise SystemExit("No instances selected.")

    ready, missing = [], []
    for instance in instances:
        image_ref = runtime_image_for_instance(str(instance["instance_id"]))
        (ready if docker_image_exists(image_ref) else missing).append(instance)
    if not ready:
        raise SystemExit("No runtime images found. Run: python build_images.py")
    if missing:
        print(f"Skipping {len(missing)} instance(s) with missing images. Build them with: python build_images.py")
    instances = ready

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(DEFAULT_TOPOLOGY).rollout.public_url
    print(f"Submitting {len(instances)} task(s) to {rollout_url} "
          f"(harness={args.harness}, samples={args.num_samples}, backend={args.runtime_backend})")

    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}  # instance_id -> rollout task_id
        for instance in instances:
            iid = str(instance["instance_id"])
            payload = build_task_request(args, instance, batch_id)
            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            task_ids[iid] = resp.json()["task_id"]

        print(f"Polling every {POLL_INTERVAL_SECONDS:.0f}s (watch live in the dashboard) ...")
        t0 = time.monotonic()
        stats: dict[str, tuple[int, int]] = {}
        while len(stats) < len(task_ids):
            time.sleep(POLL_INTERVAL_SECONDS)
            for iid, tid in task_ids.items():
                if iid in stats:
                    continue
                status = client.get(f"/rollout/task/{tid}").json()
                if status["status"] != "running":
                    r1, total = task_stats(status)
                    stats[iid] = (r1, total)
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {iid:<45} resolved={r1}/{total}  "
                          f"({len(stats)}/{len(task_ids)} done)")
        elapsed = time.monotonic() - t0

    print_summary(stats, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
