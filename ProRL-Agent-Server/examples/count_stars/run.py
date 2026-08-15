#!/usr/bin/env python3
"""Run the count-stars demo across every harness and print a comparison table.

Each harness gets the same image at `/polar/session/workspace/polar_stars.png`,
inspects it, and writes its star count to `answer.txt`. This exercises image
input through the local OpenAI-compatible inference backend. All harnesses are
submitted at once; per-session detail is visible in the dashboard
(`polar dashboard -c examples/count_stars/topology.vllm.yaml`).

    uv run python examples/count_stars/run.py                 # docker (default)
    uv run python examples/count_stars/run.py --backend apptainer
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

EXAMPLE_DIR = Path(__file__).resolve().parent
IMAGE_FILE = EXAMPLE_DIR / "assets" / "polar_stars.png"
DEFAULT_TOPOLOGY = EXAMPLE_DIR / "topology.vllm.yaml"
RUNTIME_IMAGE = "polar-localhost-count-stars:latest"
RUNTIME_IMAGE_PATH = "/polar/session/workspace/polar_stars.png"
NUM_SAMPLES = 4
TIMEOUT_SECONDS = 300.0
POLL_INTERVAL_SECONDS = 10.0

HARNESSES = ("claude_code", "codex", "gemini_cli")

INSTRUCTION = """\
Use your image viewing tool to inspect `/polar/session/workspace/polar_stars.png`.
Count the visible stars in that image.

Write the answer as a single integer line to `/polar/session/workspace/answer.txt`.
Do not write any other text to that file. Stop after writing the file.
"""

# Pinned versions keep the quickstart stable. Bump intentionally.
HARNESS_NPM_PACKAGE: dict[str, str] = {
    "claude_code": "@anthropic-ai/claude-code@2.1.111",
    "codex": "@openai/codex@0.121.0",
    "gemini_cli": "@google/gemini-cli@0.38.1",
}

# Model name the harness CLI sends; the gateway rewrites it to the served model.
HARNESS_MODEL: dict[str, str] = {
    "claude_code": "claude-opus-4-5",
    "codex": "gpt-5.4",
    "gemini_cli": "gemini-2.5-flash-lite",
}

# INIT stage: install the harness CLI, then set up a clean git workspace.
_WORKSPACE_PREPARE = (
    "rm -rf /polar/session/workspace && "
    "mkdir -p /polar/session/workspace /polar/session/logs/agent && "
    "cd /polar/session/workspace && "
    "git init -q && "
    "git config user.email 'polar@test' && "
    "git config user.name 'Polar'"
)


def runtime_image_for_backend(backend: str) -> str:
    if backend == "apptainer":
        return f"docker-daemon:{RUNTIME_IMAGE}"
    return RUNTIME_IMAGE


def build_task_payload(harness: str, batch_id: str, backend: str) -> dict[str, Any]:
    return {
        "task_id": f"count-stars-{harness}-{batch_id}",
        "instruction": INSTRUCTION,
        "num_samples": NUM_SAMPLES,
        "timeout_seconds": TIMEOUT_SECONDS,
        "runtime": {
            "backend": backend,
            "image": runtime_image_for_backend(backend),
            "prepare": [
                {"type": "exec", "command": f"npm install -g {HARNESS_NPM_PACKAGE[harness]} && {_WORKSPACE_PREPARE}"},
                {"type": "upload_file", "source": str(IMAGE_FILE), "target": RUNTIME_IMAGE_PATH},
            ],
            "network": "host",
            "workdir": "/polar/session/workspace",
        },
        "agent": {"harness": harness, "model_name": HARNESS_MODEL[harness]},
        "builder": {"strategy": "prefix_merging"},
        "evaluator": {"strategy": "session_completed"},
    }


def print_comparison(finished: dict[str, dict[str, Any]], elapsed: float) -> None:
    header = f"{'Harness':<16} {'Completed':>10}"
    print("\n" + "=" * len(header))
    print(header)
    print("-" * len(header))
    for harness, result in finished.items():
        sessions = result.get("results") or []
        done = sum(1 for s in sessions if s.get("status") == "COMPLETED")
        print(f"{harness:<16} {done:>5}/{len(sessions):<4}")
    print("=" * len(header))
    print(f"Wall time: {elapsed:.0f}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=["docker", "apptainer"], default="docker")
    backend = parser.parse_args().backend

    from polar.config import TopologyConfig

    rollout_url = TopologyConfig.load(DEFAULT_TOPOLOGY).rollout.public_url
    batch_id = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")

    print(f"Submitting {len(HARNESSES)} harnesses to {rollout_url} (backend={backend})")
    timeout = httpx.Timeout(None, connect=30.0)
    with httpx.Client(base_url=rollout_url, timeout=timeout) as client:
        task_ids: dict[str, str] = {}
        for harness in HARNESSES:
            payload = build_task_payload(harness, batch_id, backend)
            resp = client.post("/rollout/task/submit", json=payload)
            resp.raise_for_status()
            task_ids[harness] = resp.json()["task_id"]
            print(f"  {harness:<16} -> {task_ids[harness]}")

        print(f"\nPolling every {POLL_INTERVAL_SECONDS:.0f}s (watch live in the dashboard) ...")
        t0 = time.monotonic()
        finished: dict[str, dict[str, Any]] = {}
        while len(finished) < len(HARNESSES):
            time.sleep(POLL_INTERVAL_SECONDS)
            for harness, tid in task_ids.items():
                if harness in finished:
                    continue
                status = client.get(f"/rollout/task/{tid}").json()
                if status["status"] != "running":
                    finished[harness] = status
                    print(f"  [{time.monotonic() - t0:>5.0f}s] {harness} done")
        elapsed = time.monotonic() - t0

    print_comparison(finished, elapsed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
