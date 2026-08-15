#!/usr/bin/env python3
"""Prepare the full SWE-Gym train JSONL dataset for Slime.

Each row contains:
  - prompt:       chat-formatted list [{"role": "user", "content": problem_statement}]
                  (list form is required when the HF checkpoint ships a VLM
                  processor, as Qwen3.5-4B does — slime asserts list prompts
                  under that path. See `slime/slime/utils/data.py:243`.)
  - label:        always "" (reward comes from Polar evaluator, not label matching)
  - metadata:     instance ID, instance dict, and split
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from sample_tasks import (
    EXPECTED_SPLIT_SIZES,
    fetch_dataset_instances,
)

SPLIT = "train"
OUTPUT = Path(__file__).resolve().parent / "swegym_train_293.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refresh-dataset-cache",
        action="store_true",
        help="Refresh cached HuggingFace dataset rows before writing JSONL files.",
    )
    return parser.parse_args()


def row_for_instance(instance: dict, split: str) -> dict:
    instance_id = str(instance["instance_id"])
    return {
        "prompt": [
            {"role": "user", "content": str(instance["problem_statement"]).strip()}
        ],
        "label": "",
        "metadata": {
            "instance_id": instance_id,
            "instance": instance,
            "split": split,
        },
    }


def write_split(split: str, output: Path, *, refresh: bool) -> int:
    instances = fetch_dataset_instances(split, refresh=refresh)
    rows = [row_for_instance(instance, split) for instance in instances]
    output.write_text("\n".join(json.dumps(r, ensure_ascii=True) for r in rows) + "\n")

    expected = EXPECTED_SPLIT_SIZES.get(split)
    if expected is not None and len(rows) != expected:
        raise RuntimeError(f"Expected {expected} {split} rows, wrote {len(rows)}")
    print(f"Wrote {len(rows)} {split} tasks to {output}")
    return len(rows)


def main() -> None:
    args = parse_args()
    write_split(SPLIT, OUTPUT, refresh=args.refresh_dataset_cache)


if __name__ == "__main__":
    main()
