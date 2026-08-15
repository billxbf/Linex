"""Helpers for the full SWE-Gym SkyRL train dataset."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DATASET_NAME = "NovaSky-AI/SkyRL-v0-293-data"
DATASET_SPLITS = ("train",)
EXPECTED_SPLIT_SIZES = {"train": 293}
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "polar"
LEGACY_SWEBENCH_IMAGE_REPOS = {
    "marshmallow-code/marshmallow",
    "pydicom/pydicom",
    "pylint-dev/astroid",
    "pvlib/pvlib-python",
    "pyvista/pyvista",
    "sqlfluff/sqlfluff",
}


def registry_image_for_instance_id(instance_id: str) -> str:
    owner, repo_with_issue = instance_id.split("__", 1)
    repo, issue_id = repo_with_issue.rsplit("-", 1)
    if f"{owner}/{repo}" in LEGACY_SWEBENCH_IMAGE_REPOS:
        return f"swebench/sweb.eval.x86_64.{owner}_1776_{repo}-{issue_id}:latest"

    suffix = instance_id.replace("__", "_s_").lower()
    return f"xingyaoww/sweb.eval.x86_64.{suffix}:latest"


def _cache_path_for_split(split: str) -> Path:
    return DEFAULT_CACHE_DIR / f"swegym_skyrl_v0_293_{split}.json"


def _normalize_instance(instance: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(instance)
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = normalized.get(key)
        if isinstance(value, str):
            try:
                normalized[key] = json.loads(value)
            except (TypeError, json.JSONDecodeError):
                pass
    return normalized


def _load_split_from_parquet(split: str) -> list[dict[str, Any]]:
    if split not in DATASET_SPLITS:
        raise ValueError(f"Unknown split {split!r}; expected one of {DATASET_SPLITS}")

    from huggingface_hub import hf_hub_download
    import pyarrow.parquet as pq

    parquet_path = hf_hub_download(
        repo_id=DATASET_NAME,
        filename=f"{split}.parquet",
        repo_type="dataset",
    )
    table = pq.read_table(parquet_path)
    instances: list[dict[str, Any]] = []
    for row in table.to_pylist():
        instance = row.get("instance")
        if not isinstance(instance, dict):
            raise ValueError(f"Dataset row in split {split!r} does not contain an instance dict")
        instances.append(_normalize_instance(instance))
    return instances


def fetch_dataset_instances(
    split: str,
    *,
    refresh: bool = False,
    cache_path: Path | None = None,
) -> list[dict[str, Any]]:
    cache_file = cache_path or _cache_path_for_split(split)
    if cache_file.exists() and not refresh:
        instances = json.loads(cache_file.read_text())
    else:
        instances = _load_split_from_parquet(split)
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(instances, indent=2, ensure_ascii=True, sort_keys=True))

    expected = EXPECTED_SPLIT_SIZES.get(split)
    if expected is not None and len(instances) != expected:
        raise RuntimeError(f"Expected {expected} SWE-Gym {split} instances, got {len(instances)}")
    return instances


def fetch_all_instances(
    *,
    refresh: bool = False,
    splits: tuple[str, ...] = DATASET_SPLITS,
) -> list[dict[str, Any]]:
    seen: set[str] = set()
    instances: list[dict[str, Any]] = []
    for split in splits:
        for instance in fetch_dataset_instances(split, refresh=refresh):
            instance_id = str(instance["instance_id"])
            if instance_id in seen:
                continue
            seen.add(instance_id)
            instances.append(instance)
    return instances
