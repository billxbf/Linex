#!/usr/bin/env python3
"""Build shared runtime images for SWE-bench Verified tasks.

Each runtime image layers Node.js on top of the per-instance SWE-bench Docker
image.  Harness-specific CLIs are installed at task time via the prepare command.

Usage:
    python build_images.py                                           # all 500
    python build_images.py --max-tasks 10                            # first 10
    python build_images.py --instance-id django__django-15098
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

from dataset import (
    base_image_for_instance,
    load_swebench_verified,
    runtime_image_for_instance,
)

EXAMPLE_DIR = Path(__file__).resolve().parent
IMAGE_LAYOUT_VERSION = "1"
IMAGE_VERSION_LABEL = "io.polar.swebench-image-version"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instance-id",
        action="append",
        default=[],
        help="Only build for specific instance_id(s). Can be repeated.",
    )
    parser.add_argument(
        "--max-tasks",
        type=int,
        default=-1,
        help="Max instances to build. -1 = all.",
    )
    parser.add_argument("--force", action="store_true", help="Rebuild existing images.")
    parser.add_argument("--refresh-dataset-cache", action="store_true")
    return parser.parse_args()


def run_command(command: list[str]) -> None:
    print("+", " ".join(command))
    subprocess.run(command, check=True)


def image_exists(image_ref: str) -> bool:
    return subprocess.run(
        ["docker", "image", "inspect", image_ref],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0


def image_layout_version(image_ref: str) -> str | None:
    result = subprocess.run(
        [
            "docker", "image", "inspect", "--format",
            '{{ index .Config.Labels "' + IMAGE_VERSION_LABEL + '" }}',
            image_ref,
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def select_instances(args: argparse.Namespace) -> list[dict[str, Any]]:
    instances = load_swebench_verified(refresh=args.refresh_dataset_cache)
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

    dockerfile_dir = EXAMPLE_DIR / "runtime"
    if not (dockerfile_dir / "Dockerfile").is_file():
        raise SystemExit(f"No Dockerfile found at {dockerfile_dir / 'Dockerfile'}")

    print(f"Building {len(instances)} runtime image(s) ...")
    built = skipped = failed = 0
    for idx, instance in enumerate(instances, 1):
        instance_id = str(instance["instance_id"])
        base_image = base_image_for_instance(instance)
        runtime_image = runtime_image_for_instance(instance_id)

        if image_exists(runtime_image) and not args.force:
            ver = image_layout_version(runtime_image)
            if ver == IMAGE_LAYOUT_VERSION:
                print(f"  [{idx}/{len(instances)}] skip: {runtime_image}")
                skipped += 1
                continue
            print(f"  [{idx}/{len(instances)}] rebuild (v{ver} -> v{IMAGE_LAYOUT_VERSION})")

        try:
            run_command(["docker", "pull", base_image])
            run_command([
                "docker", "build",
                "--build-arg", f"BASE_IMAGE={base_image}",
                "--build-arg", f"POLAR_SWEBENCH_IMAGE_VERSION={IMAGE_LAYOUT_VERSION}",
                "--tag", runtime_image,
                str(dockerfile_dir),
            ])
            built += 1
        except subprocess.CalledProcessError as exc:
            print(f"  [{idx}/{len(instances)}] FAILED {instance_id}: {exc}")
            failed += 1

    print(f"\nDone. built={built}  skipped={skipped}  failed={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
