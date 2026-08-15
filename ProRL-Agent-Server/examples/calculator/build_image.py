#!/usr/bin/env python3
"""Build the shared calculator runtime image."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

EXAMPLE_DIR = Path(__file__).resolve().parent
DOCKERFILE_DIR = EXAMPLE_DIR / "runtime"
DEFAULT_IMAGE = "polar-localhost-calculator:latest"
IMAGE_LAYOUT_VERSION = "3"
IMAGE_VERSION_LABEL = "io.polar.calculator-image-version"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", default=DEFAULT_IMAGE)
    parser.add_argument("--force", action="store_true", help="Rebuild even if the image is current.")
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
            "docker",
            "image",
            "inspect",
            "--format",
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


def main() -> int:
    args = parse_args()
    if not (DOCKERFILE_DIR / "Dockerfile").is_file():
        raise SystemExit(f"No Dockerfile found at {DOCKERFILE_DIR / 'Dockerfile'}")

    if image_exists(args.image) and not args.force:
        version = image_layout_version(args.image)
        if version == IMAGE_LAYOUT_VERSION:
            print(f"Image is up to date: {args.image}")
            return 0
        print(f"Rebuilding {args.image} (v{version} -> v{IMAGE_LAYOUT_VERSION})")

    run_command(
        [
            "docker",
            "build",
            "--build-arg",
            f"POLAR_CALCULATOR_IMAGE_VERSION={IMAGE_LAYOUT_VERSION}",
            "--tag",
            args.image,
            str(DOCKERFILE_DIR),
        ]
    )
    print(f"Built image: {args.image}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
