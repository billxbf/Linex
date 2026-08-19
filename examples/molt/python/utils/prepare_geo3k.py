# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert VeraIsHere/geo3k_imgurl_processed to Molt VLM SFT schema.

Source: https://huggingface.co/datasets/VeraIsHere/geo3k_imgurl_processed
Geometric reasoning with a visible image and a numeric / boxed final answer.
Output schema (load_from_disk-compatible):
    datasource: str
    prompt: list[{role: "user", content: str}]   # chat-style, `<image>` literal
    response: list[{role: "assistant", content: str}]   # SFT target
    images: list[PIL.Image]
"""

import argparse
import io
import re
from pathlib import Path
from typing import Any
from urllib.request import urlopen

from datasets import load_dataset
from PIL import Image

RESPONSE_WRAPPERS = {
    "boxed": "\\boxed{{{}}}",
    "answer": "<answer>{}</answer>",
}


def _extract_answer(example: dict[str, Any]) -> str:
    """Best-effort extraction of the numeric ground truth across schema variants."""
    for key in ("answer", "label", "solution", "ground_truth"):
        value = example.get(key)
        if value is None:
            continue
        text = str(value).strip()
        if not text:
            continue
        # Strip a boxed wrapper if present.
        m = re.search(r"\\boxed\{([^{}]*)\}", text)
        if m:
            return m.group(1).strip()
        return text
    return ""


def _load_image(example: dict[str, Any]) -> Image.Image | None:
    """Return a PIL.Image. Source rows ship a PIL object, raw bytes, or a URL string.

    `VeraIsHere/geo3k_imgurl_processed` stores images as remote URLs; we fetch
    each one lazily during `datasets.map`. For local URLs (file://) urlopen
    handles them transparently.
    """
    # Check both `image` and `images` keys; prefer non-None.
    image = example.get("image")
    if image is None:
        image = example.get("images")
    if isinstance(image, list):
        image = image[0] if image else None
    if image is None:
        return None
    if isinstance(image, Image.Image):
        return image
    if isinstance(image, dict) and "bytes" in image:
        return Image.open(io.BytesIO(image["bytes"]))
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image))
    if isinstance(image, str):
        # Image URL — fetch and decode. Used by VeraIsHere/geo3k_imgurl_processed.
        with urlopen(image, timeout=30) as response:
            data = response.read()
        return Image.open(io.BytesIO(data)).convert("RGB")
    raise TypeError(f"unsupported image type: {type(image)!r}")


_PROMPT_BOILERPLATE_RE = re.compile(
    r"Solve the following math problem.*?(?=<image>|$)|"
    r"You are a math/geometry expert.*?(?=<image>|$)|"
    r"Follow this protocol:.*?(?=<image>|$)|"
    r"Reason step by step.*?$|"
    r"Answer:.*?\\boxed\{.*?\}",
    re.DOTALL,
)


def _strip_verbose_instructions(text: str) -> str:
    """Keep the source question while dropping its verbose answer protocol."""
    cleaned = _PROMPT_BOILERPLATE_RE.sub("", text)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
    return cleaned


def _format_row(example: dict[str, Any], answer_format: str = "boxed") -> dict[str, Any]:
    problem = str(example.get("problem") or example.get("question") or "").strip()
    answer = _extract_answer(example)
    user_text = _strip_verbose_instructions(problem)
    # Defensive: enforce exactly one <image> placeholder so the multimodal
    # processor's image_grid_thw lookup stays aligned with `images`.
    placeholder_count = user_text.count("<image>")
    if placeholder_count == 0:
        user_text = "<image>\n" + user_text
    elif placeholder_count > 1:
        first = user_text.find("<image>")
        head = user_text[: first + len("<image>")]
        tail = user_text[first + len("<image>") :].replace("<image>", "")
        user_text = head + tail
    image = _load_image(example)
    return {
        "datasource": "geo3k_imgurl_processed",
        "prompt": [{"role": "user", "content": user_text}],
        "response": [{"role": "assistant", "content": RESPONSE_WRAPPERS[answer_format].format(answer)}],
        "images": [image] if image is not None else [],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        default="VeraIsHere/geo3k_imgurl_processed",
        help="HF dataset id.",
    )
    parser.add_argument(
        "--train-split",
        default="train",
        help="Source train split name.",
    )
    parser.add_argument(
        "--eval-split",
        default="test",
        help="Source eval split name (fallback to last 5%% of train if missing).",
    )
    parser.add_argument("--max-train", type=int, default=None, help="Optional cap on train rows.")
    parser.add_argument("--max-eval", type=int, default=512, help="Cap on eval rows.")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(".tmp/geo3k"),
        help="Output dir; writes train/ and eval/ via save_to_disk.",
    )
    parser.add_argument("--num-proc", type=int, default=8)
    parser.add_argument(
        "--answer-format",
        choices=sorted(RESPONSE_WRAPPERS),
        default="boxed",
        help="SFT target wrapper: 'boxed' (Qwen / DeepSeek-Math) or 'answer' (Nemotron Omni).",
    )
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(args.source)
    train_ds = ds[args.train_split]
    if args.eval_split in ds:
        eval_ds = ds[args.eval_split]
    else:
        n_eval = max(1, int(0.05 * len(train_ds)))
        eval_ds = train_ds.select(range(len(train_ds) - n_eval, len(train_ds)))
        train_ds = train_ds.select(range(len(train_ds) - n_eval))

    if args.max_train is not None:
        train_ds = train_ds.select(range(min(args.max_train, len(train_ds))))
    eval_ds = eval_ds.select(range(min(args.max_eval, len(eval_ds))))

    columns_to_drop = [c for c in train_ds.column_names if c not in ("__index_level_0__",)]
    fmt_kwargs = {"answer_format": args.answer_format}
    train_out = train_ds.map(_format_row, fn_kwargs=fmt_kwargs, num_proc=args.num_proc, remove_columns=columns_to_drop)
    eval_out = eval_ds.map(_format_row, fn_kwargs=fmt_kwargs, num_proc=args.num_proc, remove_columns=columns_to_drop)

    train_out.save_to_disk(args.out_dir / "train")
    eval_out.save_to_disk(args.out_dir / "eval")
    print(f"wrote {len(train_out)} train + {len(eval_out)} eval rows to {args.out_dir}")


if __name__ == "__main__":
    main()
