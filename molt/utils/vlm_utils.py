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
#
# Adapted from OpenRLHF (https://github.com/OpenRLHF/OpenRLHF),
# Copyright (c) OpenRLHF contributors, licensed under the Apache License, Version 2.0.

"""VLM utilities: image loading, processor-based tokenization, multimodal tensor merging."""

import base64
import io
import logging
import re
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from PIL import Image

logger = logging.getLogger(__name__)


def split_image_placeholder(message: dict) -> dict:
    """User-content string `"<image>\\nproblem"` → structured
    `[{"type":"image"}, {"type":"text", "text":"problem"}]` so chat templates
    that iterate structured content (Qwen2-VL / Qwen3-VL) render correctly.
    Strings with no ``<image>`` token pass through unchanged.
    """
    content = message.get("content")
    if not isinstance(content, str) or "<image>" not in content:
        return message
    parts = []
    remaining = content
    while "<image>" in remaining:
        before, _, remaining = remaining.partition("<image>")
        if before:
            parts.append({"type": "text", "text": before})
        parts.append({"type": "image"})
    if remaining:
        parts.append({"type": "text", "text": remaining})
    return {**message, "content": parts}


def should_expand_image_placeholder(tokenizer) -> bool:
    """Whether stored `"<image>"` strings must be split into structured content.

    A model's chat template emits its image token in one of two ways:
      * **Structured-content branch** (Qwen2-VL / Qwen3-VL): the template
        iterates `[{"type": "image"}, {"type": "text", ...}]` and emits a
        model-specific token (e.g. `<|image_pad|>`). The literal `<image>` the
        dataset stores would never be expanded — so it must be split first.
      * **Literal branch** (Nemotron-Omni): the template renders the stored
        `<image>` verbatim, so splitting it would break the placeholder.

    We detect this from the tokenizer/processor's ``image_token``: anything
    other than the stored `<image>` means the structured branch is required.
    Shared by SFT (`SFTDataset`) and RL (`PromptDataset`) so the two never
    disagree for the same model.
    """
    return getattr(tokenizer, "image_token", "<image>") != "<image>"


def media_token_ids(processor) -> set:
    """Image/video placeholder token ids, resolved in ONE place.

    A family that has no such token still answers, resolving it to ``<unk>``/``<pad>``
    rather than omitting it — an image-only checkpoint reports its video token as the unk
    id. Keeping that would make the truncation guard read ordinary ``<unk>`` text in a cut
    tail as a dropped image and throw the rollout away, so drop those.
    """
    objs = (processor, getattr(processor, "tokenizer", None))
    attrs = ("image_token_id", "video_token_id", "img_context_token_id", "video_context_token_id")
    ids = {getattr(obj, attr, None) for obj in objs for attr in attrs}
    ids -= {getattr(obj, attr, None) for obj in objs for attr in ("unk_token_id", "pad_token_id")}
    return {tid for tid in ids if isinstance(tid, int)}


def _pad_to_common_hw(tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    """Right/bottom-pad a list of tensors so they share a common (H, W).

    Each tensor's last two dims are treated as spatial. Zero-padded. Used to
    stack/concat variable-resolution image patches into a single batch tensor.
    """
    max_h = max(int(t.shape[-2]) for t in tensors)
    max_w = max(int(t.shape[-1]) for t in tensors)
    return [F.pad(t, (0, max_w - int(t.shape[-1]), 0, max_h - int(t.shape[-2]))) for t in tensors]


def load_images(image_refs: Union[str, List[str], Image.Image, List[Any]]) -> List[Image.Image]:
    """Load PIL images from paths, URLs, base64 strings, raw bytes, or PIL objects.

    Invalid entries are skipped with a warning. Output is always RGB —
    Qwen3-style processors reject RGBA ("Unable to infer channel dimension
    format") or grayscale.
    """
    if image_refs is None:
        return []
    if not isinstance(image_refs, list):
        image_refs = [image_refs]

    pil_images = []
    for img in image_refs:
        try:
            if isinstance(img, Image.Image):
                loaded = img
            elif isinstance(img, bytes):
                loaded = Image.open(io.BytesIO(img))
            elif isinstance(img, dict):
                # HuggingFace datasets Image() feature serializes to
                # {"bytes": <png-bytes>, "path": <str|None>} on save_to_disk.
                if img.get("bytes") is not None:
                    loaded = Image.open(io.BytesIO(img["bytes"]))
                elif img.get("path"):
                    loaded = Image.open(img["path"])
                else:
                    logger.warning(f"Skipping image dict with no bytes/path: {img!r}")
                    continue
            elif isinstance(img, str):
                if img.startswith(("http://", "https://")):
                    import requests

                    loaded = Image.open(io.BytesIO(requests.get(img, timeout=30).content))
                elif img.startswith("file://"):
                    loaded = Image.open(img[len("file://") :])
                elif img.startswith("data:image") or (
                    len(img) > 256 and re.fullmatch(r"[A-Za-z0-9+/\n\r]+=*", img[:512])
                ):
                    raw = img.split(",", 1)[-1] if img.startswith("data:") else img
                    loaded = Image.open(io.BytesIO(base64.b64decode(raw)))
                else:
                    loaded = Image.open(img)
            else:
                logger.warning(f"Skipping unsupported image type: {type(img)}")
                continue
            if loaded.mode != "RGB":
                loaded = loaded.convert("RGB")
            pil_images.append(loaded)
        except Exception as e:
            logger.warning(f"Failed to load image {img!r}: {e}")
    return pil_images


def process_prompt_with_images(
    processor, prompt: str, images: Any
) -> Tuple[List[int], Optional[Dict], List[Image.Image]]:
    """Tokenize a prompt with images using a VLM processor (AutoProcessor).

    Returns:
        (token_ids, mm_train_inputs, pil_images)
        - mm_train_inputs: dict of multimodal tensors (pixel_values, image_grid_thw, ...)
          or None when no images are present.
        - pil_images: loaded PIL images (reused for vLLM multi_modal_data).
    """
    pil_images = load_images(images)
    refs = images if isinstance(images, list) else ([images] if images is not None else [])
    non_none_refs = [r for r in refs if r is not None]

    # No images: text-only tokenization (valid for text-only samples in mixed datasets).
    if not pil_images:
        if non_none_refs:
            # Caller provided real refs but none loaded — falling through to
            # text-only would leave placeholder tokens with no pixel_values.
            raise ValueError(
                f"All images failed to load ({images!r}). The prompt likely "
                "contains image placeholder tokens that require pixel_values."
            )
        token_ids = processor(text=prompt, add_special_tokens=False, return_tensors="pt")["input_ids"][0].tolist()
        return token_ids, None, []

    # Warn on partial load failures: prompt may expect more images than were
    # loaded, which would misalign placeholder tokens.
    if len(pil_images) < len(non_none_refs):
        logger.warning(
            f"Only {len(pil_images)}/{len(non_none_refs)} images loaded successfully. "
            "Image placeholder tokens in the prompt may not match pixel_values."
        )

    proc_out = processor(text=[prompt], images=pil_images, add_special_tokens=False, return_tensors="pt")
    token_ids = proc_out["input_ids"][0].tolist()

    # Drop sequence-length-dependent fields (input_ids/attention_mask/token_type_ids):
    # they get reconstructed from input_ids during training, where the sequence
    # also includes the response (the processor only saw the prompt).
    _skip_keys = {"input_ids", "attention_mask", "token_type_ids", "mm_token_type_ids"}
    mm_train_inputs = {k: v for k, v in proc_out.items() if k not in _skip_keys}
    return token_ids, (mm_train_inputs or None), pil_images


def merge_mm_train_inputs(mm_train_inputs_list: list, device) -> Dict[str, torch.Tensor]:
    """Merge per-sample multimodal tensor dicts into one batched dict on *device*.

    Each ``mm_train_inputs_list`` element is a per-sample dict (or list of dicts,
    or None). Tensors are concatenated along dim=0; pixel_values is padded to a
    common HxW first when entries have ndim==4.
    """
    merged: Dict[str, list] = {}
    for item in mm_train_inputs_list:
        for mm_dict in item if isinstance(item, list) else [item]:
            if mm_dict is None:
                continue
            for key, val in mm_dict.items():
                merged.setdefault(key, []).append(val if isinstance(val, torch.Tensor) else torch.tensor(val))

    output = {}
    for key, values in merged.items():
        if key == "pixel_values" and all(torch.is_tensor(v) and v.ndim == 4 for v in values):
            values = _pad_to_common_hw(values)
        output[key] = torch.cat(values, dim=0).to(device)
    return output
