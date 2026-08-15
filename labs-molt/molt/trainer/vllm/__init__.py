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

"""Lazy imports for vLLM engine helpers."""

__all__ = [
    "create_vllm_engines",
    "batch_vllm_engine_call",
]


def __getattr__(name):
    if name in __all__:
        from .vllm_engine import batch_vllm_engine_call, create_vllm_engines

        exports = {
            "batch_vllm_engine_call": batch_vllm_engine_call,
            "create_vllm_engines": create_vllm_engines,
        }
        return exports[name]
    raise AttributeError(name)
