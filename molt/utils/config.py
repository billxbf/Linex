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

"""Hierarchical argparse helper.

``hierarchize(args)`` converts a flat argparse Namespace whose dest names contain
dots (e.g. ``"muon.lr"``) into a nested SimpleNamespace so callers can write
``args.muon.lr`` instead of ``getattr(args, "muon.lr")``.  Keys without dots stay
at the top level.
"""

from types import SimpleNamespace


def hierarchize(args):
    root = {}
    for k, v in vars(args).items():
        parts = k.split(".")
        node = root
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = v

    def build(x):
        if isinstance(x, dict):
            return SimpleNamespace(**{k: build(v) for k, v in x.items()})
        return x

    return build(root)
