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

from torch.utils.data import Dataset


class PromptDataset(Dataset):
    """RL task instructions and optional per-row Polar task specifications."""

    def __init__(self, dataset, strategy) -> None:
        super().__init__()
        self.strategy = strategy
        self.dataset = dataset
        self.input_key = getattr(self.strategy.args.data, "input_key", None)
        self.task_key = getattr(self.strategy.args.data, "task_key", "task")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        instruction = data[self.input_key]
        if not isinstance(instruction, str):
            raise TypeError("Polar task instructions must be strings")
        return data.get("datasource", "default"), instruction, data.get(self.task_key)

    def collate_fn(self, item_list):
        datasources, instructions, tasks = zip(*item_list)
        return list(datasources), list(instructions), list(tasks)
