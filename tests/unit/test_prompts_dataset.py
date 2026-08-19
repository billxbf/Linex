# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from molt.datasets.prompts_dataset import PromptDataset


def _dataset(rows, task_key="task"):
    strategy = SimpleNamespace(args=SimpleNamespace(data=SimpleNamespace(input_key="prompt", task_key=task_key)))
    return PromptDataset(rows, strategy)


def test_dataset_carries_only_instruction_identity_and_task_spec():
    task = {
        "runtime": {"image": "task:latest"},
        "agent": {"harness": "codex"},
        "evaluator": {"strategy": "session_completed"},
    }
    dataset = _dataset(
        [
            {
                "datasource": "code",
                "prompt": "Fix the program",
                "task": task,
                "tools": ["ignored"],
                "images": ["ignored.png"],
                "reward_model": {"ground_truth": "ignored"},
            }
        ]
    )

    assert dataset[0] == ("code", "Fix the program", task)
    assert dataset.collate_fn([dataset[0]]) == (["code"], ["Fix the program"], [task])


def test_dataset_supports_configured_task_column():
    task = {"runtime": {"image": "task:latest"}}
    assert _dataset([{"prompt": "Do it", "polar": task}], task_key="polar")[0][2] == task


def test_dataset_rejects_message_and_preprocessed_prompt_shapes():
    with pytest.raises(TypeError, match="instructions must be strings"):
        _dataset([{"prompt": [{"role": "user", "content": "Do it"}]}])[0]
