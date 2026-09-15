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

import sys
import types
from collections import defaultdict
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from PIL import Image

if "ray" not in sys.modules:
    fake_ray = types.ModuleType("ray")

    def remote(*args, **kwargs):
        if args and len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]

        def decorator(obj):
            return obj

        return decorator

    fake_ray.remote = remote
    fake_ray.put = MagicMock()
    fake_ray.get = MagicMock()
    fake_ray.wait = MagicMock()
    fake_ray.cancel = MagicMock()
    fake_util = types.ModuleType("ray.util")
    fake_placement_group = types.ModuleType("ray.util.placement_group")
    fake_placement_group.PlacementGroup = type("PlacementGroup", (), {})
    fake_placement_group.placement_group = MagicMock()
    fake_util.placement_group = fake_placement_group
    fake_scheduling = types.ModuleType("ray.util.scheduling_strategies")
    fake_scheduling.PlacementGroupSchedulingStrategy = type("PlacementGroupSchedulingStrategy", (), {})
    fake_util.scheduling_strategies = fake_scheduling
    fake_ray.util = fake_util
    sys.modules["ray"] = fake_ray
    sys.modules["ray.util"] = fake_util
    sys.modules["ray.util.placement_group"] = fake_placement_group
    sys.modules["ray.util.scheduling_strategies"] = fake_scheduling
from molt.trainer.rollout import samples_generator
from molt.trainer.rollout.samples_generator import SamplesGenerator
from polar.rollout.models import TaskSpec


def _sample(group_id, **fields):
    fields.setdefault("rollout_ids", [group_id])
    sample = SimpleNamespace(group_ids=[group_id], **fields)
    sample.offload = lambda: sample
    return sample


def _prompt_loader(num_prompts):
    """A dataloader yielding one Polar instruction and optional task per item."""
    return [([f"d{i}"], [f"p{i}"], [None]) for i in range(num_prompts)]


def _wire_fake_vllm(generator, monkeypatch, to_sample):
    """Wire the streaming generator to an in-memory vLLM that finishes rollouts FIFO.

    Each dispatched prompt becomes one in-flight rollout handle tagged with its prompt
    string; ray.wait hands them back in dispatch order and ray.get turns a handle into
    the producer result — a list of ``(light, drop_reason)`` (here one usable light that
    ``to_sample`` maps from the prompt tag).
    """
    generator._dispatch_rollouts = lambda prompts, **kw: [SimpleNamespace(group_id=prompt) for prompt in prompts]
    generator._process_polar_task_result = lambda result, _max_length: result
    generator.args.data = SimpleNamespace(max_len=2048)
    monkeypatch.setattr(
        samples_generator.ray, "wait", lambda handles, num_returns=1: ([handles[0]], list(handles[1:]))
    )
    monkeypatch.setattr(samples_generator.ray, "get", lambda handle: [(to_sample(handle.group_id), None)])


def test_generate_samples_returns_batch_as_rollouts_finish_and_keeps_pool_saturated(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    samples, rollout_metrics, prompts_dispatched, exhausted = generator.generate_samples()

    # Returns exactly batch_size (3) finished groups, in completion (= dispatch) order.
    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1", "p2"]
    # Pool stays saturated: 5 dispatched up front, then one refill per completion →
    # 7 dispatched total, the 4 unclaimed rollouts stay in flight for the next call.
    assert prompts_dispatched == 7
    assert [handle.group_id for handle in generator._inflight_rollouts] == ["p3", "p4", "p5", "p6"]
    assert generator._finished_samples == []
    # No drops and no dynamic filtering → no rollout metrics emitted.
    assert rollout_metrics == {}
    assert exhausted is False


def test_generate_samples_emits_short_batch_when_dataloader_exhausted(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=4, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(2)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    samples, _, prompts_dispatched, exhausted = generator.generate_samples()

    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1"]
    assert prompts_dispatched == 2
    assert exhausted is True
    assert generator._inflight_rollouts == []


@pytest.mark.parametrize("force_on_policy", [False, True])
def test_generate_samples_drains_last_inflight_groups_before_next_episode(monkeypatch, force_on_policy):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
        train=SimpleNamespace(force_on_policy=force_on_policy),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)
    seen = []
    for _ in range(4):
        samples, metrics, _, exhausted = generator.generate_samples()
        seen.extend(sample.group_ids[0] for sample in samples)
    assert exhausted
    assert seen == [f"p{i}" for i in range(9 if force_on_policy else 10)]
    if force_on_policy:
        assert metrics["rollout/dropped/incomplete_batch"] == 1


def _wire_eval_generator(generator, monkeypatch, num_prompts):
    generator.eval_dataloader = _prompt_loader(num_prompts)
    dispatch_sizes = []

    def dispatch(prompts, **kwargs):
        dispatch_sizes.append(len(prompts))
        return [SimpleNamespace(group_id=prompt) for prompt in prompts]

    generator._dispatch_rollouts = dispatch
    generator._process_polar_task_result = lambda result, _max_length: result
    generator.args.data = SimpleNamespace(max_len=2048)
    monkeypatch.setattr(
        samples_generator.ray, "wait", lambda handles, num_returns=1, timeout=None: ([handles[0]], list(handles[1:]))
    )
    monkeypatch.setattr(samples_generator.ray, "get", lambda handle: [(_sample(handle.group_id), None)])
    return dispatch_sizes


def test_generate_eval_samples_uses_independent_eval_batch_size(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=1, n_samples_per_prompt=1),
        eval=SimpleNamespace(batch_size=4),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
    )
    dispatch_sizes = _wire_eval_generator(generator, monkeypatch, num_prompts=5)

    samples = generator.generate_eval_samples()

    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1", "p2", "p3", "p4"]
    assert dispatch_sizes == [4, 1]


def test_generate_eval_samples_defaults_to_rollout_batch_size_and_refills_each_slot(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=2, n_samples_per_prompt=1),
        eval=SimpleNamespace(batch_size=None),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
    )
    dispatch_sizes = _wire_eval_generator(generator, monkeypatch, num_prompts=5)

    samples = generator.generate_eval_samples()

    assert len(samples) == 5
    assert dispatch_sizes == [2, 1, 1, 1]


def test_generate_samples_pool_persists_across_calls(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    first, *_ = generator.generate_samples()
    second, _, prompts_dispatched, _ = generator.generate_samples()

    assert [sample.group_ids[0] for sample in first] == ["p0", "p1", "p2"]
    # The second batch is served from rollouts already in flight after the first call
    # (p3-p5) — vLLM never drained between steps — and the pool is topped back up.
    assert [sample.group_ids[0] for sample in second] == ["p3", "p4", "p5"]
    assert prompts_dispatched == 3  # only the 3 refills, not a fresh batch of 5
    assert [handle.group_id for handle in generator._inflight_rollouts] == ["p6", "p7", "p8", "p9"]


def test_force_sync_mode_leaves_no_rollout_in_flight(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
        train=SimpleNamespace(force_sync_mode=True),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    samples, _, prompts_dispatched, exhausted = generator.generate_samples()

    assert [sample.group_ids[0] for sample in samples] == ["p0", "p1", "p2"]
    assert prompts_dispatched == 3
    assert generator._inflight_rollouts == []
    assert exhausted is False


def test_generator_keeps_no_checkpoint_state_and_resumes_from_dataloader(monkeypatch):
    """The in-flight pool is intentionally NOT persisted.

    Persisting in-flight task payloads bloated checkpoints
    ~1000x (22-78 MB vs ~7 KB) and crashed the driver on resume, so the generator
    is stateless across checkpoints: state_dict() is empty and load_state_dict is a
    no-op tolerant of None/{}. The StatefulDataLoader cursor already points past the
    in-flight prefetch, so on resume those few prompts are skipped (a bounded loss,
    negligible for multi-epoch RL) rather than redispatched.
    """
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=3, n_samples_per_prompt=1, vllm_generate_batch_size=5),
        algo=SimpleNamespace(dynamic_filtering_enable=False),
        ckpt=SimpleNamespace(warm_resume_rollouts=False),
    )
    generator.prompts_dataloader = _prompt_loader(10)
    _wire_fake_vllm(generator, monkeypatch, _sample)

    first, *_ = generator.generate_samples()
    assert [sample.group_ids[0] for sample in first] == ["p0", "p1", "p2"]
    # p3-p6 are in flight at the checkpoint; the generator carries no state for them.
    assert generator.state_dict() == {}

    restored = object.__new__(SamplesGenerator)
    restored.args = generator.args
    # The StatefulDataLoader cursor in the checkpoint already points past the
    # prefetched prompts (p0-p6 were read), so resume starts at p7.
    restored.prompts_dataloader = [([f"d{i}"], [f"p{i}"], [None]) for i in range(7, 10)]
    _wire_fake_vllm(restored, monkeypatch, _sample)
    restored.load_state_dict(None)  # tolerate a missing payload
    restored.load_state_dict({})  # and an empty one

    second, _, newly_dispatched, _ = restored.generate_samples()
    # Resume continues from the dataloader cursor; the in-flight p3-p6 are not retrained.
    assert [sample.group_ids[0] for sample in second] == ["p7", "p8", "p9"]
    assert newly_dispatched == 3


def test_generate_samples_drops_filtered_groups_and_refills_their_slots(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(batch_size=2, n_samples_per_prompt=1, vllm_generate_batch_size=2),
        algo=SimpleNamespace(dynamic_filtering_enable=True, dynamic_filtering_range=(0.0, 1.0)),
    )
    generator.prompts_dataloader = _prompt_loader(10)

    # p1's mean score (1.0) sits on the boundary of the open range (0, 1) → filtered out.
    group_score = {"p0": 0.5, "p1": 1.0, "p2": 0.5, "p3": 0.5}

    def scored_sample(group_id):
        return _sample(group_id, scores=[torch.tensor(group_score[group_id])])

    _wire_fake_vllm(generator, monkeypatch, scored_sample)

    samples, rollout_metrics, prompts_dispatched, _ = generator.generate_samples()

    # p1 is dropped; p2 (refilled into p1's freed slot) completes the batch.
    assert [sample.group_ids[0] for sample in samples] == ["p0", "p2"]
    assert prompts_dispatched == 4  # p0,p1 up front; p2,p3 refilled one per completion
    assert rollout_metrics["dynamic_filtering_pass_rate"] == 2 / 3 * 100
    # The filtered group is tallied by reason for observability.
    assert rollout_metrics["rollout/dropped/dynamic_filter"] == 1.0
    assert rollout_metrics["rollout/dropped/total"] == 1.0


def test_dynamic_filtering_counts_compaction_segments_once_per_rollout(monkeypatch):
    generator = object.__new__(SamplesGenerator)
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(n_samples_per_prompt=2),
        algo=SimpleNamespace(dynamic_filtering_range=(0.4, 0.6)),
    )
    samples = [
        _sample("group", rollout_ids=[rollout_id], scores=torch.tensor([score]))
        for rollout_id, score in [("a", 1.0), ("a", 1.0), ("a", 1.0), ("b", 0.0)]
    ]
    monkeypatch.setattr(samples_generator.ray, "get", lambda _: [(sample, None) for sample in samples])
    generator._process_polar_task_result = lambda result, _max_length: result
    generator.args.data = SimpleNamespace(max_len=2048)
    score_stats = defaultdict(float)

    kept = generator._filter_group(object(), True, defaultdict(int), score_stats=score_stats)

    assert kept == samples
    assert dict(score_stats) == {"score_sum": 1.0, "score_n": 2, "groups": 1.0, "all_pass": 0.0, "all_fail": 0.0}


def _polar_task_result(traces, *, session_id="session-1"):
    return {
        "task_id": "task-1",
        "instruction": "Fix the program",
        "results": [
            {
                "session_id": session_id,
                "task_id": "task-1",
                "status": "COMPLETED",
                "trajectory": {"status": "COMPLETED", "traces": traces},
                "timing": {"run_ms": 15.0},
            }
        ],
    }


def _converter(tokenizer=None):
    generator = object.__new__(SamplesGenerator)
    generator.tokenizer = tokenizer or SimpleNamespace()
    return generator


def test_polar_single_turn_trace_becomes_exact_experience():
    result = _polar_task_result(
        [
            {
                "prompt_ids": [10, 11],
                "response_ids": [20, 21],
                "loss_mask": [1, 1],
                "response_logprobs": [-0.1, -0.2],
                "reward": 0.75,
                "finish_reason": "length",
            }
        ]
    )

    [(experience, drop_reason)] = _converter()._process_polar_task_result(result, max_length=16)

    assert drop_reason is None
    torch.testing.assert_close(experience.sequences, torch.tensor([[10, 11, 20, 21]]))
    torch.testing.assert_close(experience.action_mask, torch.tensor([[False, True, True]]))
    torch.testing.assert_close(experience.rollout_log_probs, torch.tensor([[0.0, -0.1, -0.2]]))
    assert experience.group_ids == ["task-1"]
    assert experience.rollout_ids == ["session-1"]
    assert experience.prompts == ["Fix the program"]
    assert experience.rewards.item() == pytest.approx(0.75)
    assert experience.scores.item() == pytest.approx(0.75)
    assert experience.response_length.item() == 2
    assert experience.truncated.item() is True
    assert experience.info["polar/run_ms"].item() == pytest.approx(15.0)


def test_polar_trace_preserves_safe_molt_truncation_behavior():
    result = _polar_task_result(
        [
            {
                "prompt_ids": [1, 2],
                "response_ids": [3, 4, 5],
                "loss_mask": [1, 1, 1],
                "response_logprobs": [-0.1, -0.2, -0.3],
                "reward": 1.0,
            }
        ]
    )

    [(experience, drop_reason)] = _converter()._process_polar_task_result(result, max_length=4)

    assert drop_reason is None
    assert experience.sequences.tolist() == [[1, 2, 3, 4]]
    assert experience.response_length.item() == 2
    assert experience.truncated.item() is True
    assert experience.info["response_clip_ratio"].item() is True


def test_polar_vlm_trace_rejects_truncation_that_drops_media():
    result = _polar_task_result(
        [
            {
                "prompt_ids": [1, 2, 9],
                "response_ids": [3],
                "loss_mask": [1],
                "response_logprobs": [-0.1],
                "reward": 1.0,
                "media_paths": ["unused.png"],
            }
        ]
    )

    assert _converter(SimpleNamespace(image_token_id=9))._process_polar_task_result(result, max_length=2) == [
        (None, "vlm_truncation")
    ]


def test_polar_media_artifact_becomes_aligned_training_input(tmp_path):
    image_path = tmp_path / "image.png"
    Image.new("RGB", (4, 4), "yellow").save(image_path)

    class ImageProcessor:
        merge_size = 2

        def __call__(self, *, images, return_tensors):
            assert len(images) == 1 and return_tensors == "pt"
            return {
                "pixel_values": torch.ones(1, 3, 4, 4),
                "image_grid_thw": torch.tensor([[1, 2, 2]]),
            }

    tokenizer = SimpleNamespace(image_processor=ImageProcessor(), image_token_id=9)
    result = _polar_task_result(
        [
            {
                "prompt_ids": [1, 9],
                "response_ids": [2, 3],
                "loss_mask": [1, 1],
                "response_logprobs": [-0.1, -0.2],
                "reward": 1.0,
                "media_paths": [str(image_path)],
            }
        ]
    )

    [(experience, drop_reason)] = _converter(tokenizer)._process_polar_task_result(result, max_length=16)

    assert drop_reason is None
    torch.testing.assert_close(experience.mm_train_inputs[0]["pixel_values"], torch.ones(1, 3, 4, 4))
    assert experience.info["image_tokens"].item() == 1


def test_polar_multiturn_tool_result_tokens_stay_masked():
    result = _polar_task_result(
        [
            {
                "prompt_ids": [1, 2],
                "response_ids": [3, 4, 50, 5],
                "loss_mask": [1, 1, 0, 1],
                "response_logprobs": [-0.1, -0.2, 0.0, -0.3],
                "reward": 1.0,
            }
        ]
    )

    [(experience, drop_reason)] = _converter()._process_polar_task_result(result, max_length=16)

    assert drop_reason is None
    torch.testing.assert_close(experience.sequences, torch.tensor([[1, 2, 3, 4, 50, 5]]))
    torch.testing.assert_close(experience.action_mask, torch.tensor([[False, True, True, False, True]]))
    torch.testing.assert_close(
        experience.rollout_log_probs,
        torch.tensor([[0.0, -0.1, -0.2, 0.0, -0.3]]),
    )
    assert experience.response_length.item() == 3


def test_polar_multi_trace_session_shares_rollout_identity():
    traces = [
        {
            "prompt_ids": [1],
            "response_ids": [token],
            "loss_mask": [1],
            "response_logprobs": [-0.1],
            "reward": reward,
        }
        for token, reward in ((2, 1.0), (3, 0.5))
    ]

    converted = _converter()._process_polar_task_result(_polar_task_result(traces), max_length=16)
    experiences = [experience for experience, drop_reason in converted if drop_reason is None]

    assert len(experiences) == 2
    assert {experience.group_ids[0] for experience in experiences} == {"task-1"}
    assert {experience.rollout_ids[0] for experience in experiences} == {"session-1"}


def test_polar_keeps_harbor_and_judge_rewards_aligned_when_an_earlier_trace_is_dropped():
    traces = [
        {"prompt_ids": prompt, "response_ids": [2], "loss_mask": [1], "response_logprobs": [-0.1], "reward": reward}
        for prompt, reward in [([], 1.2), ([1], 0.8), ([1], 1.0)]
    ]
    result = _polar_task_result(traces)
    result["results"][0]["trajectory"]["metadata"] = {"evaluation": {
        "mode": "harbor_rubric", "outcome_reward": 1.0, "judge_scores": [5, -5, None],
    }}
    converted = _converter()._process_polar_task_result(result, max_length=16)
    assert converted[0] == (None, "empty_prompt_tokens")
    for (experience, reason), reward, judge in zip(converted[1:], [0.8, 1.0], [-1.0, 0.0]):
        assert reason is None
        assert experience.info["harbor_reward"].item() == 1.0
        assert experience.info["judge_reward"].item() == judge
        assert experience.rewards.item() == pytest.approx(reward)


def test_polar_conversion_rejects_task_identity_mismatch():
    result = _polar_task_result([])
    result["results"][0]["task_id"] = "other-task"

    assert _converter()._process_polar_task_result(result, max_length=16) == [(None, "identity_mismatch")]


def test_polar_conversion_keeps_rewarded_trace_from_failed_session():
    result = _polar_task_result(
        [
            {
                "prompt_ids": [1],
                "response_ids": [2],
                "loss_mask": [1],
                "response_logprobs": [-0.1],
                "reward": 0.0,
            }
        ]
    )
    result["results"][0].update({"status": "ERROR", "error": "agent exited 1"})

    [(experience, drop_reason)] = _converter()._process_polar_task_result(result, max_length=16)

    assert drop_reason is None
    assert experience.rewards.item() == 0.0


def test_polar_dispatch_uses_cli_sampling_and_task_spec():
    payloads = []

    class RunTask:
        def remote(self, payload):
            payloads.append(payload)
            return "task-ref"

    generator = object.__new__(SamplesGenerator)
    generator.polar_rollout = SimpleNamespace(run_task=RunTask())
    generator.task_spec = TaskSpec.model_validate(
        {
            "runtime": {"image": "calculator:latest"},
            "agent": {"harness": "codex"},
            "evaluator": {"strategy": "session_completed"},
            "metadata": {"recipe": "calculator"},
        }
    )
    generator.args = SimpleNamespace(
        rollout=SimpleNamespace(n_samples_per_prompt=4, session_timeout=90.0),
        data=SimpleNamespace(max_len=2048),
    )

    refs = generator._dispatch_rollouts(
        ["Fix it"],
        task_specs=[None],
        max_new_tokens=128,
        temperature=0.7,
        top_p=0.95,
    )

    assert refs == ["task-ref"]
    request = payloads[0]
    assert request["instruction"] == "Fix it"
    assert request["num_samples"] == 4
    assert request["timeout_seconds"] == 90.0
    assert request["runtime"]["image"] == "calculator:latest"
    assert request["metadata"] == {"recipe": "calculator"}
    assert request["sampling_params"]["max_tokens"] == 128
    assert request["sampling_params"]["max_total_tokens"] == 2048
    assert request["sampling_params"]["temperature"] == 0.7
    assert request["sampling_params"]["top_p"] == 0.95
    assert request["sampling_params"]["top_k"] == -1
    assert request["sampling_params"]["seed"] is None
    assert request["sampling_params"]["logprobs"] == 1

    with pytest.raises(ValueError, match="either --rollout.task_spec or row-level"):
        generator._dispatch_rollouts(
            ["Fix it"],
            task_specs=[generator.task_spec.model_dump(mode="json")],
        )


def test_polar_task_spec_rejects_molt_owned_sampling():
    with pytest.raises(ValueError, match="sampling_params"):
        TaskSpec.model_validate(
            {
                "runtime": {"image": "calculator:latest"},
                "agent": {"harness": "codex"},
                "evaluator": {"strategy": "session_completed"},
                "sampling_params": {"temperature": 0.5},
            }
        )


def test_task_spec_resolves_upload_sources_from_validation_context(tmp_path):
    task = TaskSpec.model_validate(
        {
            "runtime": {
                "image": "calculator:latest",
                "prepare": [{"type": "upload_file", "source": "assets/input.py", "target": "/input.py"}],
                "eval_prepare": [{"type": "upload_dir", "source": "assets/tests", "target": "/tests"}],
            },
            "agent": {"harness": "codex"},
            "evaluator": {"strategy": "session_completed"},
        },
        context={"base_dir": tmp_path},
    )

    assert task.runtime.prepare[0].source == str(tmp_path / "assets/input.py")
    assert task.runtime.eval_prepare[0].source == str(tmp_path / "assets/tests")


def test_warm_resume_state_dict_materializes_lazy_samples(tmp_path, monkeypatch):
    """Under distributed rollout every finished sample is offloaded (heavy_ref set). state_dict()
    must MATERIALIZE a local copy (copy + reload) so the warm buffer survives an actor restart,
    while leaving the originals lazy so the current step still ships them cheaply."""
    import os

    from molt.trainer.algorithm import experience as exp_mod
    from molt.trainer.algorithm.experience import Experience

    # Fake Ray object store so offload()/reload() work without a live Ray.
    store, counter = {}, {"n": 0}

    def fake_put(obj):
        counter["n"] += 1
        key = f"ref{counter['n']}"
        store[key] = obj
        return key

    monkeypatch.setattr(exp_mod.ray, "put", fake_put)
    monkeypatch.setattr(exp_mod.ray, "get", lambda ref: store[ref])

    # A finished-but-untrained sample, offloaded (lazy) exactly as the generator leaves it.
    e = Experience(
        sequences=torch.tensor([[1, 2, 3]]),
        attention_mask=torch.tensor([[1, 1, 1]]),
        rewards=torch.tensor([1.0]),
        group_ids=["g0"],
        rollout_ids=["r0"],
    )
    e.offload()
    assert e.heavy_ref is not None and e.sequences is None  # lazy: heavy tensors in the store

    gen = object.__new__(SamplesGenerator)
    gen.args = SimpleNamespace(ckpt=SimpleNamespace(warm_resume_rollouts=True, path=str(tmp_path / "ckpt")))
    gen._finished_samples = [e]

    sd = gen.state_dict()
    assert sd.get("buffer_file") and os.path.exists(sd["buffer_file"])
    # Original stays lazy — materialization happened on a copy, not in place.
    assert e.heavy_ref is not None and e.sequences is None

    # A fresh generator (post-restart) restores the untrained tail as fully-local Experiences.
    gen2 = object.__new__(SamplesGenerator)
    gen2.load_state_dict(sd)
    restored = gen2._resumed_samples
    assert len(restored) == 1
    assert restored[0].heavy_ref is None  # local, not a dead handle
    assert torch.equal(restored[0].sequences, torch.tensor([[1, 2, 3]]))
    assert restored[0].group_ids == ["g0"]


def test_warm_resume_state_dict_noop_when_disabled(tmp_path):
    """The flag gates the whole feature: with warm_resume_rollouts off, no file is written."""
    gen = object.__new__(SamplesGenerator)
    gen.args = SimpleNamespace(ckpt=SimpleNamespace(warm_resume_rollouts=False, path=str(tmp_path / "ckpt")))
    gen._finished_samples = [SimpleNamespace(group_ids=["g0"], heavy_ref=None)]
    assert gen.state_dict() == {}
