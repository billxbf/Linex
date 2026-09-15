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

from types import SimpleNamespace

import pytest
import torch

import molt.trainer.algorithm.experience as experience_mod
from molt.trainer.algorithm.experience import (
    Experience,
    balance_experiences,
    get_model_parallel_size,
    make_experience_batch,
)
from molt.trainer.rollout.experience_maker import RemoteExperienceMaker


@pytest.mark.parametrize(
    "loss_mode, force_on_policy, kl_coef, expected_calls",
    [("dppo", False, 0.0, 0), ("ppo", True, 0.0, 0), ("ppo", False, 0.0, 1), ("dppo", False, 0.1, 1)],
)
def test_old_policy_forward_is_skipped_when_the_objective_does_not_need_it(
    loss_mode, force_on_policy, kl_coef, expected_calls
):
    calls = []
    maker = SimpleNamespace(
        args=SimpleNamespace(
            train=SimpleNamespace(force_on_policy=force_on_policy),
            actor=SimpleNamespace(loss_mode=loss_mode),
            algo=SimpleNamespace(kl=SimpleNamespace(init_coef=kl_coef, use_loss=False)),
        ),
        critic_model_group=None,
        initial_model_group=None,
        actor_model_group=object(),
        _dispatch_forward=lambda *args: calls.append(args),
    )
    exp = Experience(action_mask=torch.ones(1, 2, dtype=torch.bool), info={})
    RemoteExperienceMaker.make_experience(maker, [exp])
    assert len(calls) == expected_calls
    if calls:
        assert calls[0][2] == "action_log_probs"


def _args(cp=1, tp=1, ep=1, actor_gpus=1):
    return SimpleNamespace(
        actor=SimpleNamespace(num_nodes=1, num_gpus_per_node=actor_gpus),
        fsdp=SimpleNamespace(cp_size=cp, tp_size=tp, ep_size=ep),
    )


def test_model_parallel_size_excludes_ep():
    assert get_model_parallel_size(_args(cp=2, tp=3, ep=4)) == 6


def _len_sample(length, idx):
    return Experience(total_length=torch.tensor([length]), rollout_ids=[f"r{idx}"])


def test_balance_experiences_excludes_ep_and_returns_flat_samples():
    # DP size = actor_gpus // (cp*tp) = 4 (EP is excluded from the divisor). 6 per-sample
    # experiences -> keep (6 // 4) * 4 = 4 (drop the 2-sample remainder). If EP were wrongly in
    # the divisor, dp would be 2 and all 6 would be kept.
    samples = [_len_sample(6 - i, i) for i in range(6)]
    balanced = balance_experiences(samples, _args(ep=2, actor_gpus=4))

    assert len(balanced) == 4  # flat, equal count per rank (1×4), remainder dropped — not concatenated
    ids = {id(s) for s in samples}
    assert all(id(b) in ids for b in balanced)  # returns the input samples themselves (no concat)


def test_balance_experiences_equal_count_and_length_sorted():
    # 10 samples across 4 DP ranks: the 2-sample remainder is dropped so every rank receives the
    # SAME count (2) — unequal counts desync num_steps and deadlock the world all_reduce. The flat
    # result is contiguous per-rank blocks; within each block samples are sorted by length descending
    # (k-th microbatch size-matched across ranks so no straggler trips the NCCL watchdog).
    samples = [_len_sample(10 - i, i) for i in range(10)]
    balanced = balance_experiences(samples, _args(actor_gpus=4))

    assert len(balanced) == 8  # 4 ranks × 2 samples; trailing 2-sample remainder dropped
    assert len({id(b) for b in balanced}) == 8  # no duplicates
    for rank in range(4):
        block = balanced[rank * 2 : rank * 2 + 2]
        block_lengths = [int(b.total_length.item()) for b in block]
        assert block_lengths == sorted(block_lengths, reverse=True)


def _grpo_maker():
    """A duck-typed RemoteExperienceMaker self with just the fields compute_advantages reads."""
    maker = SimpleNamespace(
        advantage_estimator="grpo",
        kl_ctl=SimpleNamespace(value=0.0),
        args=SimpleNamespace(
            reward=SimpleNamespace(clip_range=None),
            algo=SimpleNamespace(advantage=SimpleNamespace(gamma=1.0, lam=1.0, no_whiten=False)),
            rollout=SimpleNamespace(n_samples_per_prompt=4),
        ),
    )
    maker.compute_advantages_and_returns = RemoteExperienceMaker.compute_advantages_and_returns.__get__(maker)
    return maker


def _sample(idx, group_id, reward, length=6):
    # Equal length so the concatenated path stacks without padding — the two paths are
    # then directly comparable row-by-row.
    return Experience(
        action_mask=torch.ones(1, length, dtype=torch.bool),
        kl=torch.zeros(1, length),
        rewards=torch.tensor([float(reward)]),
        index=[idx],
        group_ids=[group_id],
        rollout_ids=[f"r{idx}"],
        info={},
    )


def test_distributed_advantages_match_materialized():
    # The distributed path runs compute_advantages_and_returns on per-sample "light"
    # Experiences; the materialized path runs it on the concatenated batch. Advantages are
    # a function of (reward, group, mask) only, so the per-sample results must be identical
    # — this is what lets the trainer skip gathering the heavy batch on the controller.
    rewards = [1.0, 0.0, 0.5, 0.25, 0.9, 0.1, 0.4, 0.6]
    groups = ["g0"] * 4 + ["g1"] * 4

    per_sample = [_sample(i, groups[i], rewards[i]) for i in range(8)]
    RemoteExperienceMaker.compute_advantages_and_returns(_grpo_maker(), per_sample)

    concat = make_experience_batch([_sample(i, groups[i], rewards[i]) for i in range(8)])
    RemoteExperienceMaker.compute_advantages_and_returns(_grpo_maker(), [concat])

    for i in range(8):
        assert torch.allclose(per_sample[i].advantages[0], concat.advantages[i])
        assert torch.allclose(per_sample[i].returns[0], concat.returns[i])


def test_grpo_g8_dppo_counts_split_rollouts_once():
    from molt.models import PolicyLoss

    samples = [_sample(i, "prompt", float(i == 0)) for i in range(8)]
    samples[0] = _sample(0, "prompt", 1.0, length=2)
    samples.insert(1, _sample(0, "prompt", 1.0, length=4))
    maker = _grpo_maker()
    maker.args.rollout.n_samples_per_prompt = 8
    maker.compute_advantages_and_returns(samples)
    token_count = sum(s.action_mask.sum() for s in samples)
    loss_fn = PolicyLoss(loss_mode="dppo")
    for sample in samples:
        success = sample.rewards.item() == 1.0
        expected_advantage = (0.875 if success else -0.125) / (0.125**0.5)
        torch.testing.assert_close(sample.advantages, torch.full_like(sample.advantages, expected_advantage))
        q = 0.3 if success else 0.25
        logp = torch.full_like(sample.advantages, q).log().requires_grad_()
        loss, *_ = loss_fn(
            logp,
            logp.detach(),
            sample.advantages,
            action_mask=sample.action_mask,
            rollout_log_probs=torch.full_like(logp, 0.2).log(),
            batch_num_tokens=token_count,
        )
        loss.backward()
        torch.testing.assert_close(logp.grad, -sample.advantages * (q / 0.2) / token_count)


@pytest.mark.parametrize("order", [(0, 1, 2), (2, 0, 1)])
@pytest.mark.parametrize("split", [False, True])
@pytest.mark.parametrize("estimator", ["grpo", "dr_grpo", "reinforce_baseline", "rloo"])
def test_per_trace_advantages_use_rollout_equal_token_weighted_statistics(order, split, estimator):
    samples = [_sample(0, "g", 1.2, length=1), _sample(0, "g", 0.8, length=4), _sample(1, "g", 0.0, length=2)]
    samples[1].action_mask[0, 1] = False
    samples = [samples[i] for i in order]
    if split:
        index = next(i for i, sample in enumerate(samples) if sample.action_mask.shape[-1] == 4)
        samples[index:index + 1] = [_sample(0, "g", 0.8, length=1), _sample(0, "g", 0.8, length=2)]
    maker = _grpo_maker()
    maker.advantage_estimator = estimator
    maker.compute_advantages_and_returns(samples)

    # Rollout means are 0.9 and 0; variance includes both deviations within the first rollout.
    for sample in samples:
        expected = sample.rewards.item() - 0.45
        if estimator == "grpo":
            expected /= 0.435**0.5
        elif estimator == "rloo":
            expected = sample.rewards.item() if sample.rollout_ids == ["r0"] else -0.9
        torch.testing.assert_close(sample.returns, expected * sample.action_mask.float())
        torch.testing.assert_close(sample.info["group_reward_std"], torch.tensor([0.435**0.5]))


def test_per_trace_grpo_retains_variation_when_rollout_means_match():
    samples = [_sample(0, "g", 0.2), _sample(0, "g", 0.8), _sample(1, "g", 0.5)]
    _grpo_maker().compute_advantages_and_returns(samples)
    for sample, expected in zip(samples, [-1.0, 1.0, 0.0]):
        torch.testing.assert_close(sample.advantages, torch.full_like(sample.advantages, expected))


def test_constant_rewards_remain_zero_with_unequal_trace_lengths():
    samples = [_sample(0, "g", 0.7, length=1), _sample(0, "g", 0.7, length=2), _sample(1, "g", 0.7)]
    _grpo_maker().compute_advantages_and_returns(samples)
    for sample in samples:
        assert torch.count_nonzero(sample.advantages) == 0


def test_experience_offload_reload_roundtrip(monkeypatch):
    # offload() moves the heavy fields into the object store (leaving a ref) and keeps the
    # lightweight ones in place; reload() restores them exactly. The controller only ever reads
    # the light fields, so a batch of these handles never fetches an image.
    store = {}

    def fake_put(obj):
        store[len(store)] = obj
        return len(store) - 1

    monkeypatch.setattr(experience_mod.ray, "put", fake_put)
    monkeypatch.setattr(experience_mod.ray, "get", lambda key: store[key])

    exp = Experience(
        sequences=torch.arange(6).view(1, 6),
        attention_mask=torch.ones(1, 6, dtype=torch.long),
        action_mask=torch.ones(1, 5, dtype=torch.bool),
        rewards=torch.tensor([1.0]),
        mm_train_inputs=[{"pixel_values": torch.zeros(2, 3)}],
    )
    exp.offload()
    assert exp.heavy_ref is not None
    assert exp.sequences is None and exp.attention_mask is None and exp.mm_train_inputs is None
    assert exp.action_mask is not None and exp.rewards is not None  # light fields untouched

    # offload() is idempotent: a second call must not re-put the now-nulled fields and clobber the
    # ref (the reload below would then restore Nones and lose the sample's tensors).
    first_ref = exp.heavy_ref
    exp.offload()
    assert exp.heavy_ref == first_ref

    exp.reload()
    assert exp.heavy_ref is None
    assert torch.equal(exp.sequences, torch.arange(6).view(1, 6))
    assert torch.equal(exp.mm_train_inputs[0]["pixel_values"], torch.zeros(2, 3))
    exp.reload()  # idempotent — second call is a no-op
