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

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, List

import ray
import torch

from molt.models.utils import compute_approx_kl, masked_mean
from molt.trainer.algorithm.advantage import (
    GROUP_ADVANTAGE_ESTIMATORS,
    AdvantageContext,
    get_advantage_estimator,
    group_reward_moments,
)
from molt.trainer.algorithm.experience import Experience
from molt.utils.logging_utils import init_logger

if TYPE_CHECKING:
    from molt.trainer.workers.actor_group import RayActorGroup

logger = init_logger(__name__)


def rollout_and_group_ids(experience):
    """Per-sample rollout and prompt-group ids, with the fallbacks the pipeline shares.

    Multi-turn agents stamp both. Legacy single-turn rollouts stamp neither, and there every
    sample is its own rollout and its own group, so the sample index serves as both. Kept in one
    place because every consumer that averages per rollout has to agree on it.
    """
    rollout_ids = experience.rollout_ids or list(experience.index)
    return rollout_ids, (experience.group_ids or rollout_ids)


class RemoteExperienceMaker:
    """Builds train-ready experiences from rollout samples and remote model forwards."""

    def __init__(
        self,
        actor_model_group: RayActorGroup,
        initial_model_group: RayActorGroup,
        kl_controller,
        strategy,
        tokenizer,
        critic_model_group: RayActorGroup = None,
        **kwargs,
    ):
        super().__init__()

        self.strategy = strategy
        self.args = strategy.args
        self.advantage_estimator = strategy.args.algo.advantage.estimator

        self.actor_model_group = actor_model_group
        self.initial_model_group = initial_model_group
        self.critic_model_group = critic_model_group
        self.tokenizer = tokenizer
        self.kl_ctl = kl_controller

    def build_experiences(self, rollout_samples: List[Experience]) -> List[Experience]:
        """Turn balanced rollout samples into train-ready experiences: recompute the log-probs and
        values the loss needs (make_experience), then the advantages and returns."""
        experiences = self.make_experience(rollout_samples)
        return self.compute_advantages_and_returns(experiences)

    @torch.no_grad()
    def make_experience(self, experiences: List[Experience]) -> List[Experience]:
        """Recompute the log-probs and values the policy loss needs. Each model's forward runs on
        its DP ranks, which fetch their samples with Experience.reload() — so the heavy tensors stay
        in shared memory and never reach the controller. Attaches values (GAE critic),
        base_action_log_probs (KL recipes), old action_log_probs, and the per-token kl, ready for
        advantage estimation."""
        args = self.args
        if self.critic_model_group is not None:
            self._dispatch_forward(experiences, self.critic_model_group, "values")
        if self.initial_model_group is not None:
            self._dispatch_forward(experiences, self.initial_model_group, "base_action_log_probs")

        # DPPO uses stored rollout probabilities; a single-update PPO batch uses its training
        # forward. Neither needs an old-policy pass unless KL rewards consume those log-probs.
        skip_actor_old = args.algo.kl.init_coef == 0 and (args.train.force_on_policy or args.actor.loss_mode == "dppo")
        if not skip_actor_old:
            self._dispatch_forward(experiences, self.actor_model_group, "action_log_probs")

        for i, experience in enumerate(experiences):
            experience.index = [i]
            # With KL kept out of the loss, the advantage receives the per-token KL reward.
            # With KL in the loss
            # (or no ref) the advantage sees no KL reward, so kl stays zero.
            if (
                self.initial_model_group is not None
                and not args.algo.kl.use_loss
                and experience.action_log_probs is not None
            ):
                experience.kl = compute_approx_kl(
                    experience.action_log_probs,
                    experience.base_action_log_probs,
                    kl_estimator=args.algo.kl.estimator,
                )
                logprobs_diff = experience.action_log_probs.float() - experience.base_action_log_probs.float()
            else:
                experience.kl = torch.zeros_like(experience.action_mask, dtype=torch.float32)
                logprobs_diff = torch.zeros_like(experience.action_mask, dtype=torch.float32)
            experience.info["kl"] = masked_mean(experience.kl, experience.action_mask, dim=-1)
            experience.info["logprobs_diff"] = masked_mean(logprobs_diff, experience.action_mask, dim=-1)
            # With KL as a reward (or no ref) the loss needs no separate KL term, so drop the ref
            # log-probs; the KL-in-loss path keeps them for policy_train's loss.
            if not args.algo.kl.use_loss:
                experience.base_action_log_probs = None
        return experiences

    def _dispatch_forward(self, experiences: List[Experience], group: "RayActorGroup", result_attr: str) -> None:
        """Run ``group``'s forward on every sample — distributed across its DP ranks, each reloading its
        own heavy tensors so they never reach the controller — and store the per-sample result on the
        Experience under ``result_attr`` ("values" / "base_action_log_probs" / "action_log_probs"). Every
        cp/tp rank in a DP group returned the same per-sample results, so keep one copy per group (drop
        the duplicates) and flatten to one result per sample, in ``experiences`` order. Frees the
        forward's cache before the colocated actor trains on the same GPUs."""
        refs = group.async_run_method_batch(method_name="forward", experience=experiences)
        outputs = list(itertools.chain.from_iterable(ray.get(refs)[:: group.duplicate_actors]))
        for experience, output in zip(experiences, outputs):
            setattr(experience, result_attr, output)
        ray.get(group.async_run_method(method_name="empty_cache"))

    # Advantage and return computation

    @torch.no_grad()
    def compute_advantages_and_returns(self, experiences: List[Experience]) -> List[Experience]:
        """Keep per-trace rewards, compute group statistics, and assemble token advantages."""
        args = self.args
        rewards = torch.cat([exp.rewards for exp in experiences])
        exp_len = [exp.rewards.numel() for exp in experiences]
        ids = [rollout_and_group_ids(exp) for exp in experiences]
        rollout_ids = list(itertools.chain.from_iterable(r for r, _ in ids))
        group_ids = list(itertools.chain.from_iterable(g for _, g in ids))
        if not (len(rollout_ids) == len(group_ids) == rewards.numel()):
            raise ValueError("id/reward length mismatch")

        prompt_groups = {}
        for index, (rid, gid) in enumerate(zip(rollout_ids, group_ids)):
            prompt_groups.setdefault(gid, {}).setdefault(rid, []).append(index)
        groups = [list(rollouts.values()) for rollouts in prompt_groups.values()]
        if self.advantage_estimator not in GROUP_ADVANTAGE_ESTIMATORS:
            groups = [[[index]] for index in range(rewards.numel())]
        counts = torch.cat([exp.action_mask.sum(dim=-1) for exp in experiences]).to(rewards)
        weights = torch.zeros_like(rewards)
        for group in groups:
            for rollout in group:
                weights[rollout] = counts[rollout] / counts[rollout].sum().clamp_min(1)

        clip = args.reward.clip_range
        if clip:
            rewards = rewards.clamp(min=clip[0], max=clip[1])

        # PPO/gae is the only estimator that consumes a learned value baseline; the
        # critic filled exp.values during make_experience. Other estimators ignore it.
        needs_values = self.advantage_estimator == "gae"
        ctx = AdvantageContext(
            trace_weights=weights,
            exp_len=exp_len,
            action_masks=[exp.action_mask for exp in experiences],
            kl_coef=self.kl_ctl.value,
            gamma=args.algo.advantage.gamma,
            kls=[exp.kl for exp in experiences],
            lam=args.algo.advantage.lam,
            values=[exp.values for exp in experiences] if needs_values else None,
            no_whiten=args.algo.advantage.no_whiten,
        )
        advantages, returns = get_advantage_estimator(self.advantage_estimator)(rewards, groups, ctx)

        trace_stds = torch.zeros_like(rewards)
        for group in groups:
            _, std = group_reward_moments(rewards, group, weights)
            for rollout in group:
                trace_stds[rollout] = std
        sample_stds = trace_stds.split(exp_len)

        # Assemble the experiences from the computed tensors.
        for exp, adv, ret, std in zip(experiences, advantages, returns, sample_stds):
            exp.advantages = adv
            exp.returns = ret
            exp.info["return"] = masked_mean(ret, exp.action_mask, dim=-1)
            if args.rollout.n_samples_per_prompt > 1:
                exp.info["group_reward_std"] = std
            exp.kl = None

        return experiences
