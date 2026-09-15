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

"""Pluggable advantage estimators.

An estimator turns per-trace rewards into per-token advantages and returns.
Prompt groups contain rollouts, and each rollout contains its trace indices:

    estimator(rewards, groups, ctx) -> (advantages, returns)
        rewards     (S,) tensor          one clipped scalar reward per trace
        groups      list[list[list[int]]]  prompt -> rollout -> trace indices
        ctx         AdvantageContext     sample/mask/kl/gamma tensors + flags
        advantages  list[(B, L) tensor]  per experience, ready for the policy loss
        returns     list[(B, L) tensor]  per experience (advantages before whitening)

Estimators call the helpers `expand_trace_advantages` (scalar advantage -> token-level
`advantage * action_mask`) and `normalize_advantages` (cross-batch whitening), both
operating on plain tensors — estimators never see `Experience`. The outcome-reward
estimators broadcast flat; `reinforce` is REINFORCE++ (per-token KL reward +
discounted cumulative returns). The trainer clips rewards, builds the context, and
assembles the results back onto the experiences.

Register a custom estimator without editing this file::

    @register_advantage_estimator("my_estimator")
    def my_estimator(rewards, groups, ctx):
        adv = rewards.clone()
        for group in groups:
            mean, _ = group_reward_moments(rewards, group, ctx.trace_weights)
            for rollout in group:
                adv[rollout] = rewards[rollout] - mean
        returns = expand_trace_advantages(adv, ctx)
        return normalize_advantages(returns, ctx), returns
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

import torch


@dataclass
class AdvantageContext:
    """Small tensor/scalar inputs an estimator needs (never the `Experience` objects)."""

    trace_weights: torch.Tensor  # (S,) action-token weights normalized within each rollout
    exp_len: List[int]  # samples per experience (to re-split per-sample tensors)
    action_masks: List[torch.Tensor]  # per experience (B, L)
    kl_coef: float  # per-token KL reward coefficient (REINFORCE++ / GAE)
    gamma: float  # discount factor (REINFORCE++ / GAE)
    lam: float  # GAE lambda (PPO); 1.0 = Monte-Carlo return minus the value baseline
    kls: List[torch.Tensor]  # per experience (B, L) per-token KL
    values: List[torch.Tensor] | None = None  # per experience (B, L) critic V(s) at collection (PPO/gae)
    no_whiten: bool = False  # skip whitening entirely: raw returns (no mean-center, no std)


Estimator = Callable[
    [torch.Tensor, List[List[List[int]]], "AdvantageContext"],
    Tuple[List[torch.Tensor], List[torch.Tensor]],
]

ADVANTAGE_ESTIMATORS: Dict[str, Estimator] = {}


def register_advantage_estimator(name: str) -> Callable[[Estimator], Estimator]:
    """Register an estimator under `name`."""

    def decorator(fn: Estimator) -> Estimator:
        if name in ADVANTAGE_ESTIMATORS and ADVANTAGE_ESTIMATORS[name] is not fn:
            raise ValueError(f"Advantage estimator '{name}' is already registered")
        ADVANTAGE_ESTIMATORS[name] = fn
        return fn

    return decorator


def get_advantage_estimator(name: str) -> Estimator:
    if name not in ADVANTAGE_ESTIMATORS:
        raise ValueError(f"Unknown advantage estimator '{name}'. Registered: {sorted(ADVANTAGE_ESTIMATORS)}")
    return ADVANTAGE_ESTIMATORS[name]


# ──────────────── tensor helpers (no Experience; the trainer assembles those) ────────────────


def expand_trace_advantages(advantages: torch.Tensor, ctx: AdvantageContext) -> List[torch.Tensor]:
    """Apply each trace's own advantage to its action tokens."""
    sample_advantages = advantages.split(ctx.exp_len)
    return [adv.unsqueeze(-1) * mask for adv, mask in zip(sample_advantages, ctx.action_masks)]


def group_reward_moments(rewards, group, trace_weights):
    """Equal rollout weight, including within-rollout variance; sample correction uses rollout count."""
    indices = [index for rollout in group for index in rollout]
    values, weights = rewards[indices], trace_weights[indices]
    # Center first so a constant-reward group remains exactly constant in floating point.
    mean = values[0] + ((values - values[0]) * weights).sum() / len(group)
    variance = ((values - mean).square() * weights).sum() / max(len(group) - 1, 1)
    return mean, variance.sqrt()


def normalize_advantages(advantages: List[torch.Tensor], ctx: AdvantageContext) -> List[torch.Tensor]:
    """Whiten per-experience (B, L) advantages across the batch (action-mask-weighted statistics)."""
    if ctx.no_whiten:  # raw returns as advantages — no batch mean/std coupling (single-rollout / async)
        return advantages
    flat_adv = torch.cat([a.flatten() for a in advantages], dim=0).float()
    flat_mask = torch.cat([m.flatten() for m in ctx.action_masks], dim=0)
    num_actions = flat_mask.sum()
    if num_actions == 0:
        # No action tokens anywhere in the batch -> nothing to whiten. Return zeros
        # rather than a 0/0 NaN mean (mirrors agg_loss's denom==0 guard).
        return [torch.zeros_like(a) for a in advantages]

    mean = (flat_adv * flat_mask).sum() / num_actions
    rstd = (((flat_adv - mean).pow(2) * flat_mask).sum() / num_actions).clamp(min=1e-8).rsqrt()
    return [(a - mean) * rstd for a in advantages]


# Group estimators use rollout membership for their baseline, keeping every trace reward.
GROUP_ADVANTAGE_ESTIMATORS = frozenset({"grpo", "dr_grpo", "reinforce_baseline", "rloo"})


# ──────────────────────────────── estimators ────────────────────────────────


@register_advantage_estimator("reinforce")
def reinforce(
    rewards: torch.Tensor, groups: List[List[List[int]]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """REINFORCE++ (https://arxiv.org/abs/2501.03262): no group baseline.

    Each trace's clipped scalar reward is placed on its last response token, a per-token KL
    penalty (-kl_coef * kl) is added, and discounted cumulative returns are accumulated. The
    resulting advantages are whitened across the batch.
    """
    sample_rewards = rewards.split(ctx.exp_len)
    returns = []
    for reward, mask, kl in zip(sample_rewards, ctx.action_masks, ctx.kls):
        token_reward = (-ctx.kl_coef * kl).float()  # (B, L) per-token KL penalty
        has_action = mask.bool().any(dim=1)
        last = mask.size(1) - 1 - mask.long().fliplr().argmax(dim=1, keepdim=True)  # last action token
        if has_action.any():
            token_reward[has_action] = token_reward[has_action].scatter_add(
                1, last[has_action], reward[has_action, None].to(token_reward.dtype)
            )
        token_reward = token_reward * mask

        # discounted reverse-cumulative returns: G_t = r_t + gamma * G_{t+1}
        if ctx.gamma == 1.0:
            seq_returns = token_reward.flip(1).cumsum(1).flip(1)
        else:
            seq_returns = torch.zeros_like(token_reward)
            running = torch.zeros(token_reward.size(0), device=token_reward.device)
            for t in reversed(range(token_reward.size(1))):
                running = token_reward[:, t] + ctx.gamma * running
                seq_returns[:, t] = running
        returns.append(seq_returns)

    return normalize_advantages(returns, ctx), returns


@register_advantage_estimator("reinforce_baseline")
def reinforce_baseline(
    rewards: torch.Tensor, groups: List[List[List[int]]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Group-mean baseline; advantages whitened across the batch."""
    advantages = rewards.clone()
    for group in groups:
        mean, _ = group_reward_moments(rewards, group, ctx.trace_weights)
        for rollout in group:
            advantages[rollout] = rewards[rollout] - mean
    returns = expand_trace_advantages(advantages, ctx)
    return normalize_advantages(returns, ctx), returns


@register_advantage_estimator("dr_grpo")
def dr_grpo(
    rewards: torch.Tensor, groups: List[List[List[int]]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Dr.GRPO (https://arxiv.org/abs/2503.20783): group-mean baseline, no std, no whitening."""
    advantages = rewards.clone()
    for group in groups:
        mean, _ = group_reward_moments(rewards, group, ctx.trace_weights)
        for rollout in group:
            advantages[rollout] = rewards[rollout] - mean
    returns = expand_trace_advantages(advantages, ctx)
    return [ret.clone() for ret in returns], returns


@register_advantage_estimator("grpo")
def grpo(
    rewards: torch.Tensor, groups: List[List[List[int]]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Per-trace GRPO: (trace reward - group mean) / (group std + eps)."""
    advantages = rewards.clone()
    for group in groups:
        mean, std = group_reward_moments(rewards, group, ctx.trace_weights)
        for rollout in group:
            advantages[rollout] = (rewards[rollout] - mean) / (std + 1e-9)
    returns = expand_trace_advantages(advantages, ctx)
    return [ret.clone() for ret in returns], returns


@register_advantage_estimator("rloo")
def rloo(
    rewards: torch.Tensor, groups: List[List[List[int]]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """Leave-one-out baseline (RLOO, https://arxiv.org/abs/2402.14740).

    Each rollout's baseline is the mean of the *other* rollouts in its group: `(sum - r) / (n - 1)`.
    """
    advantages = rewards.clone()
    for group in groups:
        mean, _ = group_reward_moments(rewards, group, ctx.trace_weights)
        if len(group) > 1:
            for rollout in group:
                own_mean = (rewards[rollout] * ctx.trace_weights[rollout]).sum()
                baseline = (mean * len(group) - own_mean) / (len(group) - 1)
                advantages[rollout] = rewards[rollout] - baseline
        # singleton group: no leave-one-out baseline exists, so the advantage stays the raw
        # reward (REINFORCE without baseline) — intentional, not zeroed.
    returns = expand_trace_advantages(advantages, ctx)
    return [ret.clone() for ret in returns], returns


@register_advantage_estimator("gae")
def gae(
    rewards: torch.Tensor, groups: List[List[List[int]]], ctx: AdvantageContext
) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
    """PPO advantage with a learned value baseline (GAE).

    The per-token reward mirrors ``reinforce`` — ``-kl_coef * kl`` on every step,
    plus the clipped outcome reward on the last action token — and the advantages
    and returns follow the standard GAE recursion::

        delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)
        A_t     = delta_t + gamma * lam * A_{t+1}        (V(s_{T+1}) = 0)
        ret_t   = A_t + V(s_t)                            (value-regression target)

    ``ctx.values`` holds the critic's collection-time V(s), one (B, L) tensor per
    experience. Masked positions (right-padding and multi-turn observation/tool
    tokens) are made *transparent* to the recursion: the bootstrap value
    ``V(s_{t+1})`` and the running GAE are carried across them, so the last action
    token before an interior gap bootstraps off the next action token's value rather
    than a spurious terminal ``V = 0``, and a masked token contributes no TD error of
    its own. This masked-GAE treatment is required for correct multi-turn advantages
    whenever ``lam < 1``. At the
    default ``lam = 1`` it reduces to ``A_t = G_t - V(s_t)`` with ``G_t`` the
    discounted return (interior values cancel by telescoping, so carry-vs-terminal is
    invisible there).

    Advantages are then batch-whitened (mean/std over action tokens, unless ``no_whiten``).
    Whitening touches only the advantages fed to the policy loss; ``returns = A + V(s)`` is left
    un-whitened so it stays the correct value-regression target for the value loss.
    """
    if ctx.values is None:
        raise ValueError("gae requires AdvantageContext.values (advantage_estimator=gae needs a critic)")
    sample_rewards = rewards.split(ctx.exp_len)
    advantages, returns = [], []
    for reward, mask, kl, values in zip(sample_rewards, ctx.action_masks, ctx.kls, ctx.values):
        # per-token reward: KL penalty + clipped scalar reward on the last action token
        token_reward = (-ctx.kl_coef * kl).float()  # (B, L)
        has_action = mask.bool().any(dim=1)
        last = mask.size(1) - 1 - mask.long().fliplr().argmax(dim=1, keepdim=True)  # last action token
        if has_action.any():
            token_reward[has_action] = token_reward[has_action].scatter_add(
                1, last[has_action], reward[has_action, None].to(token_reward.dtype)
            )
        token_reward = token_reward * mask
        # Zero V(s) off the action span so the value-regression target below
        # (returns = A + V) carries no critic output on masked positions; the
        # recursion itself never reads these (they are carried over, see below).
        values = values * mask

        # GAE reverse recursion: A_t = (r_t + gamma * V_{t+1} - V_t) + gamma * lam * A_{t+1}.
        # Masked (padding / multi-turn observation) tokens are transparent: carry the
        # bootstrap value and the running GAE across them so an action token before a
        # gap bootstraps from the next action token's value, not a spurious terminal
        # V = 0 (matters at lam < 1, a no-op at lam = 1).
        adv = torch.zeros_like(token_reward)
        running = torch.zeros(token_reward.size(0), device=token_reward.device)
        next_value = torch.zeros_like(running)  # V(s_{T+1}) = 0
        m = mask.float()
        for t in reversed(range(token_reward.size(1))):
            delta = token_reward[:, t] + ctx.gamma * next_value - values[:, t]
            running_t = delta + ctx.gamma * ctx.lam * running
            next_value = values[:, t] * m[:, t] + (1 - m[:, t]) * next_value
            running = running_t * m[:, t] + (1 - m[:, t]) * running
            adv[:, t] = running
        advantages.append(adv * mask)
        returns.append((adv + values) * mask)  # value-regression target: returns = A + V(s)
    # Advantages must be whitened (mean/std over action tokens, unless no_whiten),
    # re-masked so off-action positions stay 0; returns are left un-whitened (they
    # remain the value-regression target A + V(s)).
    advantages = [a * m for a, m in zip(normalize_advantages(advantages, ctx), ctx.action_masks)]
    return advantages, returns
