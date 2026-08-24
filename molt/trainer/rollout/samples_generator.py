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

import copy
import os
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from uuid import uuid4

import ray
import torch
from tqdm import tqdm

from molt.trainer.algorithm.experience import Experience
from molt.utils.logging_utils import init_logger
from molt.utils.vlm_utils import load_images, media_token_ids
from polar.rollout.models import SessionStatus, TaskRequest, TaskResult, TaskSpec

logger = init_logger(__name__)


def _collect_prompt_batch(dataloader_iter, num_prompts: int):
    """Draw up to `num_prompts` items from the prompt dataloader.

    Returns an exhaustion flag indicating whether the iterator is drained *after*
    collecting the returned prompts. Callers should still process any partial
    batch that was collected before exhaustion.
    """
    prompts, task_specs = [], []
    exhausted = False

    while len(prompts) < num_prompts:
        try:
            batch = next(dataloader_iter)
            _, batch_prompts, batch_task_specs = batch
            remaining = num_prompts - len(prompts)
            prompts.extend(batch_prompts[:remaining])
            task_specs.extend(batch_task_specs[:remaining])
        except StopIteration:
            exhausted = True
            break

    return prompts, task_specs, exhausted


def _sample_group_key(sample) -> int | str:
    """Stable prompt-group key when rollout samples carry grouping metadata."""
    sample_group_ids = getattr(sample, "group_ids", None)
    return sample_group_ids[0] if sample_group_ids else id(sample)


class SamplesGenerator:
    """Stream prompts through a persistent pool of rollout workers."""

    def __init__(
        self,
        strategy,
        prompts_dataloader,
        eval_dataloader,
        tokenizer,
        polar_rollout,
        task_spec=None,
    ):
        self.strategy = strategy
        self.args = strategy.args

        self.tokenizer = tokenizer
        self.polar_rollout = polar_rollout
        self.task_spec = TaskSpec.model_validate(task_spec) if task_spec is not None else None

        self.prompts_dataloader = prompts_dataloader
        self.eval_dataloader = eval_dataloader

    # Warm-resume (opt-in via --ckpt.warm_resume_rollouts). At every train step, save the rollout
    # groups that finished but were NOT trained yet — the surplus kept beyond the batch just shipped
    # (self._finished_samples) — so a resumed run trains them right away instead of idling ~one
    # generation while the pipeline refills. Each step writes a FRESH file (never overwritten), so a
    # resume restores exactly its own checkpoint's untrained groups. Only the file PATH is returned;
    # the groups themselves are large and must not travel the Ray queue. Best-effort: on any failure
    # the run just resumes normally (regenerating those groups).
    _BUFFER = "rollout_buffer.pt"

    def state_dict(self) -> Dict:
        """Save this step's untrained rollout groups to a fresh sidecar file, return its path."""
        finished = getattr(self, "_finished_samples", None)
        if not getattr(self.args.ckpt, "warm_resume_rollouts", False) or not finished:
            return {}
        try:
            # The rollout is held as lazy Experiences whose model inputs live in the generator
            # actor's object store (Experience.heavy_ref); those refs die when the actor restarts,
            # so persisting the ref would seed a resumed run with dead
            # handles. Materialize a LOCAL copy of each untrained group instead — copy + reload
            # fetches the heavy tensors from the object store — and persist that. The reload
            # runs on the COPY (copy.copy is shallow, reload rebinds its own fields), so the
            # originals stay lazy and the current step still ships them cheaply.
            local = [copy.copy(s).reload() for s in finished]
            d = os.path.join(os.path.dirname(self.args.ckpt.path.rstrip("/")), "rollout_warm")
            os.makedirs(d, exist_ok=True)
            self._warm_seq = getattr(self, "_warm_seq", 0) + 1
            path = os.path.join(d, f"{self._BUFFER}.{os.getpid()}.{self._warm_seq}")  # pid+seq: unique per step
            torch.save(local, path)
            for stale in sorted(os.scandir(d), key=lambda e: e.stat().st_mtime)[:-3]:  # keep newest 3; bound disk
                os.remove(stale.path)
            return {"buffer_file": path}
        except Exception as e:  # bookkeeping must never break the checkpoint
            logger.warning(f"warm-resume: save skipped ({e})")
            return {}

    def load_state_dict(self, state_dict: Optional[Dict]) -> None:
        """Restore this checkpoint's untrained groups (if any) to seed the first post-resume batch."""
        path = (state_dict or {}).get("buffer_file")
        if not path or not os.path.exists(path):
            return  # no file / failed save -> normal resume
        try:
            self._resumed_samples = torch.load(path, map_location="cpu", weights_only=False)
            logger.info(f"warm-resume: restored {len(self._resumed_samples)} rollout groups")
        except Exception as e:
            logger.warning(f"warm-resume: load skipped ({e})")

    @torch.no_grad()
    def generate_eval_samples(self, **generate_kwargs) -> List[Experience]:
        """Generate the full eval set while keeping a fixed rollout window full."""
        concurrency = self.args.eval.batch_size or self.args.rollout.batch_size
        dataloader_iter = iter(self.eval_dataloader)
        all_experiences: List[Experience] = []
        pending_refs = []
        exhausted = False
        drop_counts: Dict[str, int] = defaultdict(int)
        progress = tqdm(total=len(self.eval_dataloader), desc="Generate eval samples")
        try:
            while pending_refs or not exhausted:
                free_slots = concurrency - len(pending_refs)
                if free_slots > 0 and not exhausted:
                    prompts, task_specs, exhausted = _collect_prompt_batch(dataloader_iter, free_slots)
                    if prompts:
                        pending_refs.extend(
                            self._dispatch_rollouts(
                                prompts,
                                task_specs=task_specs,
                                **generate_kwargs,
                            )
                        )

                if not pending_refs:
                    break

                ready_refs, pending_refs = ray.wait(pending_refs, num_returns=1, timeout=10.0)
                for ref in ready_refs:
                    all_experiences.extend(
                        self._filter_group(ref, False, drop_counts, **generate_kwargs)
                    )
                    progress.update()
        finally:
            progress.close()

        if drop_counts:
            logger.info(f"Eval rollout drops: {dict(drop_counts)}")
        return all_experiences

    @torch.no_grad()
    def generate_samples(self, **generate_kwargs) -> Tuple[List[Experience], Dict[str, float], int, bool]:
        """Stream one training-sized batch out of a continuously-refilled rollout pool.

        Keeps `vllm_generate_batch_size` prompt rollouts in flight at all times and
        returns as soon as `rollout.batch_size` prompt groups have *finished* — it
        does not wait for the slow tail of the dispatched batch. The unfinished
        rollouts (and any surplus finished groups) persist on the instance across
        calls, so vLLM never drains between training steps: generation of the next
        batch fully overlaps training of the current one.

        Multi-turn agents emit several step-samples per rollout, so we chunk by
        GROUP (= prompt): each returned batch holds `rollout.batch_size` prompts,
        each with its N rollouts × K_i step-samples. Safe with `partial_rollout` —
        a weight refit pauses/resumes the engines (see broadcast_to_vllm), so the
        in-flight rollouts survive it.
        """
        if getattr(self, "_dataloader_iter", None) is None:
            self._dataloader_iter = iter(self.prompts_dataloader)
            # Seed from a warm-resume buffer if load_state_dict restored one, so the first
            # post-resume batch ships without waiting for a full fresh generation. Consumed once.
            self._finished_samples: List[Experience] = list(getattr(self, "_resumed_samples", None) or [])
            self._resumed_samples = None
            self._inflight_rollouts: List = []

        groups_per_batch = self.args.rollout.batch_size
        inflight_capacity = getattr(self.args.rollout, "vllm_generate_batch_size", None) or groups_per_batch
        dynamic_filtering = self.args.algo.dynamic_filtering_enable

        def finished_group_count() -> int:
            return len({_sample_group_key(sample) for sample in self._finished_samples})

        prompts_dispatched = 0
        groups_accepted = 0
        groups_completed = 0
        drop_counts: Dict[str, int] = defaultdict(int)
        score_stats: Dict[str, float] = defaultdict(float)  # pre-DAPO-filter score stats
        progress = tqdm(
            total=groups_per_batch, initial=min(finished_group_count(), groups_per_batch), desc="Generate samples"
        )

        while finished_group_count() < groups_per_batch:
            # Refill up to `inflight_capacity` so rollout generation stays saturated.
            free_slots = inflight_capacity - len(self._inflight_rollouts)
            if getattr(getattr(self.args, "train", None), "force_sync_mode", False):
                needed = groups_per_batch - finished_group_count() - len(self._inflight_rollouts)
                free_slots = min(free_slots, needed)
            if free_slots > 0 and self._dataloader_iter is not None:
                prompts, task_specs, dataloader_exhausted = _collect_prompt_batch(self._dataloader_iter, free_slots)
                prompts_dispatched += len(prompts)
                if prompts:
                    self._inflight_rollouts.extend(
                        self._dispatch_rollouts(
                            prompts,
                            task_specs=task_specs,
                            **generate_kwargs,
                        )
                    )
                if dataloader_exhausted:
                    self._dataloader_iter = None
                    logger.info("Prompt dataloader is exhausted.")

            if not self._inflight_rollouts:
                break  # dataloader drained and pool empty — emit whatever finished

            # Take the first rollout to finish; the slow ones keep generating in vLLM.
            ready, self._inflight_rollouts = ray.wait(self._inflight_rollouts, num_returns=1)
            for finished_rollout in ready:
                groups_completed += 1
                # Dropped groups (filtered or all-unusable) come back empty; their
                # slot is refilled with a fresh prompt on the next iteration.
                group_samples = self._filter_group(
                    finished_rollout, dynamic_filtering, drop_counts, score_stats=score_stats, **generate_kwargs
                )
                if group_samples:
                    self._finished_samples.extend(group_samples)
                    groups_accepted += 1
                    progress.update(1)
        progress.close()

        # Observability: per-reason drop counts + (when filtering) the pass rate.
        rollout_metrics = {f"rollout/dropped/{reason}": float(n) for reason, n in drop_counts.items()}
        if drop_counts:
            rollout_metrics["rollout/dropped/total"] = float(sum(drop_counts.values()))
        if dynamic_filtering and groups_completed:
            rollout_metrics["dynamic_filtering_pass_rate"] = groups_accepted / groups_completed * 100
        # Pre-filter stats: the model's TRUE judge pass rate over ALL scored rollouts, BEFORE DAPO
        # drops uniform groups. (post-filter `reward`/`pivotrl_correct` only covers kept MIXED groups
        # ~0.5-0.65 by construction, so it hides the real pass rate + the all-pass saturation.)
        if score_stats.get("score_n", 0) > 0:
            rollout_metrics["rollout/pre_filter_mean_score"] = score_stats["score_sum"] / score_stats["score_n"]
        if score_stats.get("groups", 0) > 0:
            rollout_metrics["rollout/group_all_pass_rate"] = score_stats["all_pass"] / score_stats["groups"] * 100
            rollout_metrics["rollout/group_all_fail_rate"] = score_stats["all_fail"] / score_stats["groups"] * 100

        # Hand back exactly `groups_per_batch` finished groups; keep any surplus for the next call.
        selected_groups: set = set()
        batch_samples: List[Experience] = []
        leftover_samples: List[Experience] = []
        for sample in self._finished_samples:
            group_key = _sample_group_key(sample)
            if group_key in selected_groups or len(selected_groups) < groups_per_batch:
                selected_groups.add(group_key)
                batch_samples.append(sample)
            else:
                leftover_samples.append(sample)
        self._finished_samples = leftover_samples

        # Exhausted only once the dataloader is done AND nothing is buffered or in flight.
        exhausted = self._dataloader_iter is None and not self._finished_samples and not self._inflight_rollouts
        return batch_samples, rollout_metrics, prompts_dispatched, exhausted

    def _passes_dynamic_filter(self, rollout_samples) -> bool:
        """Whether a scored group's mean rollout reward lands inside the dynamic-filtering range.

        A group with any unscored rollout always passes — filtering only applies once
        every rollout in the group has a score.
        """
        if not all(s.scores is not None for s in rollout_samples):
            return True
        scores = [s.scores[0].item() for s in rollout_samples]
        mean_score = sum(scores) / len(scores)
        min_score, max_score = self.args.algo.dynamic_filtering_range
        if min_score < mean_score < max_score:
            return True
        logger.info(
            f"Filtered out group: mean_score={mean_score:.2f}, range=({min_score:.2f}, {max_score:.2f}), "
            f"scores={[f'{s:.2f}' for s in scores]}"
        )
        return False

    def _filter_group(
        self,
        finished_rollout,
        dynamic_filtering: bool,
        drop_counts: Dict[str, int],
        score_stats: Dict[str, float] | None = None,
        **generate_kwargs,
    ) -> List[Experience]:
        """Filter one finished rollout (a prompt's N responses = one group) down to
        its kept Experiences, tallying every drop into ``drop_counts`` by reason.

        The single place the keep/drop policy lives, applying both filters:
        per-response unusable traces and the group-level DAPO dynamic-reward
        filter. Returns ``[]`` when the
        whole group is dropped. Both training and eval call this method, so the
        per-group keep/drop policy stays in one place.
        """
        group_samples: List[Experience] = []
        result = ray.get(finished_rollout)
        max_length = generate_kwargs.get("max_len")
        if max_length is None:
            max_length = self.args.data.max_len
        result = self._process_polar_task_result(result, max_length)
        for experience, drop_reason in result:
            if experience is not None:
                group_samples.append(experience.offload())
            elif drop_reason is not None:
                drop_counts[drop_reason] += 1

        if group_samples:
            expected = generate_kwargs.get("n_samples_per_prompt", self.args.rollout.n_samples_per_prompt)
            rollout_count = len({sample.rollout_ids[0] for sample in group_samples})
            if rollout_count < expected:
                drop_counts["incomplete_group"] += len(group_samples)
                return []

        if dynamic_filtering and group_samples:
            # Compaction can emit several step-samples with the same terminal score. Keep one
            # representative per rollout for filtering; all segments still enter training below.
            rollout_samples = {
                (s.rollout_ids[0] if getattr(s, "rollout_ids", None) else id(s)): s for s in group_samples
            }.values()
            # Pre-filter score stats (the model's TRUE judge pass rate over scored rollouts, BEFORE
            # DAPO drops uniform groups). Accumulate here, before any keep/drop decision, so the
            # logged mean reflects all-pass + all-fail + mixed (not just the kept mixed groups).
            if score_stats is not None:
                scored = [s.scores[0].item() for s in rollout_samples if s.scores is not None]
                if scored:
                    min_score, max_score = self.args.algo.dynamic_filtering_range
                    gmean = sum(scored) / len(scored)
                    score_stats["score_sum"] += sum(scored)
                    score_stats["score_n"] += len(scored)
                    score_stats["groups"] += 1.0
                    score_stats["all_pass"] += float(gmean >= max_score)
                    score_stats["all_fail"] += float(gmean <= min_score)
            # Require COMPLETE groups: a group that lost a response to a per-response drop
            # (vlm_truncation / no_action_tokens / logprob_misalign / ...) has < n_samples
            # usable samples, which would pull the accepted count off train_batch_size and make
            # it indivisible by the DP-rank count -> per-sample forward microbatches split unevenly
            # -> NCCL collective desync/hang. Drop+backfill the whole group so each accepted group
            # contributes exactly n_samples (batch stays a clean groups_per_batch * n_samples).
            n_samples = generate_kwargs.get("n_samples_per_prompt", self.args.rollout.n_samples_per_prompt)
            n_rollouts = len(rollout_samples)
            if n_rollouts < n_samples:
                drop_counts["incomplete_group"] += len(group_samples)
                return []
            if not self._passes_dynamic_filter(rollout_samples):
                drop_counts["dynamic_filter"] += len(group_samples)
                return []
        return group_samples

    def _dispatch_rollouts(
        self,
        prompts: list[str],
        *,
        task_specs: list | None = None,
        **generate_kwargs,
    ) -> list:
        """Submit one Polar task for each prompt group."""
        if task_specs is None:
            task_specs = [None] * len(prompts)
        sampling_params = {
            "max_tokens": generate_kwargs.get("max_new_tokens"),
            "max_total_tokens": generate_kwargs.get("max_len", self.args.data.max_len),
            "temperature": generate_kwargs.get("temperature", 1.0),
            "top_p": generate_kwargs.get("top_p", 1.0),
            "top_k": generate_kwargs.get("top_k", -1),
            "min_tokens": generate_kwargs.get("min_new_tokens", 1),
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "seed": None,
            "skip_special_tokens": False,
            "ignore_eos": False,
            "include_stop_str_in_output": False,
            "logprobs": 1,
            "n": 1,
        }
        n_samples = generate_kwargs.get("n_samples_per_prompt", self.args.rollout.n_samples_per_prompt)
        refs = []
        for prompt, row_spec in zip(prompts, task_specs, strict=True):
            if row_spec is not None and self.task_spec is not None:
                raise ValueError("Choose either --rollout.task_spec or row-level task specifications, not both")
            spec = TaskSpec.model_validate(row_spec) if row_spec is not None else self.task_spec
            if spec is None:
                raise ValueError("A Polar task specification is required")
            request = TaskRequest(
                task_id=uuid4().hex,
                instruction=prompt,
                num_samples=n_samples,
                timeout_seconds=self.args.rollout.session_timeout,
                runtime=spec.runtime,
                agent=spec.agent,
                builder=spec.builder,
                evaluator=spec.evaluator,
                sampling_params=sampling_params,
                metadata=spec.metadata,
            )
            refs.append(self.polar_rollout.run_task.remote(request.model_dump(mode="json")))
        return refs

    def _process_polar_task_result(self, task_result, max_length: int):
        """Materialize each trainable Polar trace as a Molt Experience."""
        task = TaskResult.model_validate(task_result)
        converted = []
        if not task.results:
            return [(None, "empty_task")]

        for session in task.results:
            if session.task_id != task.task_id:
                converted.append((None, "identity_mismatch"))
                continue

            for trace in session.trajectory.traces:
                if not trace.prompt_ids:
                    converted.append((None, "empty_prompt_tokens"))
                    continue

                full_sequence_ids = trace.prompt_ids + trace.response_ids
                full_token_mask = [0] * len(trace.prompt_ids) + trace.loss_mask
                full_token_logprobs = [0.0] * len(trace.prompt_ids) + (trace.response_logprobs or [])
                sequence_ids = full_sequence_ids[:max_length]
                token_mask = full_token_mask[:max_length]
                token_logprobs = full_token_logprobs[:max_length]

                known_media_ids = media_token_ids(self.tokenizer) if trace.media_paths else set()
                if trace.media_paths and len(full_sequence_ids) > max_length:
                    if not known_media_ids:
                        converted.append((None, "vlm_media_unresolved"))
                        continue
                    if any(token in known_media_ids for token in full_sequence_ids[max_length:]):
                        converted.append((None, "vlm_truncation"))
                        continue
                if not any(token_mask):
                    converted.append((None, "no_action_tokens"))
                    continue
                if trace.reward is None:
                    converted.append((None, "missing_reward"))
                    continue

                mm_train_inputs = None
                image_tokens = 0
                if trace.media_paths:
                    pil_images = load_images(trace.media_paths)
                    image_processor = getattr(self.tokenizer, "image_processor", None)
                    if len(pil_images) != len(trace.media_paths):
                        converted.append((None, "vlm_media_unreadable"))
                        continue
                    if image_processor is None:
                        converted.append((None, "vlm_processor_missing"))
                        continue
                    try:
                        mm_train_inputs = dict(image_processor(images=pil_images, return_tensors="pt"))
                    except Exception:
                        logger.exception("Failed to build VLM training inputs from Polar media")
                        converted.append((None, "vlm_processing_failed"))
                        continue
                    image_tokens = sum(token in known_media_ids for token in sequence_ids)
                    if known_media_ids:
                        media_runs = sum(
                            token in known_media_ids and (index == 0 or sequence_ids[index - 1] not in known_media_ids)
                            for index, token in enumerate(sequence_ids)
                        )
                        if media_runs != len(trace.media_paths):
                            converted.append((None, "vlm_media_misaligned"))
                            continue
                        image_grid = mm_train_inputs.get("image_grid_thw")
                        if image_grid is not None:
                            image_grid = torch.as_tensor(image_grid).reshape(-1, 3).long()
                            merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
                            expected_image_tokens = int((image_grid.prod(dim=-1) // merge_size**2).sum())
                            if image_tokens != expected_image_tokens:
                                converted.append((None, "vlm_media_misaligned"))
                                continue

                reward = float(trace.reward)
                info = {
                    "reward": torch.tensor([reward]),
                    "score": torch.tensor([reward]),
                    "response_clip_ratio": torch.tensor([len(sequence_ids) >= max_length]),
                }
                if trace.media_paths:
                    info["image_tokens"] = torch.tensor([image_tokens])
                for name, value in session.timing.model_dump().items():
                    info[f"polar/{name}"] = torch.tensor([value])

                converted.append(
                    (
                        Experience(
                            sequences=torch.tensor([sequence_ids], dtype=torch.long),
                            attention_mask=torch.ones((1, len(sequence_ids)), dtype=torch.long),
                            action_mask=torch.tensor([token_mask[1:]], dtype=torch.bool),
                            rollout_log_probs=torch.tensor([token_logprobs[1:]]),
                            prompts=[task.instruction],
                            mm_train_inputs=[mm_train_inputs],
                            group_ids=[task.task_id],
                            rollout_ids=[session.session_id],
                            rewards=torch.tensor([reward]),
                            scores=torch.tensor([reward]),
                            response_length=torch.tensor([sum(token_mask)]),
                            truncated=torch.tensor(
                                [trace.finish_reason == "length" or len(full_sequence_ids) > max_length]
                            ),
                            total_length=torch.tensor([len(sequence_ids)]),
                            info=info,
                        ),
                        None,
                    )
                )
            if session.status != SessionStatus.COMPLETED and not session.trajectory.traces:
                converted.append((None, f"session_{session.status.lower()}"))
        return converted
