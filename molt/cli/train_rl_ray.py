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

import argparse
import atexit
import os
from pathlib import Path

import yaml

from molt.trainer.algorithm.experience import get_model_parallel_size


def _ray_runtime_env_vars():
    """Build the environment inherited by Ray workers."""
    env_vars = {
        "TOKENIZERS_PARALLELISM": os.environ.get("TOKENIZERS_PARALLELISM", "true"),
        "NCCL_DEBUG": os.environ.get("NCCL_DEBUG", "WARN"),
        "RAY_ENABLE_ZERO_COPY_TORCH_TENSORS": os.environ.get("RAY_ENABLE_ZERO_COPY_TORCH_TENSORS", "1"),
    }
    for name in (
        "FLASHINFER_WORKSPACE_BASE",
        "FLASHINFER_WORKSPACE_DIR",
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "NCCL_IB_DISABLE",
        "NCCL_CUMEM_ENABLE",
        "NCCL_MNNVL_ENABLE",
        "NCCL_NVLS_ENABLE",
        "NCCL_P2P_DISABLE",
        "NCCL_SHM_DISABLE",
        "NCCL_SOCKET_IFNAME",
        "TRANSFORMERS_CACHE",
        "TORCH_COMPILE_DISABLE",
        "PYTORCH_CUDA_ALLOC_CONF",
        "VLLM_WORKER_MULTIPROC_METHOD",
        "VLLM_ALLREDUCE_USE_SYMM_MEM",
        "VLLM_USE_NCCL_SYMM_MEM",
        "WANDB_API_KEY",
        "WANDB_ENTITY",
        "WANDB_MODE",
    ):
        if os.environ.get(name):
            env_vars[name] = os.environ[name]
    return env_vars


def train(args):
    import ray
    from ray.util.placement_group import placement_group

    from molt.trainer.placement import model_placement_strategy
    from molt.trainer.vllm import create_vllm_engines
    from molt.trainer.workers.actor_group import RayActorGroup, ReferenceModelActor
    from molt.trainer.workers.policy_actor import PolicyModelActor
    from molt.utils import get_strategy

    # initialize ray if not initialized
    if not ray.is_initialized():
        # Defaults respect user overrides (e.g. NCCL_DEBUG=INFO via
        # `ray job submit --runtime-env-json`); the listed names are
        # cache/workspace knobs vLLM and HF read inside Ray actors.
        ray.init(runtime_env={"env_vars": _ray_runtime_env_vars()})

    # configure strategy
    strategy = get_strategy(args)
    strategy.print(args)

    # Init vLLM before actor/ref placement. vLLM's mp backend asks Ray for
    # whole-node bundles (for example 8 GPUs); if actor/ref one-GPU bundles are
    # spread first, they fragment every node and the vLLM placement group can
    # remain pending forever.
    vllm_engines = None
    # data.max_len is the shared total-context budget (prompt + generation),
    # so vLLM's max_model_len reads it directly. For VLM (image-tokenized
    # prompts) or multi-turn agents, set --data.max_len high enough to fit
    # the longest expanded prompt plus rollout.max_new_tokens.
    max_len = args.data.max_len
    if args.vllm.num_engines is not None and args.vllm.num_engines > 0:
        vllm_engines = create_vllm_engines(
            args.vllm.num_engines,
            args.vllm.tensor_parallel_size,
            args.actor.model_name_or_path,
            args.train.seed,
            args.train.full_determinism_enable,
            args.vllm.enforce_eager,
            max_len,
            args.vllm.gpu_memory_utilization,
            "processed_logprobs" if args.algo.advantage.is_correction_level != "off" else None,
            max_images_per_prompt=getattr(args.data, "max_images_per_prompt", 0),
            mm_encoder_attn_backend=args.vllm.mm_encoder_attn_backend,
            gdn_prefill_backend=args.vllm.gdn_prefill_backend,
            attention_backend=args.vllm.attention_backend,
            mamba_ssm_cache_dtype=args.vllm.mamba_ssm_cache_dtype,
            distributed_executor_backend=args.vllm.distributed_executor_backend,
            enable_expert_parallel=args.vllm.enable_expert_parallel,
            moe_backend=args.vllm.moe_backend,
            disable_custom_all_reduce=args.vllm.disable_custom_all_reduce,
            enable_prefix_caching=args.vllm.enable_prefix_caching,
            enable_chunked_prefill=args.vllm.enable_chunked_prefill,
            max_num_batched_tokens=args.vllm.max_num_batched_tokens,
            async_scheduling=args.vllm.async_scheduling,
            decode_context_parallel_size=args.vllm.decode_context_parallel_size,
            dtype=args.vllm.dtype,
            kv_cache_dtype=args.vllm.kv_cache_dtype,
            block_size=args.vllm.block_size,
            mtp_num_speculative_tokens=args.vllm.mtp_num_speculative_tokens,
            pipeline_parallel_size=getattr(args.vllm, "pipeline_parallel_size", 1),
            data_parallel_size=getattr(args.vllm, "data_parallel_size", 1),
        )

    # Serve each engine's OpenAI API behind one router. Polar sends supported
    # rollouts through it; weight sync still goes straight to the engine workers.
    router_url = None
    _vllm_router = None  # MUST stay in scope for the whole run — the router actor dies if GC'd
    if vllm_engines:
        from molt.trainer.rollout.router import create_vllm_router

        _vllm_router, router_url = create_vllm_router(
            vllm_engines,
            policy=getattr(args.vllm, "router_policy", "consistent_hash"),
            tool_call_parser=args.vllm.tool_call_parser,
            reasoning_parser=args.vllm.reasoning_parser,
        )
        print(f"[rollout] vLLM router up at {router_url} fronting {len(vllm_engines)} engines", flush=True)

    polar_rollout = None
    polar_gateways = []

    def close_rollout_services():
        nonlocal polar_rollout, polar_gateways, _vllm_router
        rollout = polar_rollout
        gateways = polar_gateways
        router = _vllm_router
        polar_rollout = None
        polar_gateways = []
        _vllm_router = None
        if rollout is None and not gateways and router is None:
            return
        try:
            if gateways:
                ray.get([gateway.close.remote() for gateway in gateways])
        finally:
            try:
                if rollout is not None:
                    ray.get(rollout.close.remote())
            finally:
                if router is not None:
                    ray.get(router.close.remote())

    atexit.register(close_rollout_services)

    from molt.trainer.rollout.polar import PolarServiceActor
    from polar.config import TopologyConfig

    try:
        for index in range(args.rollout.gateway_count):
            polar_gateways.append(
                PolarServiceActor.options(scheduling_strategy="SPREAD").remote("gateway", f"gateway-{index}")
            )
        gateway_nodes = ray.get([gateway.descriptor.remote() for gateway in polar_gateways])
        polar_rollout = PolarServiceActor.remote("rollout")
        rollout_node = ray.get(polar_rollout.descriptor.remote())
        topology = TopologyConfig.model_validate(
            {
                "rollout": {
                    "host": rollout_node["host"],
                    "port": rollout_node["port"],
                    "public_url": rollout_node["url"],
                    "save_dir": str(Path(args.rollout.save_dir).resolve()),
                },
                "gateway": {
                    "rollout_server_url": rollout_node["url"],
                    "nodes": [
                        {
                            "id": node["node_id"],
                            "host": node["host"],
                            "port": node["port"],
                            "public_url": node["url"],
                            "model_served": "policy",
                            "inference": {"base_url": router_url},
                            "max_init_workers": args.rollout.gateway_concurrency,
                            "max_run_workers": args.rollout.gateway_concurrency,
                            "max_postrun_workers": args.rollout.gateway_concurrency,
                        }
                        for node in gateway_nodes
                    ],
                },
            }
        )
        save_dir = Path(args.rollout.save_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        topology_path = save_dir / "topology.json"
        topology_path.write_text(topology.model_dump_json(indent=2, exclude={"path"}) + "\n")
        topology_payload = topology.model_dump(mode="json", exclude={"path"})
        ray.get(polar_rollout.start.remote(topology_payload))
        ray.get([gateway.start.remote(topology_payload) for gateway in polar_gateways])
        ray.get(polar_rollout.ready.remote(len(polar_gateways)))
    except Exception:
        close_rollout_services()
        raise
    print(
        f"[rollout] Polar ready at {rollout_node['url']} with {len(polar_gateways)} gateways; "
        f"topology saved to {topology_path}",
        flush=True,
    )

    from molt.trainer.rl_trainer import RLTrainer

    # Rollout sampling kwargs shared by the eval-only and training controllers.
    gen_kwargs = {
        "do_sample": True,
        "max_len": max_len,
        "max_new_tokens": args.rollout.max_new_tokens,
        "temperature": args.rollout.temperature,
        "top_p": args.rollout.top_p,
    }

    # Eval-only: score --eval.dataset once and exit, with NO training. vLLM already holds the HF
    # weights, so passing None actors keeps the whole training side unbuilt inside RLTrainer — the
    # policy/ref/critic FSDP models never load and their GPUs go to the eval.
    if args.eval.eval_only:
        assert args.eval.dataset, "--eval.eval_only requires --eval.dataset."
        eval_trainer = RLTrainer.remote(
            args.actor.model_name_or_path,
            strategy,
            None,
            None,
            vllm_engines,
            polar_rollout=polar_rollout,
            polar_gateways=polar_gateways,
            **gen_kwargs,
        )
        try:
            print(f"[eval-only] {ray.get(eval_trainer.run_eval_only.remote())}", flush=True)
        finally:
            close_rollout_services()
        return

    # init actor / reference / critic models
    # Colocating only affects FSDP models (actor + reference + critic); they
    # time-slice one shared placement group sized to the actor. vLLM rollout
    # engines keep their own placement group.
    pg = None
    has_ref = args.algo.kl.init_coef > 0
    has_critic = args.algo.advantage.estimator == "gae"
    colocate_fsdp_models = args.train.colocate_fsdp_models and (has_ref or has_critic)

    # Fail-fast on GPU over-subscription (covers BOTH paths). vLLM holds num_engines*TP GPUs.
    # Colocated FSDP models time-slice ONE actor-sized group; otherwise the actor, ref and critic
    # each claim their own GPUs. If the total exceeds the cluster, placement deadlocks FOREVER
    # (ray.get(pg.ready()) when colocating, else the actor-group creation) — e.g. a 6-node run left
    # at the 8-node VLLM_NUM_ENGINES=24 (24*2=48 GPUs leaves 0 for the actor). Error, don't hang.
    actor_gpus = args.actor.num_nodes * args.actor.num_gpus_per_node
    model_gpus = (
        actor_gpus
        if colocate_fsdp_models
        else (
            actor_gpus
            + (args.ref.num_nodes * args.ref.num_gpus_per_node if has_ref else 0)
            + (actor_gpus if has_critic else 0)
        )
    )
    vllm_pp = getattr(args.vllm, "pipeline_parallel_size", 1)
    vllm_dp = getattr(args.vllm, "data_parallel_size", 1)
    vllm_gpus = args.vllm.num_engines * args.vllm.tensor_parallel_size * vllm_pp * vllm_dp
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    if total_gpus and model_gpus + vllm_gpus > total_gpus:
        raise RuntimeError(
            f"GPU over-subscription: FSDP models need {model_gpus} GPUs + vLLM "
            f"({args.vllm.num_engines} engines x TP{args.vllm.tensor_parallel_size} x PP{vllm_pp} x DP{vllm_dp}) = {vllm_gpus} GPUs = "
            f"{model_gpus + vllm_gpus} > {total_gpus} cluster GPUs. Lower --vllm.num_engines or add nodes. "
            f"(Otherwise placement deadlocks forever.)"
        )

    if colocate_fsdp_models:
        if has_ref:
            assert (
                args.actor.num_nodes == args.ref.num_nodes
                and args.actor.num_gpus_per_node == args.ref.num_gpus_per_node
            ), "num_nodes and num_gpus_per_node must match when colocating the actor and ref model."

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(args.actor.num_nodes * args.actor.num_gpus_per_node)]
        pg = placement_group(bundles, strategy=model_placement_strategy())
        ray.get(pg.ready())

    fsdp_mp_size = get_model_parallel_size(args)

    actor_model = RayActorGroup(
        args.actor.num_nodes,
        args.actor.num_gpus_per_node,
        PolicyModelActor,
        pg=pg,
        num_gpus_per_actor=0.2 if pg else 1,
        duplicate_actors=fsdp_mp_size,
    )

    if has_ref:
        ref_model = RayActorGroup(
            args.ref.num_nodes,
            args.ref.num_gpus_per_node,
            ReferenceModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=fsdp_mp_size,
        )
    else:
        ref_model = None

    # PPO critic: its own group, colocated on the actor's GPUs via the shared
    # placement group (same world size / mesh as the actor). Only for gae.
    if has_critic:
        from molt.trainer.workers.critic_actor import CriticModelActor

        critic_model = RayActorGroup(
            args.actor.num_nodes,
            args.actor.num_gpus_per_node,
            CriticModelActor,
            pg=pg,
            num_gpus_per_actor=0.2 if pg else 1,
            duplicate_actors=fsdp_mp_size,
        )
    else:
        critic_model = None

    # init RL trainer (single controller)
    policy_trainer = RLTrainer.remote(
        args.actor.model_name_or_path,
        strategy,
        actor_model,
        ref_model,
        vllm_engines,
        critic_model_group=critic_model,
        polar_rollout=polar_rollout,
        polar_gateways=polar_gateways,
        **gen_kwargs,
    )

    # training update steps
    max_steps = ray.get(policy_trainer.get_max_steps.remote())
    # The actor's LR scheduler must be sized to the optimizer steps it ACTUALLY takes.
    # With a PPO critic, the actor is frozen for the first `freezing_steps` (critic
    # warmup) and never advances its scheduler then, so sizing it to the full
    # `max_steps` would shift its warmup/decay and never reach min_lr. The critic
    # trains every step, so it keeps the full `max_steps`.
    actor_max_steps = max(1, max_steps - (args.actor.freezing_steps if has_critic else 0))

    # init actor/reference models
    refs = []
    refs.extend(
        actor_model.async_init_model_from_pretrained(
            strategy, args.actor.model_name_or_path, actor_max_steps, vllm_engines
        )
    )
    if ref_model is not None:
        ref_path = args.ref.model_name_or_path or args.actor.model_name_or_path
        refs.extend(ref_model.async_init_model_from_pretrained(strategy, ref_path))
    if critic_model is not None:
        refs.extend(critic_model.async_init_model_from_pretrained(strategy, args.actor.model_name_or_path, max_steps))
    ray.get(refs)

    try:
        ray.get(policy_trainer.fit.remote())
        if not args.ckpt.disable_final_save:
            ray.get(actor_model.async_export_hf_model())
    finally:
        close_rollout_services()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    from molt.cli.common_args import (
        add_ckpt_args,
        add_fsdp_args,
        add_logger_args,
        add_optimizer_args,
        resolve_ckpt_retention,
    )

    # ====================== Shared blocks (same surface as train_sft) ======================
    # FSDP2 / AutoModel backend.
    add_fsdp_args(parser)
    # Optimizer + scheduler + grad clip for the actor ("actor." prefix + lr default 1e-6).
    add_optimizer_args(parser, prefix="actor.", default_adam_lr=1e-6)
    # Checkpoints; RL disables eval when --eval.steps is -1 and adds best-ckpt selection.
    add_ckpt_args(parser, default_ckpt_path="./ckpt/checkpoints_rl_ray")
    parser.add_argument(
        "--ckpt.best_metric_key",
        type=str,
        default="",
        help="Eval metric key for best checkpoint saving (e.g., eval_default_pass1). "
        "Empty string auto-detects first pass1 metric. Set to 'none' to disable best checkpoint saving.",
    )
    parser.add_argument(
        "--ckpt.warm_resume_rollouts",
        action="store_true",
        help="At each train step, save the rollout groups that finished but were not trained yet, "
        "so a resumed async run trains them immediately instead of idling ~one generation while "
        "the pipeline refills. Opt-in/experimental; best-effort — a failed save or missing file "
        "falls back to a normal resume.",
    )
    # wandb + TensorBoard + logging cadence.
    add_logger_args(parser, default_wandb_project="molt_train_rl", run_name_prefix="rl")

    # ====================== RL-specific arguments ======================
    # Model
    parser.add_argument("--actor.model_name_or_path", type=str, default=None, help="HF model name or path")
    parser.add_argument(
        "--actor.gradient_checkpoint",
        nargs="?",
        const="full",
        default="full",
        help="Activation-checkpointing mode (string): 'full' = full-block AC (AutoModel "
        "recipe default), 'selective' = TorchTitan per-op AC, 'none'/'off'/'' = disable.",
    )
    parser.add_argument("--actor.aux_loss_coef", type=float, default=0, help="MoE balancing loss")
    parser.add_argument(
        "--actor.freezing_steps",
        type=int,
        default=0,
        help="Critic warmup: freeze the actor's policy update for the first N optimizer steps so "
        "the value model fits the initial rollouts before its early, high-variance advantages move "
        "the policy. The critic keeps training while frozen. 0 disables; only applies with "
        "--algo.advantage.estimator gae.",
    )
    parser.add_argument(
        "--actor.freeze_visual_encoder",
        action="store_true",
        default=False,
        help="Freeze vision encoder weights (only train language model). Reduces memory and weight sync overhead.",
    )
    parser.add_argument(
        "--actor.freeze_moe_router",
        action="store_true",
        default=False,
        help="Freeze the MoE router/gate weights to stabilize MoE training.",
    )
    parser.add_argument(
        "--ref.model_name_or_path",
        type=str,
        default=None,
        help="Reference/teacher checkpoint. Defaults to the actor checkpoint (standard KL-to-init RL). "
        "Set to a different checkpoint when KL should use another reference policy.",
    )
    # Critic (PPO value model; used only when --algo.advantage.estimator gae). It is
    # colocated in the actor workers and reuses the actor's optimizer/parallelism config.
    parser.add_argument(
        "--critic.model_name_or_path",
        type=str,
        default=None,
        help="Critic init checkpoint (a reward model or the policy). Defaults to the actor checkpoint.",
    )
    parser.add_argument(
        "--critic.value_clip",
        type=float,
        default=0.2,
        help="PPO value-clip range for the value loss.",
    )
    parser.add_argument(
        "--critic.freeze_moe_router",
        action="store_true",
        default=False,
        help="Freeze the critic's MoE router/gate weights, independently of --actor.freeze_moe_router "
        "(also inherited when the actor's flag is set).",
    )
    parser.add_argument(
        "--critic.max_epochs",
        type=int,
        default=None,
        help="Optimization epochs the critic runs per RL step; defaults to --train.max_epochs. Set "
        "higher to fit the value function harder each step (a PPO critic often wants more passes).",
    )
    # Independent critic optimizer + scheduler + grad-clip ("critic." prefix), so the
    # value model can use its own LR/optimizer (PPO critics often want a higher LR).
    add_optimizer_args(parser, prefix="critic.", default_adam_lr=5e-6)

    # Data
    parser.add_argument("--data.prompt_dataset", type=str, default=None, help="HF dataset name or path")
    parser.add_argument(
        "--data.prompt_probs",
        type=str,
        default=None,
        help="sampling probs for datasets",
    )
    parser.add_argument("--data.prompt_split", type=str, default="train")
    parser.add_argument("--data.max_samples", type=int, default=int(1e8), help="Max number of samples")
    parser.add_argument("--data.max_len", type=int, default=2048, help="Max total sequence length (prompt + response)")
    parser.add_argument("--data.input_key", type=str, default="input", help="JSON dataset key")
    parser.add_argument(
        "--data.task_key",
        type=str,
        default="task",
        help="Dataset column containing a complete Polar task specification for that row.",
    )
    parser.add_argument(
        "--data.max_images_per_prompt", type=int, default=0, help="Max images per prompt for vLLM (0 = text-only)"
    )
    parser.add_argument("--data.disable_fast_tokenizer", action="store_true", default=False)
    parser.add_argument(
        "--data.dataloader_num_workers",
        type=int,
        default=0,
        help="Number of dataloader workers for IO (for Ray training, ensure sufficient CPU resources per actor)",
    )

    # Algorithm: advantage estimation, KL, policy clipping, reward shaping
    parser.add_argument(
        "--algo.advantage.estimator",
        type=str,
        choices=["reinforce", "rloo", "reinforce_baseline", "grpo", "dr_grpo", "gae"],
        default="reinforce",
        help="Advantage estimation method: reinforce, rloo, reinforce_baseline, grpo, dr_grpo, or gae "
        "(PPO value baseline — builds a colocated critic; see --critic.*)",
    )
    parser.add_argument("--algo.advantage.gamma", type=float, default=1, help="discount factor")
    parser.add_argument(
        "--algo.advantage.lam",
        type=float,
        default=1.0,
        help="GAE lambda (PPO only). 1.0 = Monte-Carlo return minus the value baseline.",
    )
    parser.add_argument(
        "--algo.advantage.no_whiten",
        action="store_true",
        default=False,
        help="skip cross-batch advantage whitening — use raw returns (no mean-center, no std). "
        "Affects the whitening estimators (reinforce / reinforce_baseline / gae); for a no-std GRPO "
        "use --algo.advantage.estimator dr_grpo. Useful in single-rollout / async where the batch "
        "mean/std couple samples and are a noisy moving target.",
    )
    # Train/rollout (FSDP-actor vs vLLM) logprob-mismatch importance-sampling correction.
    parser.add_argument(
        "--algo.advantage.is_correction_level",
        type=str,
        default="off",
        choices=["off", "token", "seq", "geo"],
        help="Granularity of the gated ratio: off (correction disabled), token (each token), "
        "seq (product = exp(sum), unbiased/high-variance), geo (per-seq geometric mean = exp(mean), "
        "balanced). seq/geo are rejection filters and require --is_correction_mode mask.",
    )
    parser.add_argument(
        "--algo.advantage.is_correction_mode",
        type=str,
        default="mask",
        choices=["mask", "clip", "trunc"],
        help="Bound treatment (token level only for clip/trunc): mask (drop out-of-band units, zero "
        "gradient), clip (clamp the weight into [low, high]), trunc (clamp only the upper tail).",
    )
    parser.add_argument(
        "--algo.advantage.is_correction_threshold",
        type=float,
        nargs=2,
        default=[0.5, 5.0],
        help="Low and high bounds [low, high] for the off-policy IS ratio pi_train/pi_rollout.",
    )
    parser.add_argument(
        "--algo.kl.use_loss", action="store_true", default=False, help="whether to use KL loss from GRPO"
    )
    parser.add_argument(
        "--algo.kl.estimator",
        type=str,
        default="k1",
        choices=["k1", "k2", "k3"],
        help=(
            "In GRPO, k3 is utilized as the loss function, while k2, when used as the loss, is nearly equivalent to k1."
        ),
    )
    parser.add_argument(
        "--algo.kl.init_coef",
        type=float,
        default=None,
        help="KL-to-reference coefficient (default 0.01).",
    )
    parser.add_argument(
        "--algo.kl.target",
        type=float,
        default=None,
        help=(
            "Target KL for adaptive control. Set to a positive value to adjust the "
            "KL coefficient so measured KL moves toward this target; leave unset "
            "to keep the coefficient fixed at --algo.kl.init_coef."
        ),
    )
    parser.add_argument(
        "--algo.kl.horizon",
        type=int,
        default=10000,
        help=(
            "Adaptive-KL horizon in rollout samples; larger values produce slower "
            "coefficient updates. Used only when --algo.kl.target is set."
        ),
    )
    parser.add_argument(
        "--algo.dynamic_filtering_enable", action="store_true", default=False, help="Enable dynamic filtering"
    )
    parser.add_argument(
        "--algo.dynamic_filtering_range", nargs=2, default=(0, 1), type=float, help="Dynamic filtering rewards range"
    )
    parser.add_argument(
        "--actor.eps_clip_low_high", type=float, nargs=2, default=None, help="policy clip low and high"
    )
    parser.add_argument("--actor.dual_clip", type=float, default=None, help="Dual-clip policy objective")
    parser.add_argument(
        "--actor.loss_mode",
        type=str,
        default="ppo",
        choices=["ppo", "cispo", "gspo"],
        help="Policy-gradient surrogate: ppo (clipped min(surr1,surr2), optionally --actor.dual_clip) or "
        "cispo (https://arxiv.org/abs/2506.13585 — clips only the upper side of the IS ratio, "
        "stop-gradient through that weight, gradient flows through log-probs only; pass "
        "--actor.eps_clip_low_high's high value as the absolute ratio ceiling, e.g. 1.2, not a +offset; "
        "--actor.dual_clip is unused in this mode) or "
        "gspo (https://arxiv.org/abs/2507.18071 — clips ONE ratio per sequence, its geometric mean, "
        "so a single outlier token cannot clip the whole update; aggregated with molt's global "
        "token-mean denominator, not the paper's per-sequence 1/|y|; --actor.dual_clip is unused).",
    )
    parser.add_argument(
        "--actor.entropy_coef",
        type=float,
        default=None,
        help=(
            "Entropy coefficient. Any nonzero value enables entropy computation and "
            "logs entropy_loss; positive values encourage higher entropy. "
            "0 or unset skips entropy computation."
        ),
    )
    parser.add_argument("--reward.clip_range", type=float, nargs=2, default=(-10, 10), help="Reward clip range")

    # Rollout / generation
    parser.add_argument(
        "--rollout.task_spec",
        type=str,
        default=None,
        help="Polar runtime, agent, builder, and evaluator YAML; optional when every dataset row has --data.task_key.",
    )
    # -- vLLM engine --
    parser.add_argument(
        "--vllm.num_engines", type=int, default=None, help="number of vLLM Engines, set to 0 to disable vLLM"
    )
    parser.add_argument(
        "--vllm.tensor_parallel_size",
        type=int,
        default=1,
        help="tensor parallel size of vLLM Engine for multi-GPU inference",
    )
    parser.add_argument(
        "--vllm.pipeline_parallel_size",
        type=int,
        default=1,
        help="pipeline parallel size per vLLM engine. For a giant MoE that overflows a "
        "node, set TP to the node GPU count and PP to the node span (TP*PP GPUs/engine, "
        "ray executor): PP hands off between stages point-to-point across nodes instead "
        "of a cross-node TP all-reduce every layer.",
    )
    parser.add_argument(
        "--vllm.data_parallel_size",
        type=int,
        default=1,
        help="data parallel size per vLLM engine (single-node mp backend). vLLM has no standalone "
        "expert-parallel size: EP = TP * DP, so raise DP to decouple EP from TP "
        "(DeepSeek-V3-style TP8+DP4 attention -> EP32 experts). An engine spans "
        "TP*PP*DP GPUs; DP > 1 cannot be combined with pipeline parallelism.",
    )
    parser.add_argument("--vllm.sync_backend", type=str, default="nccl", help="trainer -> vLLM weight sync backend")
    parser.add_argument("--vllm.enforce_eager", action="store_true", default=False, help="Disable CUDA graph in vLLM")
    parser.add_argument(
        "--vllm.tool_call_parser",
        type=str,
        default=None,
        help="vLLM parser for model-emitted tool calls, for example qwen3_coder.",
    )
    parser.add_argument(
        "--vllm.reasoning_parser",
        type=str,
        default=None,
        help="vLLM parser for model reasoning output, for example qwen3.",
    )
    parser.add_argument(
        "--vllm.router_policy",
        type=str,
        default="consistent_hash",
        help="vllm-router policy (default consistent_hash: x-session-id affinity pins a rollout's "
        "render+generate to one engine for mm-feature cache; cache_aware | round_robin | power_of_two | random)",
    )
    parser.add_argument(
        "--vllm.mtp_num_speculative_tokens",
        type=int,
        default=0,
        help="MTP speculative-decoding tokens for rollout (0=off). >0 enables multi-token "
        "prediction in vLLM; the draft is auto-detected from the checkpoint's MTP head "
        "(Qwen3.6-MoE). Lossless (target verifies every token). 1 is a good default.",
    )
    parser.add_argument("--vllm.dtype", type=str, default="bfloat16", help="vLLM inference dtype")
    parser.add_argument(
        "--vllm.kv_cache_dtype",
        type=str,
        default=None,
        help="vLLM KV cache dtype override (e.g. fp8_e4m3); unset keeps vLLM auto.",
    )
    parser.add_argument(
        "--vllm.gpu_memory_utilization",
        type=float,
        default=0.95,
        help="vLLM gpu_memory_utilization",
    )
    parser.add_argument(
        "--vllm.mm_encoder_attn_backend",
        type=str,
        default=None,
        help="Optional vLLM vision encoder attention backend, e.g. TORCH_SDPA.",
    )
    parser.add_argument(
        "--vllm.gdn_prefill_backend",
        type=str,
        choices=("flashinfer", "triton"),
        default=None,
        help="Optional vLLM GDN prefill backend for Qwen3-style linear attention layers.",
    )
    parser.add_argument(
        "--vllm.attention_backend",
        type=str,
        default=None,
        help="Optional vLLM attention backend (FLASH_ATTN/FLASHINFER/TRITON_ATTN/FLEX_ATTENTION). "
        "Pass TRITON_ATTN to bypass AOT-compiled FA2/FlashInfer kernels on older drivers.",
    )
    parser.add_argument(
        "--vllm.block_size",
        type=int,
        default=None,
        help="vLLM KV cache block size (tokens). MiniMax-M3 MSA sparse attention requires 128; "
        "vLLM's default (16) raises 'No common block size' across M3's dense+sparse layers.",
    )
    parser.add_argument(
        "--vllm.mamba_ssm_cache_dtype",
        type=str,
        choices=("auto", "float32", "float16"),
        default=None,
        help="vLLM Mamba SSM state-cache dtype. Force 'float32' for hybrid Mamba2 models "
        "(NemotronH/omni3): vLLM defaults NemotronH to float16, so the recurrent SSM scan "
        "accumulates error over the rollout and rollout log-probs drift from the fp32 training "
        "recompute (inflates vllm_kl / seq-mask-TIS filtering). Matches opd-rl/nemo-rl.",
    )
    parser.add_argument(
        "--vllm.distributed_executor_backend",
        type=str,
        choices=("ray", "mp", "uni"),
        default=None,
        help="Optional vLLM distributed executor backend override.",
    )
    parser.add_argument(
        "--vllm.enable_expert_parallel",
        action="store_true",
        default=False,
        help="Enable vLLM TP+EP hybrid: experts EP-sharded across the TP ranks (Qwen3.5/3.6 MoE).",
    )
    parser.add_argument(
        "--vllm.moe_backend",
        type=str,
        default=None,
        help="Optional vLLM MoE kernel backend (for example triton); unset keeps vLLM auto-selection.",
    )
    parser.add_argument(
        "--vllm.disable_custom_all_reduce",
        action="store_true",
        default=False,
        help=(
            "Disable vLLM's custom all-reduce kernel and fall back to NCCL. "
            "Useful when custom all-reduce fails because GPU P2P or CUDA IPC "
            "is unavailable or incompatible."
        ),
    )
    # vLLM throughput features. We leave `chunked_prefill` and `async_scheduling`
    # at None so vLLM 0.21's own auto-resolution decides (both default ON for
    # non-encoder-decoder models with mp/uniproc executors). Prefix caching is
    # an explicit opt-in: it changes the rollout→training logprob path because
    # cached prefixes survive across weight updates; the trainer calls
    # reset_prefix_cache after every broadcast (see broadcast_to_vllm) so
    # enabling it is safe, but we keep it off by default until a recipe is
    # validated end-to-end.
    parser.add_argument(
        "--vllm.enable_prefix_caching",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="vLLM prefix KV cache. Multi-turn rollouts re-prefill the growing history each turn; "
        "prefix caching cuts that cost. The trainer invalidates the cache after every weight "
        "broadcast, including the blocks held by requests that straddle it — without that, "
        "rollouts get served KV computed under the previous policy. Off by default: it makes "
        "the rollout/train logprob drift arrive sooner and larger. Incompatible with MTP "
        "speculative decoding.",
    )
    parser.add_argument(
        "--vllm.enable_chunked_prefill",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="vLLM chunked prefill (default: vLLM auto — True for non-encoder-decoder models).",
    )
    parser.add_argument(
        "--vllm.max_num_batched_tokens",
        type=int,
        default=None,
        help="vLLM scheduler token budget per iteration (default: vLLM auto, ~2048 with chunked "
        "prefill). Set >= max_model_len so every prefill fits in one chunk and a recurrent-state "
        "model (Mamba2/GDN) never hands its state across chunk boundaries.",
    )
    parser.add_argument(
        "--vllm.async_scheduling",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="vLLM async scheduling (default: vLLM auto — True for mp/uniproc executors with no spec-decode).",
    )
    parser.add_argument(
        "--vllm.decode_context_parallel_size",
        type=int,
        default=1,
        help="vLLM decode-context-parallel size; shards KV cache across decode workers for long contexts.",
    )
    # -- sampling & rollout batching --
    parser.add_argument("--rollout.batch_size", type=int, default=1024, help="Batch size for make experience")
    parser.add_argument("--rollout.gateway_count", type=int, default=2, help="Number of Ray-managed Polar gateways")
    parser.add_argument(
        "--rollout.gateway_concurrency",
        type=int,
        default=2,
        help="Concurrent container sessions per Polar gateway stage",
    )
    parser.add_argument(
        "--rollout.session_timeout",
        type=float,
        default=600.0,
        help="End-to-end timeout in seconds for each Polar session",
    )
    parser.add_argument(
        "--rollout.save_dir",
        type=str,
        default="./rollout_results",
        help="Shared directory for Polar artifacts, results, and the generated runtime topology",
    )
    parser.add_argument(
        "--rollout.vllm_generate_batch_size", type=int, default=None, help="Batch size for vLLM generating samples"
    )
    parser.add_argument("--rollout.micro_batch_size", type=int, default=1)
    parser.add_argument(
        "--rollout.max_new_tokens",
        type=int,
        default=None,
        help="Max tokens to generate per sample. If None, dynamically computed as max_len - prompt_len per sample.",
    )
    parser.add_argument("--rollout.max_tokens_per_gpu", type=int, default=None)
    parser.add_argument(
        "--rollout.n_samples_per_prompt", type=int, default=1, help="number of responses for each prompt in generation"
    )
    parser.add_argument("--rollout.top_p", type=float, default=1.0)
    parser.add_argument("--rollout.temperature", type=float, default=1.0)

    # Training (loop hyperparameters; optimizer/scheduler/grad-clip are in the shared block above)
    parser.add_argument("--train.batch_size", type=int, default=128, help="Global training batch size")
    parser.add_argument("--train.micro_batch_size", type=int, default=1, help="batch size per GPU")
    parser.add_argument("--train.max_tokens_per_gpu", type=int, default=16192)
    parser.add_argument("--train.max_epochs", type=int, default=1)
    parser.add_argument("--train.num_episodes", type=int, default=1)
    parser.add_argument("--train.seed", type=int, default=42)
    parser.add_argument(
        "--train.full_determinism_enable",
        action="store_true",
        default=False,
        help="Enable reproducible behavior during distributed training",
    )
    # Dynamic batch changes microbatch grouping; the packed representation
    # still follows the loaded actor model path.
    parser.add_argument(
        "--train.dynamic_batch_enable",
        action="store_true",
        default=False,
        help="Group samples into dynamic microbatches by token budget.",
    )
    parser.add_argument(
        "--train.force_on_policy",
        action="store_true",
        default=False,
        help=(
            "Force true on-policy updates: accumulate gradients over the ENTIRE flattened rollout "
            "batch and run a single optimizer step at the final microbatch, instead of splitting "
            "the rollout into several train.batch_size / grad-accum windows. Needed for multi-turn "
            "flatten, where the per-rollout sample count is variable and a fixed train.batch_size "
            "would make every step after the first off-policy and drop the trailing samples. "
            "Requires --train.max_epochs 1."
        ),
    )

    # Distributed: Ray actor/ref placement + async pipelining (FSDP TP/CP/EP sizes are in the shared block above)
    parser.add_argument("--actor.num_nodes", type=int, default=1, help="number of nodes for actor")
    parser.add_argument("--actor.num_gpus_per_node", type=int, default=8, help="number of gpus per node for actor")
    parser.add_argument("--ref.num_nodes", type=int, default=1, help="number of nodes for reference")
    parser.add_argument("--ref.num_gpus_per_node", type=int, default=8, help="number of gpus per node for reference")
    parser.add_argument(
        "--train.colocate_fsdp_models",
        action="store_true",
        default=False,
        help="Colocate the FSDP models (actor, reference, critic) on the actor's GPUs (they time-slice the same GPUs).",
    )
    parser.add_argument("--train.async_queue_size", type=int, default=1, help="Queue size for async sampler<->trainer")
    parser.add_argument(
        "--train.partial_rollout_enable",
        action="store_true",
        default=False,
        help="Use vLLM pause/resume during weight sync so generation can overlap with training.",
    )
    parser.add_argument(
        "--train.force_sync_mode",
        action="store_true",
        default=False,
        help="Strictly on-policy: free the rollout slot only AFTER train_step refits vLLM, so the "
        "next batch is generated with the same weights the trainer recomputes it under. Removes the "
        "1-step-stale rollout that inflates vllm_kl on routing-sensitive MoE checkpoints, at the "
        "cost of the generate/train overlap.",
    )
    # Debug / repro: dump a rollout batch and replay it train-only (skip generation) to
    # iterate on the training+refit path in isolation; check_weight_update_equal checks every broadcast.
    parser.add_argument(
        "--train.rollout_dump_dir",
        type=str,
        default=None,
        help="Save each rollout batch to <dir>/rollout_step{N}.pt for later train-only replay.",
    )
    parser.add_argument(
        "--train.rollout_replay_dir",
        type=str,
        default=None,
        help="Load <dir>/rollout_step{N}.pt instead of generating (train-only replay).",
    )
    parser.add_argument(
        "--train.check_weight_update_equal",
        action="store_true",
        default=False,
        help="After each vLLM weight broadcast, warn which params vLLM did NOT refresh (stale rollout weights).",
    )

    # Eval
    parser.add_argument("--eval.dataset", type=str, default=None, help="Path to the evaluation dataset")
    parser.add_argument(
        "--eval.batch_size",
        type=int,
        default=None,
        help="Concurrent eval prompt groups; defaults to --rollout.batch_size when unset.",
    )
    parser.add_argument("--eval.split", type=str, default="train")
    parser.add_argument("--eval.steps", type=int, default=-1, help="Evaluate every N steps; -1 disables eval.")
    parser.add_argument(
        "--eval.temperature",
        type=float,
        default=None,
        help="Eval temperature; falls back to --rollout.temperature when unset.",
    )
    parser.add_argument(
        "--eval.top_p", type=float, default=None, help="Eval top-p; falls back to --rollout.top_p when unset."
    )
    parser.add_argument(
        "--eval.max_new_tokens",
        type=int,
        default=None,
        help="Eval max new tokens; falls back to --rollout.max_new_tokens when unset.",
    )
    parser.add_argument(
        "--eval.n_samples_per_prompt",
        type=int,
        default=None,
        help="Eval samples per prompt; falls back to --rollout.n_samples_per_prompt when unset.",
    )
    parser.add_argument(
        "--eval.eval_at_start",
        action="store_true",
        help="Run one baseline eval at global_step 0 (before any update) to measure the pre-RL model. "
        "Fresh runs only — gated on the consumed-prompt counter being 0, so a resume (which loads a "
        "non-zero step) does not add a redundant eval.",
    )
    parser.add_argument(
        "--eval.eval_only",
        action="store_true",
        help="Score --eval.dataset once and exit — no training. vLLM already holds the HF weights, so "
        "the policy/ref/critic FSDP actors are never built and their GPUs go to the eval (use all nodes "
        "for vLLM engines / env runners, or fewer nodes). Requires --eval.dataset.",
    )

    # Runtime / misc
    parser.add_argument("--local_rank", type=int, default=-1, help="local_rank from torchrun")
    parser.add_argument("--use_ms", action="store_true", default=False, help="Resolve models from ModelScope hub.")

    args = parser.parse_args()
    from molt.utils.config import hierarchize

    args = hierarchize(args)
    resolve_ckpt_retention(args.ckpt)

    # ============================ Validate / derive arguments ============================
    # NOTE: ordering matters where a check derives state a later check reads.

    # --- Required inputs ---
    if not args.actor.model_name_or_path:
        raise ValueError("--actor.model_name_or_path is required")

    if not args.vllm.num_engines or args.vllm.num_engines <= 0:
        raise NotImplementedError(
            "RL rollout currently requires vLLM. Set --vllm.num_engines > 0; "
            "actor-side generation fallback is not wired in this AutoModel path."
        )

    args.rollout.task = None
    if args.rollout.task_spec:
        from polar.rollout.models import TaskSpec

        task_path = Path(args.rollout.task_spec).resolve()
        with task_path.open() as task_file:
            task_payload = yaml.safe_load(task_file) or {}
        args.rollout.task = TaskSpec.model_validate(
            task_payload, context={"base_dir": task_path.parent}
        ).model_dump(mode="json")

    # --- Algorithm setup & defaults ---
    if args.actor.eps_clip_low_high is None:
        # Default to the standard symmetric PPO clip; every launch script passes
        # --actor.eps_clip_low_high explicitly, so this is just the bare-CLI default.
        args.actor.eps_clip_low_high = (0.2, 0.2)

    if args.algo.kl.init_coef is None:
        args.algo.kl.init_coef = 0.01

    # --- Agent / rollout ---
    if not args.vllm.tool_call_parser:
        raise ValueError("Polar rollout requires --vllm.tool_call_parser (for example qwen3_coder).")
    if args.vllm.router_policy != "consistent_hash":
        raise ValueError("Polar rollout requires --vllm.router_policy consistent_hash for session affinity.")
    if args.rollout.gateway_count <= 0 or args.rollout.gateway_concurrency <= 0:
        raise ValueError("Polar gateway count and concurrency must both be positive.")
    if args.rollout.session_timeout <= 0:
        raise ValueError("--rollout.session_timeout must be positive.")

    # Set vLLM generate_batch_size to rollout_batch_size if not specified
    if not args.rollout.vllm_generate_batch_size:
        args.rollout.vllm_generate_batch_size = args.rollout.batch_size

    # --- Algorithm checks ---
    # Group-relative estimators need >1 sample per prompt to form a baseline during training;
    # eval-only never trains, so skip that gate (eval uses --eval.n_samples_per_prompt).
    if not args.eval.eval_only and args.algo.advantage.estimator in ["rloo", "reinforce_baseline", "grpo", "dr_grpo"]:
        assert args.rollout.n_samples_per_prompt > 1, (
            f"{args.algo.advantage.estimator} requires n_samples_per_prompt > 1"
        )

    if args.algo.kl.use_loss and args.algo.kl.estimator not in ("k2", "k3"):
        print(f"Recommend setting {args.algo.kl.estimator} to 'k2' or 'k3' when using KL as a loss")
    elif not args.algo.kl.use_loss and args.algo.kl.estimator != "k1":
        print(f"Recommend setting {args.algo.kl.estimator} to 'k1' when not using KL as a loss.")

    if args.algo.dynamic_filtering_enable:
        assert args.algo.dynamic_filtering_range[0] < args.algo.dynamic_filtering_range[1], (
            "dynamic_filtering_range[0] must be less than dynamic_filtering_range[1]"
        )
        assert args.rollout.n_samples_per_prompt > 1, (
            "n_samples_per_prompt must be greater than 1 when using dynamic filtering"
        )

    if args.algo.advantage.is_correction_level == "off":
        # The HTTP path has no per-token policy-boundary mask. Async and partial
        # rollout can cross broadcasts between requests, so require IS correction.
        if args.train.async_queue_size > 1 or args.train.partial_rollout_enable:
            raise ValueError(
                "Off-policy rollout (--train.async_queue_size > 1 or --train.partial_rollout_enable) "
                "produces tokens across weight broadcasts that the router path does NOT mask "
                "(off_policy_len is always 0 over HTTP). Set --algo.advantage.is_correction_level "
                "(token|seq|geo) to correct them, or run strictly on-policy (--train.async_queue_size 1 "
                "AND --train.force_sync_mode, no --train.partial_rollout_enable). Note: async_queue_size 1 "
                "alone frees the rollout slot before the refit, so the next batch is still 1-step stale."
            )
        if not args.train.force_sync_mode:
            print(
                "[Warning] Rollout samples may be off-policy. Set "
                "--algo.advantage.is_correction_level (token|seq|geo) to correct rollout logprobs during training."
            )
    elif args.train.partial_rollout_enable:
        # A session may continue under new weights after a drained broadcast.
        print(
            "[Warning] --train.partial_rollout_enable can continue sessions across weight broadcasts. "
            "Per-token IS is correcting those off-policy tokens."
        )

    if args.algo.advantage.is_correction_level != "off" and args.rollout.top_p < 1.0:
        # vLLM computes `processed_logprobs` AFTER the top-p mask, so they are renormalized over the
        # kept nucleus while training recomputes over the full vocabulary. Every rollout log-prob is
        # then offset by -log(kept mass), biasing vllm_kl and the IS ratio on every token.
        raise ValueError(
            f"--rollout.top_p {args.rollout.top_p} biases the rollout log-probs the IS correction "
            "consumes; use --rollout.top_p 1.0, or --algo.advantage.is_correction_level off."
        )

    # --- Data ---
    if args.data.max_images_per_prompt > 0 and args.fsdp.packing_samples:
        print("[Warning] VLM training does not support --fsdp.packing_samples; disabling packing for this run.")
        args.fsdp.packing_samples = False

    # --- Parallelism / FSDP ---
    if args.fsdp.pp_size > 1:
        raise NotImplementedError("Molt trainers are not pipeline-parallel aware yet; set --fsdp.pp_size 1")

    if args.vllm.enable_prefix_caching and args.vllm.mtp_num_speculative_tokens > 0:
        # Isolation-tested: each feature alone is logprob-clean, together they
        # inflate vllm_kl ~10x (spec-decode KV rollback vs cached-block reuse),
        # and the seq-mask-tis band then drops half the batch. Engine-side issue;
        # refuse the combination.
        raise ValueError(
            "--vllm.mtp_num_speculative_tokens is incompatible with --vllm.enable_prefix_caching: "
            "speculative decoding corrupts rollout logprobs when prefix-cached blocks are "
            "reused. Enable at most one of the two."
        )

    if args.fsdp.packing_samples:
        assert args.vllm.num_engines > 0, "Only support `--fsdp.packing_samples` with vLLM."
        # tilelang joins te/fa2: DSA (glm_moe_dsa) is THD-native and *requires* packing.
        if args.fsdp.attn_implementation not in {"te", "flash_attention_2", "tilelang"}:
            raise ValueError(
                "--fsdp.packing_samples requires --fsdp.attn_implementation te, flash_attention_2, or tilelang."
            )

    # --- Training / rollout sizing ---
    if args.train.dynamic_batch_enable:
        if args.rollout.max_tokens_per_gpu is None:
            print("[Warning] Set --rollout.max_tokens_per_gpu to --train.max_tokens_per_gpu.")
            args.rollout.max_tokens_per_gpu = args.train.max_tokens_per_gpu

    if args.train.force_on_policy and args.train.max_epochs != 1:
        raise ValueError(
            "--train.force_on_policy requires --train.max_epochs 1: max_epochs is the number of PPO "
            "epochs over each rollout, and on-policy training must take exactly one pass — any epoch "
            "after the first trains on data the (now updated) weights did not generate."
        )

    if not args.eval.eval_only:
        assert (
            args.rollout.n_samples_per_prompt * args.rollout.batch_size // args.rollout.micro_batch_size
            >= args.actor.num_nodes * args.actor.num_gpus_per_node // get_model_parallel_size(args)
        ), "The number of sample batches must be greater than or equal to the effective number of actor processes."

    # --- Eval ---
    if args.eval.batch_size is not None and args.eval.batch_size <= 0:
        raise ValueError(f"--eval.batch_size must be greater than zero, got {args.eval.batch_size}.")

    # --- Runtime ---
    if args.use_ms:
        from modelscope.utils.hf_util import patch_hub

        # Patch hub to download models from modelscope to speed up.
        patch_hub()

    train(args)
