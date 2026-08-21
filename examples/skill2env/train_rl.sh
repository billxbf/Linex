#!/usr/bin/env bash
# Eight-GPU Skill2Env acceptance run: four actor GPUs and four vLLM GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
PROMPT_DATASET="${PROMPT_DATASET:-$SCRIPT_DIR/training.jsonl}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/skill2env-rl-qwen3.5-9b}"

test -f "$PROMPT_DATASET"
test "$(wc -l < "$PROMPT_DATASET")" -ge 40
command -v apptainer >/dev/null || { echo "Skill2Env RL requires Apptainer in PATH"; exit 1; }
mkdir -p "$SAVE_ROOT"
ulimit -n 65536 2>/dev/null || { echo "Skill2Env RL requires a nofile limit of at least 65536"; exit 1; }

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP16=0
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus=8 --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && ray stop --force >/dev/null 2>&1 || true' EXIT

RESUME_ARGS=()
[ "${RESUME:-0}" = "1" ] && RESUME_ARGS=(--ckpt.load_enable --ckpt.warm_resume_rollouts)

cd "$REPO_ROOT"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$PROMPT_DATASET" \
  --data.input_key prompt \
  --data.max_samples 40 \
  --data.max_len 131072 \
  --rollout.gateway_count 8 \
  --rollout.gateway_concurrency 16 \
  --rollout.session_timeout 3600 \
  --rollout.save_dir "$SAVE_ROOT/rollouts" \
  --rollout.batch_size 4 \
  --rollout.vllm_generate_batch_size 8 \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 16 \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --rollout.max_new_tokens 4096 \
  --rollout.max_tokens_per_gpu 8192 \
  --train.batch_size 64 \
  --train.micro_batch_size 1 \
  --train.max_tokens_per_gpu 8192 \
  --train.dynamic_batch_enable \
  --train.max_epochs 1 \
  --train.num_episodes 1 \
  --train.async_queue_size 2 \
  --train.force_on_policy \
  --train.rollout_dump_dir "$SAVE_ROOT/rollout_dumps" \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node 4 \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size 4 \
  --vllm.tool_call_parser qwen3_coder \
  --vllm.reasoning_parser qwen3 \
  --vllm.sync_backend nccl \
  --vllm.disable_custom_all_reduce \
  --vllm.enforce_eager \
  --vllm.gpu_memory_utilization 0.9 \
  --vllm.distributed_executor_backend mp \
  --vllm.gdn_prefill_backend triton \
  --vllm.mamba_ssm_cache_dtype float32 \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation te \
  --fsdp.tp_size 1 \
  --fsdp.ep_size 1 \
  --fsdp.cp_size 4 \
  --actor.freeze_visual_encoder \
  --actor.gradient_checkpoint full \
  --actor.adam.lr 1e-6 \
  --actor.eps_clip_low_high 0.2 0.27 \
  --actor.dual_clip 10.0 \
  --algo.advantage.estimator reinforce_baseline \
  --algo.advantage.is_correction_level geo \
  --algo.advantage.is_correction_threshold 0.5 5.0 \
  --algo.kl.init_coef 0 \
  --reward.clip_range -10 10 \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps 5 \
  --ckpt.max_num 2 \
  --logger.logging_steps 1 \
  --logger.tensorboard_dir "$SAVE_ROOT/tensorboard" \
  --logger.wandb.project skill2env_rl \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-skill2env_qwen3.5_9b_$$}" \
  "${RESUME_ARGS[@]}" \
  "$@"
