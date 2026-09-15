#!/usr/bin/env bash
# Async GRPO + binary-KL DPPO: Qwen3.8-27B, four vLLM engines and four FSDP GPUs.
# Keep the existing 96k rollout context and 64k training cap with CP4 and CPU optimizer state.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-/raid/binfeng/models/Qwen3.8-27B}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/skill2env_rl_dppo}"
PROMPT_DATASET="${PROMPT_DATASET:-$SAVE_ROOT/train_tasks.jsonl}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-98304}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-8}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-64}"
INFLIGHT_ROLLOUTS="${INFLIGHT_ROLLOUTS:-128}"

# The CLI counts prompt groups; these environment variables count individual rollouts.
if (( SAMPLES_PER_PROMPT <= 0 || ROLLOUT_BATCH_SIZE <= 0 || INFLIGHT_ROLLOUTS <= 0 ||
      ROLLOUT_BATCH_SIZE % SAMPLES_PER_PROMPT != 0 || INFLIGHT_ROLLOUTS % SAMPLES_PER_PROMPT != 0 )); then
  echo "Batch and in-flight rollout counts must be positive multiples of SAMPLES_PER_PROMPT" >&2
  exit 1
fi

test -d "$MODEL_PATH" || { echo "Policy model not found: $MODEL_PATH"; exit 1; }
test -s "$PROMPT_DATASET" || { echo "Prepared task dataset not found: $PROMPT_DATASET (run prepare.py)"; exit 1; }
if [ -z "${NVIDIA_API_KEY:-}" ] && grep -Eq '"judge_api_key_env"[[:space:]]*:[[:space:]]*"NVIDIA_API_KEY"' "$PROMPT_DATASET"; then
  echo "NVIDIA_API_KEY is required by the prepared rubric tasks" >&2
  exit 1
fi
command -v apptainer >/dev/null || { echo "Skill2Env RL requires Apptainer in PATH"; exit 1; }
mkdir -p "$SAVE_ROOT"
ulimit -n 65536 2>/dev/null || { echo "Skill2Env RL requires a nofile limit of at least 65536"; exit 1; }

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# Task containers have their own memory limits; the Ray monitor cannot reclaim their memory.
export RAY_memory_monitor_refresh_ms=0
export MOLT_DEFER_GRAD_SYNC="${MOLT_DEFER_GRAD_SYNC:-0}"

if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus="${RAY_GPUS:-8}" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && ray stop --force >/dev/null 2>&1 || true' EXIT

RESUME_ARGS=()
[ "${RESUME:-0}" = "1" ] && RESUME_ARGS=(--ckpt.load_enable)
EVAL_ARGS=()
if [ -n "${EVAL_DATASET:-}" ]; then
  EVAL_ARGS=(--eval.dataset "$EVAL_DATASET" --eval.steps "${EVAL_STEPS:-50}" --eval.eval_at_start --eval.n_samples_per_prompt "${EVAL_SAMPLES_PER_PROMPT:-1}")
fi
PREFIX_CACHE_ARGS=()
[ "${VLLM_PREFIX_CACHING:-1}" = "1" ] && PREFIX_CACHE_ARGS=(--vllm.enable_prefix_caching)
# Partial rollout lets weight refits interrupt generation. DPPO uses each token's stored
# sampling probability even when one session spans several weight updates.
PARTIAL_ROLLOUT_ARGS=()
[ "${PARTIAL_ROLLOUT:-1}" = "1" ] && PARTIAL_ROLLOUT_ARGS=(--train.partial_rollout_enable)
# One sample per microbatch bounds padded training memory; token-budget bins can pad
# beyond their summed-length budget. Dynamic batching remains opt-in.
DYNAMIC_BATCH_ARGS=()
[ "${DYNAMIC_BATCH:-0}" = "1" ] && DYNAMIC_BATCH_ARGS=(--train.dynamic_batch_enable --train.max_tokens_per_gpu "${MAX_TOKENS_PER_GPU:-16384}")
WANDB_ARGS=()
[ -n "${WANDB_KEY:-}" ] && WANDB_ARGS=(--logger.wandb.key "$WANDB_KEY")

cd "$REPO_ROOT"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$PROMPT_DATASET" \
  --data.input_key prompt \
  --data.max_samples "${MAX_SAMPLES:-100000}" \
  --data.max_len "$CONTEXT_LENGTH" \
  --data.train_max_len "${TRAIN_MAX_LEN:-65536}" \
  --rollout.gateway_count "${GATEWAY_COUNT:-4}" \
  --rollout.gateway_concurrency "${GATEWAY_CONCURRENCY:-32}" \
  --rollout.session_timeout "${SESSION_TIMEOUT:-3600}" \
  --rollout.save_dir "$SAVE_ROOT/rollouts" \
  --rollout.batch_size "$((ROLLOUT_BATCH_SIZE / SAMPLES_PER_PROMPT))" \
  --rollout.vllm_generate_batch_size "$((INFLIGHT_ROLLOUTS / SAMPLES_PER_PROMPT))" \
  --rollout.n_samples_per_prompt "$SAMPLES_PER_PROMPT" \
  --rollout.temperature 1.0 \
  --rollout.top_p 1.0 \
  --rollout.max_new_tokens "${MAX_NEW_TOKENS:-32768}" \
  --train.batch_size "$ROLLOUT_BATCH_SIZE" \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes "${NUM_EPISODES:-1}" \
  --train.async_queue_size "${ASYNC_QUEUE_SIZE:-2}" \
  --train.force_on_policy \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "${ACTOR_GPUS:-4}" \
  --actor.freeze_visual_encoder \
  --actor.gradient_checkpoint full \
  --actor.adam.lr "${LR:-1e-6}" \
  --actor.lr_scheduler constant \
  --actor.max_norm 1.0 \
  --algo.advantage.estimator grpo \
  --actor.loss_mode dppo \
  --actor.dppo_kl_threshold "${DPPO_KL_THRESHOLD:-0.05}" \
  --algo.advantage.is_correction_level off \
  --algo.kl.init_coef 0 \
  --vllm.num_engines "${VLLM_NUM_ENGINES:-4}" \
  --vllm.tensor_parallel_size "${VLLM_TP_SIZE:-1}" \
  --vllm.tool_call_parser "${TOOL_CALL_PARSER:-qwen3_xml}" \
  --vllm.reasoning_parser "${REASONING_PARSER:-qwen3}" \
  --vllm.gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  --vllm.max_num_batched_tokens "${VLLM_MAX_NUM_BATCHED_TOKENS:-16384}" \
  --vllm.mamba_ssm_cache_dtype float32 \
  --vllm.gdn_prefill_backend triton \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation "${FSDP_ATTN_IMPLEMENTATION:-sdpa}" \
  --fsdp.cp_size "${CP_SIZE:-4}" \
  --fsdp.offload "${FSDP_OFFLOAD:-optimizer}" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.save_steps "${SAVE_STEPS:-25}" \
  --ckpt.save_hf \
  --ckpt.max_num "${MAX_CHECKPOINTS:-10}" \
  --ckpt.dcp_max_num 2 \
  --logger.logging_steps 1 \
  --logger.tensorboard_dir "$SAVE_ROOT/tensorboard" \
  --logger.wandb.project "${WANDB_PROJECT:-skill2env_rl_dppo}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-dppo_g${SAMPLES_PER_PROMPT}_b${ROLLOUT_BATCH_SIZE}_$$}" \
  "${RESUME_ARGS[@]}" "${EVAL_ARGS[@]}" "${PREFIX_CACHE_ARGS[@]}" "${PARTIAL_ROLLOUT_ARGS[@]}" "${DYNAMIC_BATCH_ARGS[@]}" "${WANDB_ARGS[@]}" \
  "$@"
