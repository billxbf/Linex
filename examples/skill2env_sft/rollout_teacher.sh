#!/usr/bin/env bash
# Generate Skill2Env teacher sessions with a TP8+EP vLLM engine, then export them.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/skill2env_sft}"
TASK_DATASET="${TASK_DATASET:-$SAVE_ROOT/teacher_tasks.jsonl}"
MODEL_PATH="${MODEL_PATH:-/raid/binfeng/models/Inferact/GLM-5.3-NVFP4}"
CONTEXT_LENGTH="${CONTEXT_LENGTH:-131072}"

test -d "$MODEL_PATH" || { echo "Teacher model not found: $MODEL_PATH"; exit 1; }
test -s "$TASK_DATASET" || { echo "Prepared task dataset not found: $TASK_DATASET"; exit 1; }
command -v apptainer >/dev/null || { echo "Teacher rollout requires Apptainer in PATH"; exit 1; }
command -v ray >/dev/null || { echo "Teacher rollout requires Ray in PATH"; exit 1; }
mkdir -p "$SAVE_ROOT"
ulimit -n 65536 2>/dev/null || { echo "Teacher rollout requires a nofile limit of at least 65536"; exit 1; }

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"
export VLLM_ALLREDUCE_USE_SYMM_MEM="${VLLM_ALLREDUCE_USE_SYMM_MEM:-0}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus="${RAY_GPUS:-8}" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && ray stop --force >/dev/null 2>&1 || true' EXIT

# CUDA graphs on by default: eager decode on the 78-layer MoE is kernel-launch
# bound and slow enough that a long turn can outlive pi's request patience,
# truncating sessions mid-task.
VLLM_EAGER_ARGS=()
[ "${VLLM_ENFORCE_EAGER:-0}" = "1" ] && VLLM_EAGER_ARGS=(--vllm.enforce_eager)

cd "$REPO_ROOT"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --eval.dataset "$TASK_DATASET" \
  --eval.eval_only \
  --eval.batch_size "${EVAL_BATCH_SIZE:-64}" \
  --eval.n_samples_per_prompt "${TRAJECTORIES_PER_TASK:-1}" \
  --eval.max_new_tokens "${MAX_NEW_TOKENS:-32768}" \
  --data.input_key prompt \
  --data.max_len "$CONTEXT_LENGTH" \
  --rollout.gateway_count "${GATEWAY_COUNT:-1}" \
  --rollout.gateway_concurrency "${GATEWAY_CONCURRENCY:-64}" \
  --rollout.session_timeout "${SESSION_TIMEOUT:-3600}" \
  --rollout.save_dir "$SAVE_ROOT/rollouts" \
  --rollout.batch_size "${EVAL_BATCH_SIZE:-8}" \
  --rollout.temperature "${TEMPERATURE:-1.0}" \
  --rollout.top_p "${TOP_P:-1.0}" \
  --vllm.num_engines "${VLLM_NUM_ENGINES:-1}" \
  --vllm.tensor_parallel_size "${VLLM_TP_SIZE:-8}" \
  --vllm.pipeline_parallel_size "${VLLM_PP_SIZE:-1}" \
  --vllm.data_parallel_size "${VLLM_DP_SIZE:-1}" \
  --vllm.enable_expert_parallel \
  --vllm.kv_cache_dtype "${VLLM_KV_CACHE_DTYPE:-fp8_e4m3}" \
  --vllm.tool_call_parser "${TOOL_CALL_PARSER:-glm47}" \
  --vllm.reasoning_parser "${REASONING_PARSER:-glm45}" \
  --vllm.gpu_memory_utilization "${VLLM_GPU_MEMORY_UTILIZATION:-0.9}" \
  --vllm.distributed_executor_backend "${VLLM_EXECUTOR_BACKEND:-mp}" \
  --vllm.disable_custom_all_reduce \
  "${VLLM_EAGER_ARGS[@]}" \
  "$@"

EXPORT_ARGS=()
[ "${REJECTION_SAMPLING:-0}" = "1" ] && EXPORT_ARGS=(--rejection-sampling)

python3 -m examples.skill2env_sft.export_sft \
  "$SAVE_ROOT/rollouts" "$SAVE_ROOT/teacher_sft.jsonl" \
  "${EXPORT_ARGS[@]}"
test -s "$SAVE_ROOT/teacher_sft.jsonl" || { echo "No eligible teacher trajectories were exported"; exit 1; }
