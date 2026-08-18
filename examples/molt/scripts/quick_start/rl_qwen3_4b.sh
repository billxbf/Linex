#!/bin/bash
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

# Single-node quick-start: Qwen3-4B container-agent RL through Polar.
#
# 8 GPUs on one machine, split 4 actor + 4 vLLM rollout. No Slurm. The Molt job
# owns Ray, vLLM, training, and the generated Polar topology. The
# calculator task image must already exist on every node that can host a gateway.
#
#   MODEL_PATH=/path/to/Qwen3-4B bash examples/molt/scripts/quick_start/rl_qwen3_4b.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:?Set MODEL_PATH to a Qwen3-4B checkpoint.}"

PROMPT_DATASET="${PROMPT_DATASET:-$REPO_ROOT/examples/polar/calculator/prompts.jsonl}"
TASK_SPEC="${TASK_SPEC-$REPO_ROOT/examples/polar/calculator/task.yaml}"
test -e "$PROMPT_DATASET" || { echo "PROMPT_DATASET not found: $PROMPT_DATASET"; exit 1; }
[ -z "$TASK_SPEC" ] || test -e "$TASK_SPEC" || { echo "TASK_SPEC not found: $TASK_SPEC"; exit 1; }
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/quick_start-qwen3-4b/run}"
TASK_SPEC_ARGS=()
[ -z "$TASK_SPEC" ] || TASK_SPEC_ARGS=(--rollout.task_spec "$TASK_SPEC")

GPUS_PER_NODE="${GPUS_PER_NODE:-8}"
ACTOR_GPUS="${ACTOR_GPUS:-4}"
VLLM_TP="${VLLM_TP:-4}"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export RAY_USAGE_STATS_ENABLED=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export VLLM_USE_FLASHINFER_MOE_FP16=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
if ! ray status >/dev/null 2>&1; then
  ray start --head --num-gpus="$GPUS_PER_NODE" --disable-usage-stats >/dev/null
  STARTED_RAY=1
else
  STARTED_RAY=0
fi
trap '[ "$STARTED_RAY" = "1" ] && ray stop --force >/dev/null 2>&1 || true' EXIT

cd "$REPO_ROOT"
python3 -u -m molt.cli.train_rl_ray \
  --actor.model_name_or_path "$MODEL_PATH" \
  --data.prompt_dataset "$PROMPT_DATASET" \
  --data.input_key prompt \
  --data.max_samples "${MAX_SAMPLES:-1}" \
  --data.max_len "${MAX_LENGTH:-65536}" \
  "${TASK_SPEC_ARGS[@]}" \
  --rollout.gateway_count "${POLAR_GATEWAY_COUNT:-2}" \
  --rollout.gateway_concurrency "${POLAR_GATEWAY_CONCURRENCY:-2}" \
  --rollout.session_timeout "${POLAR_SESSION_TIMEOUT:-1200}" \
  --rollout.save_dir "$SAVE_ROOT/rollouts" \
  --rollout.batch_size 1 \
  --rollout.vllm_generate_batch_size 1 \
  --rollout.micro_batch_size 1 \
  --rollout.n_samples_per_prompt 4 \
  --rollout.temperature 1.0 \
  --train.batch_size 4 \
  --train.micro_batch_size 1 \
  --train.max_epochs 1 \
  --train.num_episodes 1 \
  --train.async_queue_size 1 \
  --train.force_sync_mode \
  --train.force_on_policy \
  --train.colocate_fsdp_models \
  --actor.num_nodes 1 \
  --actor.num_gpus_per_node "$ACTOR_GPUS" \
  --ref.num_nodes 1 \
  --ref.num_gpus_per_node "$ACTOR_GPUS" \
  --vllm.num_engines 1 \
  --vllm.tensor_parallel_size "$VLLM_TP" \
  --vllm.tool_call_parser "${VLLM_TOOL_CALL_PARSER:-qwen3_coder}" \
  --vllm.reasoning_parser "${VLLM_REASONING_PARSER:-qwen3}" \
  --vllm.sync_backend nccl \
  --vllm.gpu_memory_utilization 0.8 \
  --vllm.distributed_executor_backend mp \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation flash_attention_2 \
  --fsdp.tp_size 1 \
  --fsdp.ep_size 1 \
  --fsdp.cp_size 1 \
  --fsdp.packing_samples \
  --actor.gradient_checkpoint full \
  --actor.adam.lr 1e-6 \
  --actor.eps_clip_low_high 0.2 0.27 \
  --actor.dual_clip 10.0 \
  --algo.advantage.estimator reinforce_baseline \
  --algo.kl.use_loss \
  --algo.kl.estimator k2 \
  --algo.kl.init_coef 0.001 \
  --reward.clip_range -10 10 \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.save_steps 5 \
  --logger.logging_steps 1 \
  --logger.wandb.project "${WANDB_PROJECT:-molt_quickstart_qwen3_4b}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-qwen3_4b_quickstart_$$}"
