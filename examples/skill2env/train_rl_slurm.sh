#!/usr/bin/env bash
#SBATCH --nodes=4
#SBATCH --gpus-per-node=8
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --job-name=skill2env-rl

# Four-node Skill2Env run: 16 actor GPUs and 16 vLLM GPUs.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-${SLURM_SUBMIT_DIR:-$(cd "$SCRIPT_DIR/../.." && pwd)}}"
CONTAINER_IMAGE="${CONTAINER_IMAGE:?Set CONTAINER_IMAGE to the Molt training image.}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-9B}"
PROMPT_DATASET="${PROMPT_DATASET:-$SCRIPT_DIR/training.jsonl}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/skill2env-rl-qwen3.5-9b-$SLURM_JOB_ID}"
RAY_PORT="${RAY_PORT:-6379}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8265}"

test "$SLURM_JOB_NUM_NODES" -eq 4
test -f "$PROMPT_DATASET"
test "$(wc -l < "$PROMPT_DATASET")" -ge 40
mkdir -p "$SAVE_ROOT"
ulimit -n 65536 2>/dev/null || { echo "Skill2Env RL requires a nofile limit of at least 65536"; exit 1; }

mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_NODE="${NODES[0]}"
HEAD_IP="$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address | awk '{print $1}')"
MOUNTS="${CONTAINER_MOUNTS:-$REPO_ROOT:/molt,/raid:/raid,$HOME/.cache:/root/.cache,/dev/shm:/dev/shm}"
CONTAINER_ARGS=(--overlap --no-container-mount-home --container-image="$CONTAINER_IMAGE" --container-mounts="$MOUNTS")
RAY_ENV="cd /molt && export HF_HOME=/root/.cache/huggingface TOKENIZERS_PARALLELISM=true RAY_USAGE_STATS_ENABLED=0 VLLM_WORKER_MULTIPROC_METHOD=spawn VLLM_USE_FLASHINFER_MOE_FP16=0 NCCL_NVLS_ENABLE=${NCCL_NVLS_ENABLE:-0} VLLM_ALLREDUCE_USE_SYMM_MEM=${VLLM_ALLREDUCE_USE_SYMM_MEM:-0} PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" "${CONTAINER_ARGS[@]}" \
  bash -lc "$RAY_ENV && ray start --head --node-ip-address=$HEAD_IP --port=$RAY_PORT --dashboard-host=0.0.0.0 --dashboard-port=$DASHBOARD_PORT --num-gpus=8 --block --disable-usage-stats" &
for NODE in "${NODES[@]:1}"; do
  srun --nodes=1 --ntasks=1 -w "$NODE" "${CONTAINER_ARGS[@]}" \
    bash -lc "$RAY_ENV && ray start --address=$HEAD_IP:$RAY_PORT --num-gpus=8 --block --disable-usage-stats" &
done
trap 'srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" "${CONTAINER_ARGS[@]}" bash -lc "ray stop --force" >/dev/null 2>&1 || true' EXIT

until srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" "${CONTAINER_ARGS[@]}" bash -lc \
  "$RAY_ENV && ray status | grep -q \"0 pending\""; do sleep 5; done

RL_ARGS=(
  --actor.model_name_or_path "$MODEL_PATH"
  --data.prompt_dataset "$PROMPT_DATASET" --data.input_key prompt --data.max_samples 40 --data.max_len 131072
  --rollout.gateway_count 32 --rollout.gateway_concurrency 8 --rollout.session_timeout 3600
  --rollout.save_dir "$SAVE_ROOT/rollouts" --rollout.batch_size 4 --rollout.vllm_generate_batch_size 8
  --rollout.micro_batch_size 1 --rollout.n_samples_per_prompt 16 --rollout.temperature 1.0 --rollout.top_p 1.0
  --rollout.max_new_tokens 4096
  --rollout.max_tokens_per_gpu 8192
  --train.batch_size 64 --train.micro_batch_size 1 --train.max_tokens_per_gpu 8192 --train.dynamic_batch_enable
  --train.max_epochs 1 --train.num_episodes 1 --train.async_queue_size 2 --train.force_on_policy
  --train.rollout_dump_dir "$SAVE_ROOT/rollout_dumps"
  --actor.num_nodes 2 --actor.num_gpus_per_node 8
  --vllm.num_engines 4 --vllm.tensor_parallel_size 4 --vllm.distributed_executor_backend ray
  --vllm.tool_call_parser qwen3_coder --vllm.reasoning_parser qwen3 --vllm.sync_backend nccl --vllm.disable_custom_all_reduce --vllm.enforce_eager
  --vllm.gpu_memory_utilization 0.9 --vllm.gdn_prefill_backend triton --vllm.mamba_ssm_cache_dtype float32
  --fsdp.param_dtype bf16 --fsdp.attn_implementation te
  --fsdp.tp_size 1 --fsdp.ep_size 1 --fsdp.cp_size 4
  --actor.freeze_visual_encoder --actor.gradient_checkpoint full --actor.adam.lr 1e-6
  --actor.eps_clip_low_high 0.2 0.27 --actor.dual_clip 10.0
  --algo.advantage.estimator reinforce_baseline --algo.advantage.is_correction_level geo
  --algo.advantage.is_correction_threshold 0.5 5.0 --algo.kl.init_coef 0 --reward.clip_range -10 10
  --ckpt.output_dir "$SAVE_ROOT/hf" --ckpt.path "$SAVE_ROOT/state" --ckpt.save_steps 5 --ckpt.max_num 2
  --logger.logging_steps 1 --logger.tensorboard_dir "$SAVE_ROOT/tensorboard"
  --logger.wandb.project skill2env_rl --logger.wandb.run_name "skill2env_qwen3.5_9b_$SLURM_JOB_ID"
)
[ "${RESUME:-0}" = "1" ] && RL_ARGS+=(--ckpt.load_enable --ckpt.warm_resume_rollouts)
RL_ARGS+=("$@")
printf -v RL_ARGS_Q ' %q' "${RL_ARGS[@]}"

srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" "${CONTAINER_ARGS[@]}" bash -lc \
  "$RAY_ENV && ray job submit --address=http://localhost:$DASHBOARD_PORT -- bash -lc 'cd /molt && python3 -u -m molt.cli.train_rl_ray$RL_ARGS_Q'"
