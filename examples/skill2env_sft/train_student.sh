#!/usr/bin/env bash
# Train Qwen3.5-4B on exported teacher conversations in a separate GPU lifetime.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/skill2env_sft}"
SFT_DATASET="${SFT_DATASET:-$SAVE_ROOT/teacher_sft.jsonl}"
MODEL_PATH="${MODEL_PATH:-/raid/binfeng/models/Qwen3.5-4B}"
FINAL_MODEL="${FINAL_MODEL:-/raid/binfeng/models/Qwen3.5-4B-skill2env_sft}"

test -d "$MODEL_PATH" || { echo "Student model not found: $MODEL_PATH"; exit 1; }
test -s "$SFT_DATASET" || { echo "Exported SFT dataset not found: $SFT_DATASET"; exit 1; }
mkdir -p "$SAVE_ROOT"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node="${GPUS_PER_NODE:-8}" -m molt.cli.train_sft \
  --data.dataset "$SFT_DATASET" \
  --data.input_key prompt_messages \
  --data.output_key response_messages \
  --data.image_key images \
  --data.max_len "${CONTEXT_LENGTH:-131072}" \
  --data.max_samples "${MAX_SAMPLES:-1000000}" \
  --model.model_name_or_path "$MODEL_PATH" \
  --model.freeze_visual_encoder \
  --model.gradient_checkpoint "${GRADIENT_CHECKPOINTING:-full}" \
  --train.max_epochs "${MAX_EPOCHS:-3}" \
  --train.batch_size "${TRAIN_BATCH_SIZE:-1}" \
  --train.micro_batch_size "${MICRO_BATCH_SIZE:-1}" \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation "${FSDP_ATTN_IMPLEMENTATION:-sdpa}" \
  --fsdp.tp_size "${TP_SIZE:-1}" \
  --fsdp.ep_size "${EP_SIZE:-1}" \
  --fsdp.cp_size "${CP_SIZE:-8}" \
  --adam.lr "${LR:-1e-6}" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.output_dir "$FINAL_MODEL" \
  --ckpt.save_steps "${SAVE_STEPS:-0}" \
  --ckpt.save_hf \
  --ckpt.max_num "${MAX_CHECKPOINTS:-3}" \
  --logger.logging_steps "${LOGGING_STEPS:-1}" \
  --logger.tensorboard_dir "$SAVE_ROOT/tensorboard" \
  --logger.wandb.project "${WANDB_PROJECT:-skill2env_sft}" \
  --logger.wandb.run_name "${WANDB_RUN_NAME:-skill2env_sft_$$}" \
  "$@"
