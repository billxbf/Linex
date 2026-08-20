#!/usr/bin/env bash
# Qwen3.5-4B text-only SFT on the annotated Skill2Env trajectories.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${MOLT_PATH:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3.5-4B}"
SFT_DATASET="${SFT_DATASET:-/raid/binfeng/data/s2e/skill2env-sft-1k-k3max.parquet}"
SAVE_ROOT="${SAVE_ROOT:-$REPO_ROOT/outputs/skill2env-sft-qwen3.5-4b-128k-1epoch}"

test -f "$SFT_DATASET"
mkdir -p "$SAVE_ROOT"

export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
export TOKENIZERS_PARALLELISM=true
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

cd "$REPO_ROOT"
torchrun --standalone --nproc_per_node=8 -m molt.cli.train_sft \
  --data.dataset "$SFT_DATASET" \
  --data.input_key messages \
  --data.max_len 131072 \
  --data.max_samples 1711 \
  --model.model_name_or_path "$MODEL_PATH" \
  --model.freeze_visual_encoder \
  --model.gradient_checkpoint full \
  --train.max_epochs 1 \
  --train.batch_size 2 \
  --train.micro_batch_size 1 \
  --fsdp.param_dtype bf16 \
  --fsdp.attn_implementation sdpa \
  --fsdp.tp_size 1 \
  --fsdp.ep_size 1 \
  --fsdp.cp_size 8 \
  --adam.lr "${LR:-1e-6}" \
  --ckpt.path "$SAVE_ROOT/state" \
  --ckpt.output_dir "$SAVE_ROOT/hf" \
  --ckpt.save_steps "${SAVE_STEPS:-121}" \
  --ckpt.max_num 1 \
  --logger.logging_steps 1 \
  --logger.tensorboard_dir "$SAVE_ROOT/tensorboard" \
  --logger.wandb.run_name skill2env-sft-qwen3.5-4b-128k-1epoch
