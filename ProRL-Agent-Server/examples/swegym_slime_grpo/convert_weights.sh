#!/usr/bin/env bash
# Convert Qwen3.5-4B HF weights to Megatron torch_dist format for Slime training.
# Qwen3.5-4B is a VLM checkpoint (Qwen3_5ForConditionalGeneration) with hybrid
# attention (1 full + 3 GatedDeltaNet linear per 4 layers).  Weight loading goes
# through slime_plugins.mbridge.qwen3_5 (text_config-aware).
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"

SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

if [ ! -f "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}. Clone it first:"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi

HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
OUTPUT_DIR="${TORCH_DIST_DIR:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
mkdir -p "$OUTPUT_DIR"

# shellcheck source=./model_args.sh
source "${SCRIPT_DIR}/model_args.sh"

echo "Converting ${HF_CHECKPOINT} -> ${OUTPUT_DIR}"

CUDA_DEVICE_MAX_CONNECTIONS=1 \
PYTHONPATH="${MEGATRON_DIR}:${SLIME_DIR}:${PROJECT_ROOT}/src" \
torchrun --nproc_per_node 1 \
    "${SLIME_DIR}/tools/convert_hf_to_torch_dist.py" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --save "$OUTPUT_DIR" \
    --tensor-model-parallel-size 1 \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --no-gradient-accumulation-fusion

echo "Done: ${OUTPUT_DIR}"
