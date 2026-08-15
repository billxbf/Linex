#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────
# Async GRPO training on SWE-Gym via Polar + Slime (Qwen3.5-4B).
#
# Qwen3.5-4B is a VLM checkpoint (Qwen3_5ForConditionalGeneration) with
# hybrid attention (1 full + 3 GatedDeltaNet linear per 4 layers). Text-only
# RL requires the SGLang VLM input_ids patch (see MEMORY.md).
#
#
# Port layout:
#   9000        – SGLang router (slime-managed, load-balances engines)
#   8080        – Polar rollout server (task coordinator)
#   8100        – Polar gateway node (dispatches agent sessions)
#   8265        – Ray dashboard
#
# Weight sync: native GPU-to-GPU via NCCL every training step.
# Slime manages SGLang engines; Polar gateway proxies LLM calls to them.
# Dynamic-history: every trace in each agent session becomes one training
# sample, so gradients learn from *every* turn (not just the last one).
# ──────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." && pwd)"
RUN_DIR="${RUN_DIR:-${PROJECT_ROOT}/tmp/swegym_slime_grpo}"
mkdir -p "${RUN_DIR}" "${PROJECT_ROOT}/logs"
PYTHON_BIN="${PYTHON_BIN:-${PROJECT_ROOT}/.venv/bin/python3}"
if [ ! -x "${PYTHON_BIN}" ]; then
    PYTHON_BIN="$(command -v python3 || command -v python)"
fi
PYTHON_BIN_DIR="$(cd -- "$(dirname -- "${PYTHON_BIN}")" &>/dev/null && pwd)"
export PATH="${PYTHON_BIN_DIR}:${PATH}"

is_path_like() {
    case "$1" in
        /*|./*|../*|~*) return 0 ;;
        *) return 1 ;;
    esac
}

detect_host_ip() {
    "${PYTHON_BIN}" - <<'PY'
import socket

try:
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.connect(("8.8.8.8", 80))
    print(sock.getsockname()[0])
    sock.close()
except Exception:
    try:
        print(socket.gethostbyname(socket.gethostname()))
    except Exception:
        print("127.0.0.1")
PY
}

# ── External deps ──────────────────────────────────────────────────
SLIME_DIR="${SLIME_DIR:-${PROJECT_ROOT}/slime}"
if [ ! -f "${SLIME_DIR}/train_async.py" ]; then
    echo "ERROR: Slime not found at ${SLIME_DIR}"
    echo "  git clone git@github.com:THUDM/slime.git ${SLIME_DIR}"
    exit 1
fi

MEGATRON_DIR="${MEGATRON_DIR:-${PROJECT_ROOT}/Megatron-LM}"
if [ ! -d "${MEGATRON_DIR}/megatron" ]; then
    echo "ERROR: Megatron-LM not found at ${MEGATRON_DIR}"
    echo "  git clone https://github.com/NVIDIA/Megatron-LM.git ${MEGATRON_DIR}"
    exit 1
fi

# ── Model ──────────────────────────────────────────────────────────
# Qwen3.5-4B: VLM checkpoint; we train text-only.  HF weights are loaded
# through slime_plugins.mbridge.qwen3_5 (text_config-aware) at convert-time.
HF_CHECKPOINT="${HF_CHECKPOINT:-Qwen/Qwen3.5-4B}"
REF_LOAD="${REF_LOAD:-${PROJECT_ROOT}/tmp/checkpoints/Qwen3.5-4B_torch_dist}"
RUN_ID="${RUN_ID:-swegym-slime-grpo-$(date -u +%Y%m%dT%H%M%SZ)}"
SAVE_ROOT="${SAVE_ROOT:-${PROJECT_ROOT}/tmp/ckpt/swegym_slime_grpo_qwen35_4b}"
SAVE_DIR="${SAVE_DIR:-${SAVE_ROOT}/${RUN_ID}}"
mkdir -p "$SAVE_DIR"
if is_path_like "$HF_CHECKPOINT" && [ ! -e "$HF_CHECKPOINT" ]; then
    echo "ERROR: HF checkpoint not found at $HF_CHECKPOINT"
    echo "  hf download Qwen/Qwen3.5-4B"
    exit 1
fi
if [ ! -d "$REF_LOAD" ] || [ ! -f "$REF_LOAD/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: Megatron torch_dist checkpoint not found at $REF_LOAD"
    echo "  Run bash examples/swegym_slime_grpo/convert_weights.sh first."
    exit 1
fi

# shellcheck source=./model_args.sh
source "${SCRIPT_DIR}/model_args.sh"

# First run has an empty SAVE_DIR — slime's load_checkpoint asserts on empty.
# Pick REF_LOAD (torch_dist) until the first save lands.
if [ -f "$SAVE_DIR/latest_checkpointed_iteration.txt" ]; then
    LOAD_DIR="$SAVE_DIR"
else
    LOAD_DIR="$REF_LOAD"
fi

# ── Data ───────────────────────────────────────────────────────────
PROMPT_DATA="${PROMPT_DATA:-${SCRIPT_DIR}/swegym_train_293.jsonl}"
if [ ! -f "$PROMPT_DATA" ]; then
    echo "Preparing train data..."
    "${PYTHON_BIN}" "${SCRIPT_DIR}/prepare_data.py"
fi

# ── Runtime configs ─────────────────────────────────────────────────
export AGENT_CLI_DIR="${AGENT_CLI_DIR:-${PROJECT_ROOT}/tmp/swegym_agent_cli/opt_node}"
export APPTAINER_IMAGE_DIR="${APPTAINER_IMAGE_DIR:-${PROJECT_ROOT}/tmp/swegym_apptainer_images}"
# Prefer apptainer in PATH (HPC modules etc.); fall back to /usr/bin for Ubuntu defaults.
export POLAR_APPTAINER_BIN="${POLAR_APPTAINER_BIN:-$(command -v apptainer || echo /usr/bin/apptainer)}"
SGLANG_ROUTER_PORT="${SGLANG_ROUTER_PORT:-9000}"
SGLANG_ROUTER_HOST="${SGLANG_ROUTER_HOST:-$(detect_host_ip)}"
export SGLANG_ROUTER_BASE_URL="${SGLANG_ROUTER_BASE_URL:-http://${SGLANG_ROUTER_HOST}:${SGLANG_ROUTER_PORT}}"
TOPOLOGY_TEMPLATE="${TOPOLOGY_TEMPLATE:-${SCRIPT_DIR}/topology.yaml}"
POLAR_CONFIG_TEMPLATE="${POLAR_CONFIG_TEMPLATE:-${SCRIPT_DIR}/polar_config.yaml}"
TOPOLOGY_PATH="${TOPOLOGY_PATH:-${RUN_DIR}/topology.yaml}"
CUSTOM_CONFIG_PATH="${CUSTOM_CONFIG_PATH:-${RUN_DIR}/polar_config.yaml}"
COMPILER_CACHE_ROOT="${COMPILER_CACHE_ROOT:-${RUN_DIR}/compiler_cache}"
TORCHINDUCTOR_CACHE_DIR="${TORCHINDUCTOR_CACHE_DIR:-${COMPILER_CACHE_ROOT}/torchinductor}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-${COMPILER_CACHE_ROOT}/triton}"
mkdir -p "$TORCHINDUCTOR_CACHE_DIR" "$TRITON_CACHE_DIR"

# Render YAML templates: only the listed ${VARS} are expanded, so literal
# $HOME / $... inside polar_config.yaml are left untouched.
command -v envsubst >/dev/null || { echo "ERROR: envsubst not found (install gettext-base)"; exit 1; }
TEMPLATE_VARS='${SGLANG_ROUTER_BASE_URL} ${AGENT_CLI_DIR} ${APPTAINER_IMAGE_DIR}'
mkdir -p "$(dirname "$TOPOLOGY_PATH")" "$(dirname "$CUSTOM_CONFIG_PATH")"
envsubst "$TEMPLATE_VARS" < "$TOPOLOGY_TEMPLATE"     > "$TOPOLOGY_PATH"
envsubst "$TEMPLATE_VARS" < "$POLAR_CONFIG_TEMPLATE" > "$CUSTOM_CONFIG_PATH"

echo "Using topology: ${TOPOLOGY_PATH}"
echo "Using Polar config: ${CUSTOM_CONFIG_PATH}"
echo "Using Apptainer image dir: ${APPTAINER_IMAGE_DIR}"
echo "Using run id: ${RUN_ID}"
echo "Using save dir: ${SAVE_DIR}"
echo "Using SGLang router URL for Polar gateway: ${SGLANG_ROUTER_BASE_URL}"

# ── Cleanup on exit ────────────────────────────────────────────────
PIDS=()
cleanup() {
    echo "Shutting down..."
    for pid in "${PIDS[@]}"; do kill "$pid" 2>/dev/null || true; done
    ray stop --force 2>/dev/null || true
    wait 2>/dev/null || true
}
trap cleanup EXIT

# ── Step 1: Polar services (runs on host, CPU only) ───────────────
echo "=== Starting Polar rollout server (:8080) ==="
polar serve_rollout -c "${TOPOLOGY_PATH}" &
PIDS+=($!)
sleep 2

echo "=== Starting Polar gateway (:8100) ==="
polar serve_gateway -c "${TOPOLOGY_PATH}" --node-id localhost-node-01 &
PIDS+=($!)
sleep 2

curl -sf http://127.0.0.1:8080/health || { echo "Polar rollout server not healthy"; exit 1; }

# ── Step 2: Ray + Slime (manages SGLang engines + training) ───────
ACTOR_NUM_GPUS_PER_NODE="${ACTOR_NUM_GPUS_PER_NODE:-4}"
ROLLOUT_NUM_GPUS="${ROLLOUT_NUM_GPUS:-4}"
ROLLOUT_NUM_GPUS_PER_ENGINE="${ROLLOUT_NUM_GPUS_PER_ENGINE:-1}"
ROLLOUT_BATCH_SIZE="${ROLLOUT_BATCH_SIZE:-4}"
N_SAMPLES_PER_PROMPT="${N_SAMPLES_PER_PROMPT:-16}"
MAX_TOKENS_PER_GPU="${MAX_TOKENS_PER_GPU:-30000}"
SGLANG_CONTEXT_LENGTH="${SGLANG_CONTEXT_LENGTH:-50000}"

# Ray sizing — derive total GPUs from the actor/rollout split.
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$((ACTOR_NUM_GPUS_PER_NODE + ROLLOUT_NUM_GPUS))}"
RAY_HEAD_IP="${RAY_HEAD_IP:-127.0.0.1}"

echo "=== Starting Ray on ${RAY_HEAD_IP} (${RAY_NUM_GPUS} GPUs) ==="
ray stop --force 2>/dev/null || true
sleep 1
ray start --head --node-ip-address "$RAY_HEAD_IP" --num-gpus "$RAY_NUM_GPUS" --disable-usage-stats

# cuDNN lib path — probe the active venv instead of hardcoding python3.13.
if [ -z "${CUDNN_LIB:-}" ]; then
    CUDNN_LIB="$("${PYTHON_BIN}" -c 'import nvidia.cudnn, os; print(os.path.join(list(nvidia.cudnn.__path__)[0], "lib"))' 2>/dev/null || true)"
fi
RUNTIME_LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
if [ -n "${CUDNN_LIB}" ] && [ -d "$CUDNN_LIB" ]; then
    RUNTIME_LD_LIBRARY_PATH="${CUDNN_LIB}:${RUNTIME_LD_LIBRARY_PATH}"
fi
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"${MEGATRON_DIR}:${PROJECT_ROOT}/src\",
    \"PATH\": \"${PYTHON_BIN_DIR}:${PATH}\",
    \"VIRTUAL_ENV\": \"${VIRTUAL_ENV:-${PROJECT_ROOT}/.venv}\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"WANDB_DIR\": \"${PROJECT_ROOT}/logs\",
    \"TORCHINDUCTOR_CACHE_DIR\": \"${TORCHINDUCTOR_CACHE_DIR}\",
    \"TRITON_CACHE_DIR\": \"${TRITON_CACHE_DIR}\",
    \"LD_LIBRARY_PATH\": \"${RUNTIME_LD_LIBRARY_PATH}\",
    \"PYTORCH_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"max_split_size_mb:2048,expandable_segments:True\",
    \"NVTE_DEBUG\": \"1\",
    \"NVTE_DEBUG_LEVEL\": \"2\"
  }
}"

# Rollout sizing: 4 prompts × 16 trajectories = 64 trajectories/rollout.
# This matches the earlier high-util baseline and keeps request groups smaller
# so long tails do not collapse usable token throughput.
# With --dynamic-history each trajectory explodes into one sample per trace,
# so sample count per rollout is variable.
# The custom data source rounds epoch length up to 37 rollout batches, so all
# 293 train prompts are consumed once; the final fixed-size batch wraps 3 prompts.
echo "=== Launching train_async.py ==="
ray job submit --address="http://${RAY_HEAD_IP}:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- "${PYTHON_BIN}" "${SLIME_DIR}/train_async.py" \
    --actor-num-nodes 1 \
    --actor-num-gpus-per-node "$ACTOR_NUM_GPUS_PER_NODE" \
    --rollout-num-gpus "$ROLLOUT_NUM_GPUS" \
    --rollout-num-gpus-per-engine "$ROLLOUT_NUM_GPUS_PER_ENGINE" \
    "${MODEL_ARGS[@]}" \
    --hf-checkpoint "$HF_CHECKPOINT" \
    --ref-load "$REF_LOAD" \
    --load "$LOAD_DIR" \
    --save "$SAVE_DIR" \
    --save-interval "${SAVE_INTERVAL:-10}" \
    --update-weights-interval 1 \
    --rollout-function-path slime_bridge.rollout.generate_rollout_polar_async \
    --custom-rm-path slime_bridge.reward.reward_func \
    --custom-reward-post-process-path slime_bridge.reward_post_process.post_process_rewards \
    --custom-config-path "${CUSTOM_CONFIG_PATH}" \
    --data-source-path slime_bridge.data_source.CeilEpochRolloutDataSourceWithBuffer \
    --prompt-data "$PROMPT_DATA" \
    --input-key prompt \
    --label-key label \
    --metadata-key metadata \
    --rollout-shuffle \
    --reward-key score \
    --num-epoch "${NUM_EPOCH:-1}" \
    --rollout-batch-size "$ROLLOUT_BATCH_SIZE" \
    --n-samples-per-prompt "$N_SAMPLES_PER_PROMPT" \
    --rollout-max-response-len 16000 \
    --rollout-max-prompt-len 32000 \
    --dynamic-history \
    --num-steps-per-rollout 1 \
    --tensor-model-parallel-size 2 \
    --sequence-parallel \
    --pipeline-model-parallel-size 1 \
    --context-parallel-size 1 \
    --expert-model-parallel-size 1 \
    --expert-tensor-parallel-size 1 \
    --recompute-granularity full \
    --recompute-method uniform \
    --recompute-num-layers 1 \
    --use-dynamic-batch-size \
    --max-tokens-per-gpu "$MAX_TOKENS_PER_GPU" \
    --log-probs-chunk-size 256 \
    --distributed-timeout-minutes 30 \
    --advantage-estimator grpo \
    --normalize-advantages \
    --use-tis \
    --use-kl-loss \
    --kl-loss-coef 0.001 \
    --kl-loss-type low_var_kl \
    --entropy-coef 0.0 \
    --eps-clip 0.2 \
    --eps-clip-high 0.28 \
    --optimizer adam \
    --lr 1e-6 \
    --lr-decay-style constant \
    --weight-decay 0.1 \
    --adam-beta1 0.9 \
    --adam-beta2 0.98 \
    --attention-dropout 0.0 \
    --hidden-dropout 0.0 \
    --accumulate-allreduce-grads-in-fp32 \
    --attention-softmax-in-fp32 \
    --attention-backend auto \
    --no-gradient-accumulation-fusion \
    --sglang-mem-fraction-static 0.8 \
    --sglang-context-length "$SGLANG_CONTEXT_LENGTH" \
    --sglang-tool-call-parser qwen3_coder \
    --router-policy "${SGLANG_ROUTER_POLICY:-round_robin}" \
    --use-wandb \
    --wandb-project "${WANDB_PROJECT:-polar-swegym-grpo}" \
    --wandb-group "${WANDB_GROUP:-swegym-qwen35-4b-async-grpo}" \
    --sglang-router-port "$SGLANG_ROUTER_PORT"
