# Skill2Env SFT

- Teacher: `zai-org/GLM-5.3` (NVFP4 quant: `Inferact/GLM-5.3-NVFP4`)
- Student: `Qwen/Qwen3.5-4B`
- Harness: Pi
- Context Length: 128k

Run all commands from the repository root.

The stock molt image (vLLM 0.27.1) serves GLM-5.3 natively: the `glm_moe_dsa`
architecture, `glm47` tool parser, and `glm45` reasoning parser all ship in it.
The NVFP4 checkpoint (433GB vs 704GB FP8) is the default — it fits TP8 on
8xB200 with a ~2.2M-token fp8 KV cache and decodes ~10x faster than the FP8
checkpoint under eager mode. Keep CUDA graphs on (`VLLM_ENFORCE_EAGER=0`,
the default): eager decode is slow enough that a single long turn can outlive
pi's request patience and truncate the session mid-task.

## Prepare all tasks

Materialize every `s2ev2_sft_1k` task with its Pi TaskSpec into `teacher_tasks.jsonl`.

```bash
export SAVE_ROOT=/raid/binfeng/Linex/outputs/skill2env_sft

PYTHONPATH=. .venv/bin/python examples/skill2env_sft/prepare.py \
  --dataset-dir /raid/binfeng/data/s2e/s2ev2_sft_1k \
  --image-dir /raid/binfeng/data/s2e/s2ev2_sif \
  --max-tasks -1 \
  --model-name openai/Inferact/GLM-5.3-NVFP4 \
  --context-window 131072 \
  --output "$SAVE_ROOT/teacher_tasks.jsonl"
```

## Roll out and export the teacher

Run 64 concurrent eval-only Polar sessions against a single TP8 GLM-5.3 engine
with expert parallelism and an fp8 KV cache. Each teacher request becomes an
independent row in `teacher_sft.jsonl`, with its full conversation history as
context. GLM-5.3 thinks adaptively — mechanical mid-trajectory turns may carry
no `reasoning_content`, unlike Qwen3.8 which thought every turn.

By default every well-formed teacher trace is exported regardless of task
outcome, so the student imitates full teacher behavior. Set
`REJECTION_SAMPLING=1` to keep only traces whose session reward was positive.

The flashinfer mount persists the first bringup's sm_100a JIT kernels
(~15 min) across containers.

```bash
export MOLT_IMAGE=hijkzzz/molt:latest
export SAVE_ROOT="${SAVE_ROOT:-/raid/binfeng/Linex/outputs/skill2env_sft}"

docker run --rm \
  --gpus all \
  --privileged \
  --ipc=host \
  --network=host \
  --ulimit nofile=65536:65536 \
  --entrypoint bash \
  -v "$PWD:/molt" \
  -v /raid:/raid \
  -v /usr/bin/apptainer:/usr/bin/apptainer:ro \
  -v /usr/libexec/apptainer:/usr/libexec/apptainer:ro \
  -v /etc/apptainer:/etc/apptainer:ro \
  -v /raid/binfeng/.cache/flashinfer:/root/.cache/flashinfer \
  -e SAVE_ROOT="$SAVE_ROOT" \
  -e HF_HOME=/raid/binfeng/.cache/huggingface \
  "$MOLT_IMAGE" -lc '
    mkdir -p /var/lib/apptainer/mnt/session
    cd /molt
    bash examples/skill2env_sft/rollout_teacher.sh
  '
```

## Train the student

Train Qwen3.5-4B for three epochs on the exported conversations
(batch size 16, lr 5e-6). Each epoch end saves a resumable checkpoint plus an
HF snapshot under `$SAVE_ROOT/state/_hf/`, and the final consolidated Hugging
Face checkpoint lands at `$FINAL_MODEL`.

```bash
export MOLT_IMAGE=hijkzzz/molt:latest
export SAVE_ROOT="${SAVE_ROOT:-/raid/binfeng/Linex/outputs/skill2env_sft}"

docker run --rm \
  --gpus all \
  --ipc=host \
  --network=host \
  --entrypoint bash \
  -v "$PWD:/molt" \
  -v /raid:/raid \
  -e SAVE_ROOT="$SAVE_ROOT" \
  -e HF_HOME=/raid/binfeng/.cache/huggingface \
  "$MOLT_IMAGE" -lc '
    cd /molt
    bash examples/skill2env_sft/train_student.sh
  '
```
