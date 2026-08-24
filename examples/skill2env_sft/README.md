# Skill2Env SFT

- Teacher: `Qwen/Qwen3.8-27B`
- Student: `Qwen/Qwen3.5-4B`
- Harness: Pi
- Context Length: 128k

Run all commands from the repository root.

## Prepare all tasks

Materialize every `s2ev2_sft_1k` task with its Pi TaskSpec into `teacher_tasks.jsonl`.

```bash
export SAVE_ROOT=/raid/binfeng/Linex/outputs/skill2env_sft

PYTHONPATH=. .venv/bin/python examples/skill2env_sft/prepare.py \
  --dataset-dir /raid/binfeng/data/s2e/s2ev2_sft_1k \
  --image-dir /raid/binfeng/data/s2e/s2ev2_sif \
  --max-tasks -1 \
  --model-name openai/Qwen/Qwen3.8-27B \
  --context-window 131072 \
  --output "$SAVE_ROOT/teacher_tasks.jsonl"
```

## Roll out and export the teacher

Run 32 concurrent eval-only Polar sessions over eight TP1 data-parallel Qwen3.8-27B replicas. Each teacher request becomes an independent row in `teacher_sft.jsonl`, with its full conversation history as context.

By default every well-formed teacher trace is exported regardless of task outcome, so the student imitates full teacher behavior. Set `REJECTION_SAMPLING=1` to keep only traces whose session reward was positive.

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
  -e SAVE_ROOT="$SAVE_ROOT" \
  -e HF_HOME=/raid/binfeng/.cache/huggingface \
  "$MOLT_IMAGE" -lc '
    mkdir -p /var/lib/apptainer/mnt/session
    cd /molt
    bash examples/skill2env_sft/rollout_teacher.sh
  '
```

## Train the student

Train Qwen3.5-4B on the exported conversations and write the consolidated Hugging Face checkpoint.

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
