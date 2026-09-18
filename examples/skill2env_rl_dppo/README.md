# Skill2Env RL: asynchronous DPPO

Running Qwen3.8-27B on single B200 node: four vLLM TP1 engines and four FSDP trainer GPUs with
CP4 and CPU optimizer state. The default recipe uses **GRPO advantages with eight
rollouts per prompt** and **binary-KL DPPO**:

- Eight prompt groups per update, eight rollouts each: **64 trajectories**.
- Sixteen prompt groups in flight: up to **128 sessions**.
- Async queue size **2**, partial rollout enabled, one optimizer step per batch.
- Binary KL threshold **0.05**, learning rate **1e-6**, gradient clip **1.0**.
- Global action-token mean, no entropy mask, no reference KL, no extra TIS weight.
- Existing context budgets: **96k rollout**, **32k maximum per completion**, and
  **64k training cap per prefix-merging segment**, including prompt and observations.



## Prepare Image and Dataset

Run preparation in an environment with this checkout and its Python dependencies.
Replace `/path/to/...` with your local paths. Existing SIF images are reused;
`--build-missing` builds missing images using Docker and Apptainer, and
`--build-missing --force` rebuilds selected images.

```bash
export SAVE_ROOT="$PWD/outputs/skill2env_rl_dppo"

PYTHONPATH=. python3 examples/skill2env_rl_dppo/prepare.py \
  --dataset-dir /path/to/training/tasks \
  --image-dir /path/to/images \
  --output "$SAVE_ROOT/train_tasks.jsonl" --skip-image-check
PYTHONPATH=. python3 examples/skill2env_rl_dppo/prepare.py \
  --dataset-dir /path/to/evaluation/tasks \
  --image-dir /path/to/images \
  --output "$SAVE_ROOT/eval_tasks.jsonl" --skip-image-check
```

Prepared records use pi with thinking level `high`, prefix merging, and the Harbor
verifier's reward in `[0, 1]`. The default pi context window is 65536, equal to
`CONTEXT_LENGTH - MAX_NEW_TOKENS`; change `--context-window` at preparation time
if those budgets change. Existing records with these settings can be supplied
through `PROMPT_DATASET`.

## Train

In the configured training container:

```bash
export MODEL_PATH=/path/to/policy/model
EVAL_DATASET="$SAVE_ROOT/eval_tasks.jsonl" EVAL_STEPS=25 \
  bash examples/skill2env_rl_dppo/train_rl.sh
```

Evaluation defaults to one rollout per prompt, independently of training's G=8,
and runs before the first update when `EVAL_DATASET` is supplied. `RESUME=1`
resumes the checkpoints under this recipe's `SAVE_ROOT`.

### Optional Harbor rubric

Select `harbor_rubric` and supply your judge endpoint root and model name.
The endpoint must support `/chat/completions`. The default additive coefficient
is 0.2; keep evaluation records on plain Harbor to measure task quality.

```bash
PYTHONPATH=. python3 examples/skill2env_rl_dppo/prepare.py \
  --dataset-dir /path/to/training/tasks \
  --image-dir /path/to/images \
  --evaluator harbor_rubric --rubric-coefficient 0.2 \
  --judge-base-url "$JUDGE_BASE_URL" --judge-model "$JUDGE_MODEL" \
  --output "$SAVE_ROOT/train_tasks.jsonl" --skip-image-check
```

Set `JUDGE_BASE_URL` and `JUDGE_MODEL` for your deployment before preparation.
Train with the same launcher and the resulting records.

The training environment needs the checkout, model, datasets, and SIF images
at accessible paths, with Apptainer installed. When using a training container,
mount these paths consistently with the paths stored in the prepared records.


## Settings and monitoring

| Environment variable | Default | Meaning |
|---|---:|---|
| `SAMPLES_PER_PROMPT` | 8 | Rollouts in each reward group |
| `ROLLOUT_BATCH_SIZE` | 64 | Individual trajectories per optimizer step |
| `INFLIGHT_ROLLOUTS` | 128 | Individual sessions in the dispatch pool |
| `ASYNC_QUEUE_SIZE` | 2 | Buffered batch capacity |
| `DPPO_KL_THRESHOLD` | 0.05 | Binary KL mask threshold |
| `LR` | 1e-6 | Constant actor learning rate |
| `CONTEXT_LENGTH` | 98304 | Rollout context budget |
| `TRAIN_MAX_LEN` | 65536 | Training sequence cap per segment |
| `MAX_NEW_TOKENS` | 32768 | Maximum tokens per completion |
| `SESSION_TIMEOUT` | 3600 | Agent execution budget in seconds |
| `PARTIAL_ROLLOUT` | 1 | Pause/refit/resume during generation |
| `EVAL_SAMPLES_PER_PROMPT` | 1 | Independent evaluation sampling count |