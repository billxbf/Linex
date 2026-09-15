# Skill2Env RL: asynchronous DPPO

Qwen3.8-27B on eight GPUs: four vLLM TP1 engines and four FSDP trainer GPUs with
CP4 and CPU optimizer state. The default recipe uses **GRPO advantages with eight
rollouts per prompt** and **binary-KL DPPO**:

- Eight prompt groups per update, eight rollouts each: **64 trajectories**.
- Sixteen prompt groups in flight: up to **128 sessions**.
- Async queue size **2**, partial rollout enabled, one optimizer step per batch.
- Binary KL threshold **0.05**, learning rate **1e-6**, gradient clip **1.0**.
- Global action-token mean, no entropy mask, no reference KL, no extra TIS weight.
- Existing context budgets: **96k rollout**, **32k maximum per completion**, and
  **64k training cap per prefix-merging segment**, including prompt and observations.

The batch and in-flight environment variables count **individual rollouts**. The
launch script divides them by `SAMPLES_PER_PROMPT` before passing the CLI's
prompt-group counts. Each prompt's samples finish as a group; queue capacity does
not bound the policy age of a slow session.

## Objective

At each action token, let `p = exp(rollout_log_prob)` and
`q = exp(current_log_prob)`. The behavior probability is stored at sampling time,
including when partial rollout crosses a weight update. DPPO computes

```text
r = q / p
D = p log(p/q) + (1-p) log((1-p)/(1-q))
blocked = D > threshold and ((A > 0 and r > 1) or (A < 0 and r < 1))
loss = -sum(unblocked * r * A) / total_action_tokens
```

`A` uses each trace's reward, with equal rollout weight in group statistics and
action-token weighting within each rollout. Equal trace rewards within a rollout
recover the shared advantage. The mask has no gradient. Blocked tokens remain in
the denominator. Binary KL uses only the two sampled probabilities, so it needs no full-vocabulary entropy,
top-K payload, reference model, or old-policy forward in this zero-KL recipe.
Probabilities at numerical zero/one are clamped for KL evaluation; the existing
log-ratio overflow guard also applies.

The relevant CLI options are `--actor.loss_mode dppo` and
`--actor.dppo_kl_threshold 0.05`. Keep `--algo.advantage.is_correction_level off`:
DPPO already includes the rollout importance ratio. Separate IS correction,
PPO dual clipping, and nucleus sampling (`top_p < 1`) are rejected. The algorithm
follows [DPPO equations 12 and 14](https://arxiv.org/abs/2602.04879).

`--train.force_on_policy` retains its existing execution meaning here: accumulate
all microbatches and update once. DPPO still uses the actual rollout anchor.
`--train.force_sync_mode` is not enabled; generation and training overlap.

## Prepare

Run preparation in an environment with this checkout and its Python dependencies.
The SIF images are reused; `--build-missing` builds missing images.

```bash
export SAVE_ROOT=/raid/binfeng/Linex/outputs/skill2env_rl_dppo

PYTHONPATH=. python3 examples/skill2env_rl_dppo/prepare.py \
  --dataset-dir /raid/binfeng/data/s2e/s2ev2_terminal_coding \
  --output "$SAVE_ROOT/train_tasks.jsonl" --skip-image-check
PYTHONPATH=. python3 examples/skill2env_rl_dppo/prepare.py \
  --dataset-dir /raid/binfeng/data/s2e/s2ev2_eval \
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
EVAL_DATASET="$SAVE_ROOT/eval_tasks.jsonl" EVAL_STEPS=25 \
  bash examples/skill2env_rl_dppo/train_rl.sh
```

Evaluation defaults to one rollout per prompt, independently of training's G=8,
and runs before the first update when `EVAL_DATASET` is supplied. `RESUME=1`
resumes the checkpoints under this recipe's `SAVE_ROOT`.

### Harbor rubric with NVIDIA GPT-6

Select `harbor_rubric` when preparing training records. This configures
`https://inference-api.nvidia.com/v1` with model `openai/openai/gpt-6-astra`.
The default additive coefficient is 0.2; set `--rubric-coefficient` at preparation
time to change it. Keep evaluation records on plain Harbor to measure task quality.

```bash
export SAVE_ROOT=/raid/binfeng/Linex/outputs/skill2env_rl_dppo_rubric
: "${NVIDIA_API_KEY:?Set NVIDIA_API_KEY before launching rubric training}"
export NVIDIA_API_KEY

PYTHONPATH=. python3 examples/skill2env_rl_dppo/prepare.py \
  --evaluator harbor_rubric --rubric-coefficient 0.2 \
  --output "$SAVE_ROOT/train_tasks.jsonl" --skip-image-check
PYTHONPATH=. python3 examples/skill2env_rl_dppo/prepare.py \
  --dataset-dir /raid/binfeng/data/s2e/s2ev2_eval \
  --output "$SAVE_ROOT/eval_tasks.jsonl" --skip-image-check

PROMPT_DATASET="$SAVE_ROOT/train_tasks.jsonl" \
EVAL_DATASET="$SAVE_ROOT/eval_tasks.jsonl" EVAL_STEPS=25 \
WANDB_RUN_NAME=harbor_rubric_gpt6_dppo \
  bash examples/skill2env_rl_dppo/train_rl.sh
```

The key is read from the Polar gateway's environment and forwarded through the
Molt Ray job environment, including when attaching to an existing Ray cluster.
Prepared task JSON contains only the environment-variable name. For a Docker
launch, also pass `-e NVIDIA_API_KEY` to the training container.

For a two-update checkpoint smoke, keep the full training dataset and use
`SAVE_STEPS=2 bash examples/skill2env_rl_dppo/train_rl.sh --train.max_steps 2 --ckpt.disable_final_save`.
Then resume with `RESUME=1`, `SAVE_STEPS=25`, and
without `--train.max_steps`. Keep `WANDB_RUN_ID` unchanged and set
`WANDB_RESUME=allow` for both launches. The first pass defaults to one episode;
the actual update count depends on the complete usable groups.

For fixed-weight comparisons, run `--eval.eval_only` separately before training
and against the final HF checkpoint, with identical evaluation settings. HF
snapshots live under `state/_hf/global_stepN`; resumable DCP checkpoints live
under `state/_actor/global_stepN`. Training also saves the last update when it
falls between regular save intervals. `rollout/reward_ma8` starts at update 12,
and its history survives DCP resume. Logs retain every rollout reward and each
group's mean reward for distribution analysis.

For a new container on the existing Apptainer host:

```bash
export MOLT_IMAGE=hijkzzz/molt:latest
export SAVE_ROOT="${SAVE_ROOT:-/raid/binfeng/Linex/outputs/skill2env_rl_dppo}"

docker run -d --name s2e_rl_dppo \
  --gpus all --privileged --ipc=host \
  --ulimit nofile=65536:65536 --entrypoint bash \
  -v "$PWD:/molt" -v /raid:/raid \
  -v /usr/bin/apptainer:/usr/bin/apptainer:ro \
  -v /usr/libexec/apptainer:/usr/libexec/apptainer:ro \
  -v /etc/apptainer:/etc/apptainer:ro \
  -v /raid/binfeng/.cache/flashinfer:/root/.cache/flashinfer \
  -e SAVE_ROOT="$SAVE_ROOT" \
  -e HF_HOME=/raid/binfeng/.cache/huggingface \
  -e MOLT_DEFER_GRAD_SYNC=0 \
  -e POLAR_APPTAINER_ROOTLESS=1 \
  -e WANDB_KEY="$WANDB_KEY" \
  "$MOLT_IMAGE" -lc '
    mkdir -p /var/lib/apptainer/mnt/session
    sed -e "s/^mount dev = yes/mount dev = minimal/" -e "s/^systemd cgroups = yes/systemd cgroups = no/" \
      /etc/apptainer/apptainer.conf > /run/apptainer.conf
    mount --bind /run/apptainer.conf /etc/apptainer/apptainer.conf
    mkdir -p /sys/fs/cgroup/init
    for p in $(cat /sys/fs/cgroup/cgroup.procs); do echo $p > /sys/fs/cgroup/init/cgroup.procs; done
    echo "+memory +pids +cpu" > /sys/fs/cgroup/cgroup.subtree_control
    mkdir -p /sys/fs/cgroup/polar
    for i in $(seq 0 255); do [ -e /dev/loop$i ] || mknod /dev/loop$i b 7 $i; done
    apt-get update -qq >/dev/null && apt-get install -y -qq uidmap liblzo2-2 fuse3 libfuse3-3 >/dev/null
    grep -q "^ubuntu:" /etc/subuid || echo "ubuntu:100000:65536" >> /etc/subuid
    grep -q "^ubuntu:" /etc/subgid || echo "ubuntu:100000:65536" >> /etc/subgid
    mkdir -p /home/ubuntu && chown 1000:1000 /home/ubuntu
    umount /proc/driver/nvidia/params 2>/dev/null || true
    cd /molt
    bash examples/skill2env_rl_dppo/train_rl.sh 2>&1 | tee "$SAVE_ROOT/train.log"
  '
```

The launcher defaults `MOLT_DEFER_GRAD_SYNC=0`, optimizer-only offload, full
activation checkpointing, and one sample per microbatch. These keep the 27B
training memory bounded without streaming model parameters over PCIe on every
forward. Token-budget dynamic batching remains available with `DYNAMIC_BATCH=1`;
padding can increase its actual memory use. The runtime uses per-task cgroup
memory limits and rootless Apptainer when `POLAR_APPTAINER_ROOTLESS=1`.

The training cap is a memory tradeoff: actions beyond 64k in a segment do not
receive gradients even though its terminal reward is retained. `TRAIN_MAX_LEN`
can be raised when trainer memory permits. The rollout budget is independent.

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

Track `rollout/harbor_mean` (raw verifier reward), `rollout/reward_mean` (training
reward), and `rollout/judge_mean` (normalized judge contribution in `[-1, 1]`,
zero when absent), alongside fixed-dataset evaluation scores. These means use
action-token weights within each rollout, then equal rollout weights. WandB
prefixes them with `train/`. The plain Harbor recipe has zero judge contribution.
`policy_clip_ratio` is the fraction of action tokens blocked by DPPO; `policy_kl` is the sampled mean
`log(mu) - log(pi)`, not the binary KL threshold statistic. `rollout_abs_logratio`
tracks absolute behavior mismatch. `timing/policy_train`, `timing/actor_idle_wait`,
`timing/vllm_idle_wait`, and `timing/step_total` distinguish training cost from
rollout waiting; `perf/gpu_mem_peak_gb` tracks trainer memory.

Timeouts with usable completions train at `--rollout.timeout_reward` (default 0).
`polar/timeout` counts their trained rows; `rollout/dropped/*` records unusable
sessions and infrastructure errors. `response_clip_ratio` records training-cap
hits, separately from the completion/context truncation metric `truncated`.

Weights are pushed after every step. Partial rollout pauses the engines, refits
weights, invalidates prefix caches, and resumes requests under the new weights.
`--rollout.refit_drain` remains available to drain completions first, but is off
in this recipe because long completions delay the refit.
