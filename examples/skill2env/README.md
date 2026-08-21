# Skill2Env

This example is the end-to-end Skill2Env path from RFC 05. Molt owns Ray,
vLLM, optimization, checkpoints, and the only training CLI. Polar owns Hermes,
Apptainer sessions, `prefix_merging` trajectories, and the final-state Harbor
reward.

## Prerequisites

- `Qwen/Qwen3.5-9B` available from Hugging Face or a local `MODEL_PATH`.
- `/raid/binfeng/data/s2e/s2ev2_full_8k`, with each Harbor task's
  `instruction.md`, `task.toml`, `environment/Dockerfile`, and `tests/test.sh`.
- Docker and Apptainer to build missing SIFs; only Apptainer is needed once the
  SIFs exist. Task containers use host networking and internet access so Hermes
  can install and reach the Polar gateway.
- If Molt itself runs in a container, expose the host's Apptainer binary,
  `libexec`, configuration, and state directories and grant its required mount
  privileges.
- Eight GPUs for the local acceptance run. Multi-node runs also require the
  dataset, SIF directory, model/cache, repository, and output directory at the
  same absolute paths on every node.
- A Molt runtime environment containing the dependencies from this repository.
- A per-process open-file limit of at least 65,536 for Ray and concurrent task
  runtimes. Container launches may need `--ulimit nofile=65536:65536`.

## Prepare

The acceptance set is the first 40 tasks in stable path order. Preparation is
resumable: existing non-empty SIFs are reused, new SIFs are atomically renamed
after conversion, and the JSONL is replaced only after every selected task has
validated. The default preflight starts every selected image and checks its
`/workspace`; the Harbor verifier stays outside the image and its host path is
stored in the corresponding Polar task.

```bash
python3 examples/skill2env/prepare.py \
  --dataset-dir /raid/binfeng/data/s2e/s2ev2_full_8k \
  --image-dir /raid/binfeng/data/s2e/s2ev2_sif \
  --max-tasks 40 --build-missing
```

Each record contains the instruction and complete per-task Polar shape:
Apptainer image, requested resources, Hermes with a 131,072-token context, a
four-turn acceptance cap, and only its terminal/file toolsets,
`prefix_merging`, plain `harbor`, verifier timeout/path, and source metadata.
No rubric or judge model is configured.

## Local acceptance run

```bash
MODEL_PATH=Qwen/Qwen3.5-9B \
  bash examples/skill2env/train_rl.sh
```

Forty prompts at four prompts per rollout round produce exactly ten optimizer
steps. Every round requests sixteen trajectories per prompt and trains its
complete variable-size Experience set once. The asynchronous queue holds at
most two rounds; geometric importance sampling corrects their policy lag, and
each optimizer step refreshes vLLM weights. GPUs are split four for FSDP
training and four for vLLM rollout. Set `SAVE_ROOT` to choose the shared output
root and `RESUME=1` to resume its latest checkpoint.

Normal artifacts land under `SAVE_ROOT`: Polar results and `topology.json` in
`rollouts/`, TensorBoard metrics in `tensorboard/`, resumable checkpoints in
`state/`, and the final Hugging Face export in `hf/`.

## Four-node Slurm

The Slurm recipe preserves the 1:1 ratio with 16 actor GPUs and 16 vLLM GPUs:

```bash
CONTAINER_IMAGE=/shared/images/molt-cu13.sqsh \
  sbatch examples/skill2env/train_rl_slurm.sh
```

Override `CONTAINER_MOUNTS` when the repository, `/raid`, model cache, or output
root use different shared paths. The Slurm path is supplied for deployment but
is not part of RFC 05 acceptance testing.

## Existing SFT recipe

The previous Skill2Env SFT workflow remains available from the same example:

```bash
bash examples/skill2env/train_sft.sh
```
