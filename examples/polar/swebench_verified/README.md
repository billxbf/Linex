# SWE-bench Verified training recipe

This recipe trains through the Molt CLI while Polar runs each coding agent in
its per-instance SWE-bench container and applies the official evaluator. There
is no separately launched Polar service or user-maintained topology.

## Prepare

Install the SWE-bench extra and build the runtime images on every possible
gateway node:

```bash
uv pip install -e '.[swebench]'
uv run python examples/polar/swebench_verified/build_images.py --max-tasks 10
```

Materialize complete, validated Polar task shapes in the training dataset:

```bash
uv run python examples/polar/swebench_verified/submit_swebench_tasks.py \
  --harness codex --max-tasks 10 \
  --output examples/polar/swebench_verified/training.jsonl
```

Each row owns its instruction, runtime image, evaluator instance, and other
instance-specific task data. Molt remains the sole owner of sampling, sample
count, timeout, gateway concurrency, task identity, and training settings.

## Train

The small single-node configuration can be reused for a smoke run:

```bash
MODEL_PATH=/path/to/Qwen3-4B \
PROMPT_DATASET=$PWD/examples/polar/swebench_verified/training.jsonl \
TASK_SPEC= MAX_SAMPLES=10 POLAR_SESSION_TIMEOUT=3600 \
  bash examples/molt/scripts/quick_start/rl_qwen3_4b.sh
```

For a production run, carry the same dataset arguments into the Slurm recipe
and size actor/vLLM resources for the selected model. Task-specific harness and
SWE-bench dependencies belong in the runtime images or `runtime.prepare`.
