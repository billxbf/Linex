# TMax-15K-Harbor training recipe

TMax supplies containerized terminal-agent tasks with programmatic Harbor
verifiers. Molt launches the full training stack; Polar runs each harness and
returns token-faithful traces and verifier rewards to Molt.

## Prepare

Pull the task directories and build the selected runtime images:

```bash
uv pip install harbor
harbor download 'tmax/TMax-15K-Harbor@latest' --export --output-dir ~/tmax15k
uv run python examples/polar/tmax-15k/build_images.py \
  --dataset-dir ~/tmax15k --max-tasks 10
```

Materialize the instruction and complete Polar task shape for every row:

```bash
uv run python examples/polar/tmax-15k/submit_tmax_tasks.py \
  --dataset-dir ~/tmax15k --harness codex --max-tasks 10 \
  --output examples/polar/tmax-15k/training.jsonl
```

The exported task directory must be visible at the same path on gateway nodes
because the evaluator uploads each task's `tests/`. Runtime images and pinned
harness dependencies must likewise be available on those nodes.

## Train

For a small single-node smoke run:

```bash
MODEL_PATH=/path/to/Qwen3-4B \
PROMPT_DATASET=$PWD/examples/polar/tmax-15k/training.jsonl \
TASK_SPEC= MAX_SAMPLES=10 POLAR_SESSION_TIMEOUT=3600 \
  bash examples/molt/scripts/quick_start/rl_qwen3_4b.sh
```

Molt derives and saves the topology; do not launch Polar or vLLM separately.
Sample count and timeout are Molt CLI settings, not dataset fields.

## Apptainer

On a Docker-capable machine, convert built images to `.sif`, copy them and the
dataset directory to the cluster, then materialize rows that point at the
shared `.sif` directory:

```bash
uv run python examples/polar/tmax-15k/prepare_apptainer_images.py \
  --dataset-dir ~/tmax15k --image-dir ~/tmax15k-sif --max-tasks 10
uv run python examples/polar/tmax-15k/submit_tmax_tasks.py \
  --dataset-dir ~/tmax15k --harness codex --max-tasks 10 \
  --runtime-backend apptainer --apptainer-image-dir ~/tmax15k-sif
```

The launch environment must already provide Docker or Apptainer; Linex does
not provision container runtimes.
