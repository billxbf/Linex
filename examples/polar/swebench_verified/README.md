# SWE-bench Verified Example

Evaluate Polar agent harnesses on [SWE-bench Verified](https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified)
(500 human-validated tasks). Each task runs an agent inside a per-instance
container at the repo's `base_commit`, then grades the patch with the official
`swebench` harness.

## Prerequisites

Install LiNex with vLLM as described in the
[top-level README](../../../README.md#installation). This example also needs
the official SWE-bench grading harness:

```bash
uv pip install -e ".[swebench]"
```

This example assumes 1 node **8×H100** — two inference servers (tensor-parallel 4 each).

Adjust the setup and topology for your hardware.

## Quick Start

### 1. Build runtime images

Each runtime image layers Node.js on the per-instance SWE-bench image; harness
CLIs install at task time during the **INIT** stage. Build a subset first:

```bash
uv run python examples/polar/swebench_verified/build_images.py --max-tasks 10   # or no flag for all 500
```

### 2. Start two inference servers

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve Qwen/Qwen3.6-27B --port 8000 \
  --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run vllm serve Qwen/Qwen3.6-27B --port 8001 \
  --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

### 3. Start Polar services

```bash
POLAR_TOPOLOGY=examples/polar/swebench_verified/topology.yaml uv run python -m polar.rollout.server
POLAR_TOPOLOGY=examples/polar/swebench_verified/topology.yaml POLAR_GATEWAY_NODE_ID=localhost-node-01 uv run python -m polar.gateway.server
POLAR_TOPOLOGY=examples/polar/swebench_verified/topology.yaml POLAR_GATEWAY_NODE_ID=localhost-node-02 uv run python -m polar.gateway.server
```

### 4. Submit tasks

Pick a harness and how many tasks to run; the resolved-rate summary prints to
the console when the batch finishes. Supported harnesses: `claude_code`, `codex`, `opencode`, `qwen_code`.


```bash
# pass@1 over the first 10 tasks
uv run python examples/polar/swebench_verified/submit_swebench_tasks.py --harness claude_code --max-tasks 10

# pass@8 over the first 10 tasks
uv run python examples/polar/swebench_verified/submit_swebench_tasks.py --harness claude_code --max-tasks 10 --num-samples 8

# a single instance
uv run python examples/polar/swebench_verified/submit_swebench_tasks.py --harness codex --instance-id django__django-15098
```

Use Apptainer instead of Docker with `--runtime-backend apptainer`.
