# Count Stars Example

A minimal image-input (VLM) run. Each harness gets the same
`polar_stars.png` in its workspace, inspects it, and writes the number of
visible stars to `answer.txt`. Use it to check that harnesses can work from an
image through the local OpenAI-compatible inference backend.

## Prerequisites

Install LiNex with vLLM as described in the
[top-level README](../../../README.md#installation). This example assumes one
node with **8×H100**.

Adjust the setup and topology for your hardware.

## Quick Start

### 1. Build the runtime image (once)

```bash
uv run python examples/polar/count_stars/build_image.py
```

### 2. Start two inference servers

The served model must be a vision-language model (`Qwen/Qwen3.6-27B` is
multimodal).

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
POLAR_TOPOLOGY=examples/polar/count_stars/topology.yaml uv run python -m polar.rollout.server
POLAR_TOPOLOGY=examples/polar/count_stars/topology.yaml POLAR_GATEWAY_NODE_ID=localhost-node-01 uv run python -m polar.gateway.server
POLAR_TOPOLOGY=examples/polar/count_stars/topology.yaml POLAR_GATEWAY_NODE_ID=localhost-node-02 uv run python -m polar.gateway.server
```

### 4. Run

Submits each example harness and prints a completion comparison:

```bash
uv run python examples/polar/count_stars/run.py
```

Use Apptainer instead of Docker with `--backend apptainer`.
