# Calculator Example

The smallest end-to-end Polar run. Each harness gets a tiny `calculator.py`
with parser stubs, edits it, and the evaluator runs `python3 test_calculator.py`.
Use it as a quick smoke test that rollout, gateway, runtime, harness execution,
and evaluation all work together.

## Prerequisites

Install **Polar** and one inference backend — **vLLM** or **SGLang** — as described in the
[top-level README](../../README.md#installation). This example assumes 1 node **8×H100**
Adjust the setup and topology for your hardware.

## Quick Start

### 1. Build the runtime image (once)

```bash
uv run python examples/calculator/build_image.py
```

### 2. Start two inference servers

Pick **one** backend (don't install both in the same environment).

**vLLM** → use `topology.vllm.yaml`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run vllm serve Qwen/Qwen3.6-27B --port 8000 \
  --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run vllm serve Qwen/Qwen3.6-27B --port 8001 \
  --tensor-parallel-size 4 --max-model-len 262144 \
  --reasoning-parser qwen3 --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

**SGLang** → use `topology.sgl.yaml`

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 uv run python -m sglang.launch_server --model-path Qwen/Qwen3.6-27B --port 8000 \
  --tp 4 --context-length 262144 --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder

CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python -m sglang.launch_server --model-path Qwen/Qwen3.6-27B --port 8001 \
  --tp 4 --context-length 262144 --mem-fraction-static 0.85 \
  --reasoning-parser qwen3 --tool-call-parser qwen3_coder
```

### 3. Start Polar Servers

Use the topology file that matches your backend (`topology.vllm.yaml` shown; swap for `topology.sgl.yaml`):

```bash
uv run polar serve_rollout -c examples/calculator/topology.vllm.yaml
uv run polar serve_gateway -c examples/calculator/topology.vllm.yaml --node-id localhost-node-01
uv run polar serve_gateway -c examples/calculator/topology.vllm.yaml --node-id localhost-node-02
```

### 4. Run

Submits example harness at once and prints a reward comparison. The same command
works for either backend — it just talks to the rollout server (the inference
engine is whichever one you started in steps 2–3):

```bash
uv run python examples/calculator/run.py
```

Use Apptainer instead of Docker with `--backend apptainer`.

### 5. (Optional) Watch in the dashboard

`topology.vllm.yaml` shown; swap for `topology.sgl.yaml` if you use sglang.

```bash
uv run polar dashboard -c examples/calculator/topology.vllm.yaml
```

Open <http://127.0.0.1:8090> to inspect live tasks, sessions, trajectories,
and evaluations.

