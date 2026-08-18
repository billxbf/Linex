# Calculator training recipe

This is the smallest container-agent RL recipe. Molt is the only launcher: it
creates Ray, the policy vLLM engines and router, the Polar rollout server and
gateways, then trains directly from Polar traces. The generated runtime
topology is saved under the configured `--rollout.save_dir`.

The task gives Codex a small `calculator.py`, asks it to implement a recursive
descent parser, and rewards the resulting patch with the containerized tests.

## Prerequisites

- Install Linex from the repository root.
- Make Docker available on every Ray node that may host a Polar gateway.
- Build the task runtime image once:

```bash
uv run python examples/polar/calculator/build_image.py
```

The Linex image does not install or mount Docker/Apptainer. Agent-specific
packages belong in the task image or in `runtime.prepare`; this recipe installs
the pinned Codex CLI during session initialization.

## Run one optimizer step

On a single eight-GPU node:

```bash
MODEL_PATH=/path/to/Qwen3-4B \
  bash examples/molt/scripts/quick_start/rl_qwen3_4b.sh
```

The launch consumes [`prompts.jsonl`](prompts.jsonl) and the validated Polar
[`task.yaml`](task.yaml). Override `PROMPT_DATASET` or `TASK_SPEC` to reuse the
same Molt recipe. No hand-written topology or separately launched inference
server is used.
