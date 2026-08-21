# Linex

Linex combines [Molt](https://github.com/NVIDIA-NeMo/labs-molt) training with
[Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) container-agent
rollout. Molt is the only training CLI and owns Ray, vLLM, sampling, weight
updates, and optimization. Polar owns harness execution, container lifecycles,
trajectory construction, and evaluation.

## Install

Linex requires Python 3.11 or newer:

```bash
python3 -m pip install -e .
```

Supervised fine-tuning remains available through `molt.cli.train_sft`; the
Molt–Polar integration changes only the RL rollout path.

Training nodes need NVIDIA GPUs. Every Ray node eligible to host a Polar
gateway must already provide the Docker or Apptainer runtime selected by the
task. Linex does not provision daemons, sockets, binaries, images, mounts, or
permissions. Task-specific agent dependencies belong in the task runtime image
or its `runtime.prepare` actions.

## Training flow

One `python -m molt.cli.train_rl_ray` job starts the complete stack:

1. Ray-managed vLLM engines and one vLLM router.
2. Ray-managed Polar rollout and gateway services.
3. Container sessions for each prompt group.
4. Polar `Trace` records and their media references converted directly
   into Molt `Experience` batches.
5. Molt policy optimization and coordinated weight synchronization.

Molt derives gateway URLs, router URL, served model, and concurrency after Ray
placement. It writes the resolved `topology.json` under
`--rollout.save_dir`. The same directory is the shared root for VLM media and
other retained rollout outputs; users do not configure another artifact path.
On multi-node runs, that path must be visible at the same absolute location on
every Ray node.

## Task configuration

For a uniform recipe, pass one validated YAML specification:

```bash
python -m molt.cli.train_rl_ray \
  --actor.model_name_or_path /models/Qwen3-4B \
  --data.prompt_dataset prompts.jsonl \
  --data.input_key prompt \
  --rollout.task_spec task.yaml \
  --vllm.num_engines 1 \
  --vllm.tool_call_parser qwen3_coder \
  --vllm.reasoning_parser qwen3 \
  ...
```

The RL dataset owns only the string task instruction and an optional complete
Polar task specification. The task YAML owns `runtime`, `agent`, `builder`,
`evaluator`, and optional metadata. Molt CLI flags remain the sole owner of sampling,
samples per prompt, batching, gateway count/concurrency, session timeout,
persistence, and asynchronous/partial rollout policy.

When runtime or evaluator data varies by instance, put a complete task object
in each dataset row under `task` (or `--data.task_key`) and omit
`--rollout.task_spec`:

```json
{"prompt":"Fix the bug","task":{"runtime":{"image":"task:1"},"agent":{"harness":"codex"},"builder":{"strategy":"prefix_merging"},"evaluator":{"strategy":"session_completed"}}}
```

Partial rollout requires rollout-log-probability correction. During refits,
Molt pauses and drains Polar gateways before pausing vLLM, broadcasts weights,
resets enabled prefix caches, then resumes vLLM and the gateways. Use
`--train.force_sync_mode` for strict batches that leave no session in flight
across a policy update.

## Recipes

- [Calculator](examples/polar/calculator/README.md): smallest complete optimizer-step smoke test.
- [SWE-bench Verified](examples/polar/swebench_verified/README.md): per-instance runtime and evaluator rows.
- [TMax-15K-Harbor](examples/polar/tmax-15k/README.md): terminal-agent tasks and Harbor rewards.
- [Count Stars](examples/polar/count_stars/README.md): one-step VLM artifact smoke test.
- [Skill2Env](examples/skill2env/README.md): Apptainer/Hermes RL acceptance and the existing SFT recipe.

Text and VLM agent rollouts all run through Polar. Media stays in the shared run
directory while task results carry only paths.
On-policy distillation and Molt's former direct-agent rollout interface are not
part of Linex.

If you are an agent editing this repository, read `AGENTS.md` first.
