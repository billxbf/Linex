# LiNex

Nenux is a power fork from [Molt](https://github.com/NVIDIA-NeMo/labs-molt)🦋 (distributed RL) and [Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server)⭐ (harness-native rollout) with glue layer removed.

The goal is to find the ultra *simple* and *effective* Agentic RL infra & recipe at scale.

## Installation

LiNex requires Python 3.11 or newer. Install the repository package and its
vLLM rollout dependencies from the repository root:

```bash
python3 -m pip install -e .
```

Molt's existing training modules remain the user-facing CLI. Polar is retained
as the internal rollout service package under `polar/`.

If you are an Agent, make sure to read `AGENTS.md` before edition.
