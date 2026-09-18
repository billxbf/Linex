# Linex (wip)

**Linex** is the all-in-one infra for Agentic RL - training, inference, rollout, sandbox layers combined without glue.

**Linex** features minimalism - optimized interfaces, boilerplate-free, light dependency, and the one-piece philosophy (deprecated recipe and model supports are dynamically dropped).

It's a power fork from [Molt](https://github.com/NVIDIA-NeMo/labs-molt) (Pytorch FSDP + glue-free Trainer) and [Polar](https://github.com/NVIDIA-NeMo/ProRL-Agent-Server) (efficient containerized agent rollout). 

<p align="center">
  <img src="assets/linex.png" alt="linex" width="700">
</p>

## Install

Linex requires Python 3.11 or newer:

```bash
python3 -m pip install -e .
```

## Recipes

[Skill2Env RL DPPO](examples/skill2env_rl_dppo/README.md): asynchronous DPPO training with [skill2env](https://github.com/NVlabs/Skill2Env/tree/main) dataset.
