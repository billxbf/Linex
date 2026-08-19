# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import types
from pathlib import Path

import pytest

from molt.cli.train_rl_ray import _ray_runtime_env_vars

COUNT_STARS_TASK = Path(__file__).resolve().parents[2] / "examples" / "polar" / "count_stars" / "task.yaml"


def _run_cli(*extra_args: str):
    return subprocess.run(
        [
            sys.executable,
            "-m",
            "molt.cli.train_rl_ray",
            "--actor.model_name_or_path",
            "unused",
            "--vllm.num_engines",
            "1",
            "--vllm.tool_call_parser",
            "qwen3_coder",
            *extra_args,
            "--fsdp.pp_size",
            "2",
        ],
        capture_output=True,
        check=False,
        text=True,
    )


def test_strict_sync_does_not_warn_off_policy():
    result = _run_cli("--train.force_sync_mode")

    assert result.returncode != 0
    assert "pipeline-parallel" in result.stderr
    assert "may be off-policy" not in result.stdout


def test_partial_rollout_requires_logprob_correction():
    result = _run_cli("--train.partial_rollout_enable")

    assert result.returncode != 0
    assert "Set --algo.advantage.is_correction_level" in result.stderr


def test_corrected_partial_rollout_passes_cli_gate():
    result = _run_cli(
        "--train.partial_rollout_enable",
        "--algo.advantage.is_correction_level",
        "geo",
    )

    assert result.returncode != 0
    assert "pipeline-parallel" in result.stderr
    assert "Per-token IS is correcting those off-policy tokens" in result.stdout


def test_vlm_passes_the_polar_cli_gate():
    result = _run_cli(
        "--data.max_images_per_prompt",
        "1",
        "--rollout.task_spec",
        str(COUNT_STARS_TASK),
        "--vllm.moe_backend",
        "triton",
    )

    assert result.returncode != 0
    assert "pipeline-parallel" in result.stderr
    assert "deferred" not in result.stderr


@pytest.mark.parametrize(
    "extra_args",
    [
        ("--train.agent_path", "unused"),
        ("--train.routing_replay",),
        ("--algo.advantage.estimator", "on_policy_distill"),
    ],
)
def test_removed_rollout_configuration_is_rejected(extra_args):
    result = _run_cli(*extra_args)

    assert result.returncode != 0
    assert "unrecognized arguments" in result.stderr or "invalid choice" in result.stderr


def test_ray_runtime_env_forwards_wandb_settings(monkeypatch):
    expected = {
        "NCCL_CUMEM_ENABLE": "0",
        "NCCL_P2P_DISABLE": "1",
        "VLLM_ALLREDUCE_USE_SYMM_MEM": "0",
        "VLLM_USE_NCCL_SYMM_MEM": "0",
        "WANDB_API_KEY": "test-api-key",
        "WANDB_ENTITY": "test-entity",
        "WANDB_MODE": "offline",
    }
    for name, value in expected.items():
        monkeypatch.setenv(name, value)

    env_vars = _ray_runtime_env_vars()

    assert {name: env_vars[name] for name in expected} == expected


def test_ray_runtime_env_omits_empty_optional_settings(monkeypatch):
    for name in ("WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE"):
        monkeypatch.delenv(name, raising=False)

    env_vars = _ray_runtime_env_vars()

    assert not {"WANDB_API_KEY", "WANDB_ENTITY", "WANDB_MODE"} & env_vars.keys()


def test_polar_gateways_wrap_vllm_weight_update(monkeypatch):
    try:
        import vllm
    except ModuleNotFoundError:
        vllm = types.ModuleType("vllm")
        vllm.__version__ = "0.27.1"
        vllm.AsyncEngineArgs = type("AsyncEngineArgs", (), {})
        vllm.AsyncLLMEngine = type("AsyncLLMEngine", (), {})
        monkeypatch.setitem(sys.modules, "vllm", vllm)
    if "ray.util.queue" not in sys.modules:
        ray_queue = types.ModuleType("ray.util.queue")
        ray_queue.Queue = type("Queue", (), {})
        monkeypatch.setitem(sys.modules, "ray.util.queue", ray_queue)

    from molt.trainer import rl_trainer

    events = []

    class RemoteCall:
        def __init__(self, name):
            self.name = name

        def remote(self, *_args):
            events.append((self.name, *_args) if _args else self.name)
            return self.name

    gateway = types.SimpleNamespace(
        pause=RemoteCall("gateway_pause"),
        resume=RemoteCall("gateway_resume"),
    )
    metadata = getattr(rl_trainer.TrainingActor, "__ray_metadata__", None)
    actor_class = metadata.modified_class if metadata else rl_trainer.TrainingActor
    actor = object.__new__(actor_class)
    actor.vllm_lock = types.SimpleNamespace(
        acquire=RemoteCall("lock_acquire"),
        release=RemoteCall("lock_release"),
    )
    actor.polar_gateways = [gateway]
    actor._gateway_pause_timeout = 90.0
    actor.vllm_engines = []
    actor._prefix_caching_enabled = True

    monkeypatch.setattr(rl_trainer.ray, "get", lambda value: value)
    monkeypatch.setattr(
        rl_trainer,
        "batch_vllm_engine_call",
        lambda _engines, method: events.append(method),
    )
    monkeypatch.setattr(
        rl_trainer.BaseRLTrainer,
        "broadcast_to_vllm",
        lambda _actor: events.append("broadcast"),
    )

    actor.broadcast_to_vllm()

    assert events == [
        "lock_acquire",
        ("gateway_pause", 90.0),
        "pause_generation",
        "broadcast",
        "reset_prefix_cache",
        "resume_generation",
        "gateway_resume",
        "lock_release",
    ]
