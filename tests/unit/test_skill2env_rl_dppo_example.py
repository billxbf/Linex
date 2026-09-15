from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

import pytest

from examples.skill2env_rl_dppo.prepare import PI_INSTALL
from polar.rollout.models import TaskSpec


@pytest.mark.parametrize("evaluator", ["harbor", "harbor_rubric"])
def test_prepare_writes_pi_prefix_merging_harbor_task(tmp_path: Path, monkeypatch, evaluator: str) -> None:
    monkeypatch.setenv("NVIDIA_API_KEY", "unit-test-judge-key")
    dataset = tmp_path / "dataset"
    task_dir = dataset / "task_example_abcd1234"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Fix the workspace.\n")
    (task_dir / "environment" / "Dockerfile").write_text('FROM scratch\nWORKDIR "/work dir"\n')
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (task_dir / "tests" / "rubric.md").write_text("Verify your changes and report accurately.\n")
    (task_dir / "task.toml").write_text(
        '[task]\nname = "skill2env/task_example_abcd1234"\n[metadata]\nsource_skill = "local/example"\n'
        "[verifier]\ntimeout_sec = 900\n"
    )
    images = tmp_path / "images"
    images.mkdir()
    (images / "task_example_abcd1234.sif").write_bytes(b"test")
    output = tmp_path / "out" / "train_tasks.jsonl"

    subprocess.run(
        [
            sys.executable,
            "examples/skill2env_rl_dppo/prepare.py",
            "--dataset-dir",
            str(dataset),
            "--image-dir",
            str(images),
            "--output",
            str(output),
            "--skip-image-check",
            "--evaluator",
            evaluator,
            "--rubric-coefficient",
            "0.15",
        ],
        check=True,
    )

    row = json.loads(output.read_text())
    task = TaskSpec.model_validate(row["task"])
    assert row["prompt"] == "Fix the workspace."
    assert task.runtime.image == str((images / "task_example_abcd1234.sif").resolve())
    assert task.runtime.workdir == "/work dir"
    assert task.runtime.memory_mb == 16384
    assert "pi-coding-agent@0.84.2" in task.runtime.prepare[0].command
    assert task.agent.harness == "pi"
    assert task.agent.model_name == "openai/Qwen/Qwen3.8-27B"
    assert task.agent.settings == {"context_window": 65536, "thinking": "high"}  # 98304 total - 32768 new tokens
    assert task.builder.strategy == "prefix_merging"
    assert task.evaluator.strategy == evaluator
    assert task.evaluator.refresh_runtime is False
    expected_config = {"tests_dir": str((task_dir / "tests").resolve()), "verifier_timeout": 900.0}
    if evaluator == "harbor_rubric":
        expected_config.update(
            judge_base_url="https://inference-api.nvidia.com/v1",
            judge_model="openai/openai/gpt-6-astra",
            judge_api_key_env="NVIDIA_API_KEY",
            rubric_coefficient=0.15,
        )
    assert task.evaluator.config == expected_config
    assert task.evaluator.env == {}
    assert "unit-test-judge-key" not in output.read_text()
    assert task.metadata["skill2env_task"] == "task_example_abcd1234"


@pytest.mark.parametrize("upgrade_node", [False, True])
def test_pi_install_bypasses_task_npm_shim(tmp_path: Path, upgrade_node: bool) -> None:
    node_bin = tmp_path / "node" / "bin"
    node_bin.mkdir(parents=True)
    node = node_bin / "node"
    # Stand in for Node with Python so this shell regression needs no Node installation.
    node.write_text(
        f"#!{sys.executable}\n"
        "import os, sys\n"
        "if sys.argv[1] == '-e': sys.exit(0)\n"
        "os.execv(sys.executable, [sys.executable, *sys.argv[1:]])\n"
    )
    node.chmod(0o755)
    npm_cli = node_bin.parent / "lib/node_modules/npm/bin/npm-cli.js"
    npm_cli.parent.mkdir(parents=True)
    capture = tmp_path / "npm-arguments.json"
    npm_cli.write_text(
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_NPM'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    for name in ("npm", "pi"):
        shim = node_bin / name
        shim.write_text("#!/bin/sh\necho 'offline contract shim' >&2\nexit 2\n")
        shim.chmod(0o755)
    command = PI_INSTALL
    initial_bin = node_bin
    if upgrade_node:
        archive = tmp_path / "node.tar.gz"
        with tarfile.open(archive, "w:gz") as tar:
            tar.add(node_bin.parent, arcname="node")
        initial_bin = tmp_path / "old-bin"
        initial_bin.mkdir()
        old_node = initial_bin / "node"
        old_node.write_text("#!/bin/sh\necho 'cached old Node selected' >&2\nexit 1\n")
        old_node.chmod(0o755)
        command = PI_INSTALL.replace(
            "/usr/bin/curl -LsSf https://nodejs.org/dist/v22.23.2/node-v22.23.2-linux-x64.tar.gz",
            f"cat {shlex.quote(str(archive))}",
        )
        assert command != PI_INSTALL
    result = subprocess.run(
        ["bash", "-c", command],
        env={
            **os.environ,
            "HOME": str(tmp_path / "home"),
            "PATH": f"{initial_bin}:{os.environ['PATH']}",
            "CAPTURE_NPM": str(capture),
        },
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(capture.read_text()) == [
        "install", "-g", "--offline=false", "--registry=https://registry.npmjs.org",
        "@earendil-works/pi-coding-agent@0.84.2",
    ]
    assert "offline contract shim" in (node_bin / "npm").read_text()


@pytest.mark.parametrize("samples,batch,inflight", [(None, 64, 128), (8, 128, 256), (4, 64, 128)])
def test_launch_converts_rollout_counts_to_groups(tmp_path, monkeypatch, samples, batch, inflight):
    commands = tmp_path / "bin"
    commands.mkdir()
    for name in ("ray", "apptainer"):
        command = commands / name
        command.write_text("#!/bin/sh\nexit 0\n")
        command.chmod(0o755)
    capture = tmp_path / "arguments.json"
    python = commands / "python3"
    python.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        "with open(os.environ['CAPTURE_ARGS'], 'w') as f: json.dump(sys.argv[1:], f)\n"
    )
    python.chmod(0o755)
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text("{}\n")
    for name, value in {
        "PATH": f"{commands}:{os.environ['PATH']}",
        "MODEL_PATH": str(tmp_path),
        "SAVE_ROOT": str(tmp_path / "output"),
        "PROMPT_DATASET": str(dataset),
        "SAMPLES_PER_PROMPT": str(samples),
        "ROLLOUT_BATCH_SIZE": str(batch),
        "INFLIGHT_ROLLOUTS": str(inflight),
        "PARTIAL_ROLLOUT": "1",
        "ASYNC_QUEUE_SIZE": "2",
        "EVAL_DATASET": str(dataset),
        "CAPTURE_ARGS": str(capture),
    }.items():
        monkeypatch.setenv(name, value)
    if samples is None:
        monkeypatch.delenv("SAMPLES_PER_PROMPT")
        samples = 8
    subprocess.run(["bash", "examples/skill2env_rl_dppo/train_rl.sh"], check=True)
    arguments = json.loads(capture.read_text())
    for flag, expected in {
        "--rollout.n_samples_per_prompt": samples,
        "--rollout.batch_size": batch // samples,
        "--rollout.vllm_generate_batch_size": inflight // samples,
        "--train.batch_size": batch,
        "--train.async_queue_size": 2,
        "--train.max_epochs": 1,
        "--actor.loss_mode": "dppo",
        "--algo.advantage.estimator": "grpo",
        "--algo.advantage.is_correction_level": "off",
        "--eval.n_samples_per_prompt": 1,
    }.items():
        assert arguments[arguments.index(flag) + 1] == str(expected)
    assert "--train.partial_rollout_enable" in arguments
    assert "--train.force_on_policy" in arguments
    assert "--train.force_sync_mode" not in arguments


def test_rubric_launch_requires_key_before_starting_services(tmp_path, monkeypatch):
    dataset = tmp_path / "tasks.jsonl"
    dataset.write_text(json.dumps({"task": {"evaluator": {"config": {"judge_api_key_env": "NVIDIA_API_KEY"}}}}))
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)
    monkeypatch.setenv("MODEL_PATH", str(tmp_path))
    monkeypatch.setenv("PROMPT_DATASET", str(dataset))
    result = subprocess.run(["bash", "examples/skill2env_rl_dppo/train_rl.sh"], capture_output=True, text=True)
    assert result.returncode == 1
    assert "NVIDIA_API_KEY is required" in result.stderr
