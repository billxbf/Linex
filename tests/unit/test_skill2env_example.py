from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from polar.agent.presets.pi import PiHarness
from polar.rollout.models import TaskSpec


def test_local_recipe_encodes_ten_step_acceptance_topology() -> None:
    script = Path("examples/skill2env/train_rl.sh").read_text()

    for argument in (
        "--data.max_samples 40",
        "--data.max_len 131072",
        "--rollout.batch_size 4",
        "--rollout.n_samples_per_prompt 16",
        "--rollout.max_new_tokens 4096",
        "--train.async_queue_size 2",
        "--train.max_tokens_per_gpu 8192",
        "--train.force_on_policy",
        "--actor.num_gpus_per_node 4",
        "--vllm.tensor_parallel_size 4",
        "--fsdp.attn_implementation te",
        "--fsdp.cp_size 4",
        "--algo.advantage.estimator reinforce_baseline",
        "--algo.advantage.is_correction_level geo",
    ):
        assert argument in script
    assert "command -v apptainer" in script
    assert "--fsdp.packing_samples" not in script
    assert "--train.force_sync_mode" not in script


def test_slurm_recipe_preserves_even_four_node_gpu_split() -> None:
    script = Path("examples/skill2env/train_rl_slurm.sh").read_text()

    assert "#SBATCH --nodes=4" in script
    assert "--actor.num_nodes 2 --actor.num_gpus_per_node 8" in script
    assert "--vllm.num_engines 4 --vllm.tensor_parallel_size 4" in script
    assert "--fsdp.cp_size 4" in script


def test_skill2env_sft_scripts_keep_stages_and_context_separate() -> None:
    readme = Path("examples/skill2env_sft/README.md").read_text()
    prepare = Path("examples/skill2env_sft/prepare.py").read_text()
    rollout = Path("examples/skill2env_sft/rollout_teacher.sh").read_text()
    student = Path("examples/skill2env_sft/train_student.sh").read_text()

    assert "examples/skill2env_sft/prepare.py" in readme
    assert "examples/skill2env/prepare.py" not in readme
    assert '"harness": "pi"' in prepare
    assert "--harness" not in prepare
    assert "--build-missing" in prepare
    assert '"builder": {"strategy": "per_request"}' in prepare
    assert "--eval.eval_only" in rollout
    assert "--eval.dataset" in rollout
    assert "--data.prompt_dataset" not in rollout
    assert "${CONTEXT_LENGTH:-131072}" in rollout
    assert "${VLLM_TP_SIZE:-1}" in rollout
    assert "${VLLM_DP_SIZE:-8}" in rollout
    assert "${EVAL_BATCH_SIZE:-32}" in rollout
    assert "${GATEWAY_CONCURRENCY:-32}" in rollout
    assert "examples.skill2env_sft.export_sft" in rollout

    assert "torchrun --standalone" in student
    assert "--data.input_key prompt_messages" in student
    assert "--data.output_key response_messages" in student
    assert "${CONTEXT_LENGTH:-131072}" in student
    assert "${CP_SIZE:-8}" in student
    assert "/raid/binfeng/models/Qwen3.5-4B-skill2env_sft" in student
    assert "train_rl_ray" not in student and "ray start" not in student


def test_prepare_writes_complete_skill2env_task(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    task_dir = dataset / "task_example_abcd1234"
    (task_dir / "environment").mkdir(parents=True)
    (task_dir / "tests").mkdir()
    (task_dir / "instruction.md").write_text("Fix the workspace.\n")
    (task_dir / "environment" / "Dockerfile").write_text(
        "FROM scratch\nWORKDIR /tmp\nWORKDIR /work\n"
    )
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")
    (task_dir / "task.toml").write_text(
        """artifacts = [\"/workspace/result.txt\"]
[task]
name = "skill2env/task_example_abcd1234"
description = "Example"
[metadata]
source_skill = "local/example"
[agent]
timeout_sec = 1800
[verifier]
timeout_sec = 600
[environment]
cpus = 2
memory_mb = 4096
storage_mb = 4096
"""
    )
    images = tmp_path / "images"
    images.mkdir()
    (images / "task_example_abcd1234.sif").write_bytes(b"test")
    output = tmp_path / "training.jsonl"

    subprocess.run(
        [
            sys.executable,
            "examples/skill2env/prepare.py",
            "--dataset-dir",
            str(dataset),
            "--image-dir",
            str(images),
            "--output",
            str(output),
            "--skip-image-check",
        ],
        check=True,
    )

    row = json.loads(output.read_text())
    task = TaskSpec.model_validate(row["task"])
    assert row["prompt"] == "Fix the workspace."
    assert task.runtime.backend == "apptainer"
    assert task.runtime.image == str((images / "task_example_abcd1234.sif").resolve())
    assert task.runtime.network == "host"
    assert task.runtime.allow_internet is True
    assert task.runtime.workdir == "/work"
    assert (task.runtime.cpus, task.runtime.memory_mb, task.runtime.storage_mb) == (None, None, None)
    assert "uv tool install --force --managed-python" in task.runtime.prepare[0].command
    assert "node-v22.23.2-linux-x64.tar.gz" in task.runtime.prepare[0].command
    assert task.agent.harness == "hermes"
    assert task.agent.settings["context_length"] == 131072
    assert task.agent.settings["max_turns"] == 99
    assert task.agent.settings["toolsets"] == "terminal,file"
    assert task.builder.strategy == "prefix_merging"
    assert task.evaluator.strategy == "harbor"
    assert task.evaluator.refresh_runtime is False
    assert task.evaluator.config["tests_dir"] == str((task_dir / "tests").resolve())
    assert task.metadata["harbor_metadata"]["source_skill"] == "local/example"
    assert task.metadata["harbor_environment"] == {"cpus": 2, "memory_mb": 4096, "storage_mb": 4096}

    pi_output = tmp_path / "pi.jsonl"
    subprocess.run(
        [
            sys.executable,
            "examples/skill2env_sft/prepare.py",
            "--dataset-dir",
            str(dataset),
            "--image-dir",
            str(images),
            "--output",
            str(pi_output),
            "--skip-image-check",
            "--model-name",
            "openai/Qwen/Qwen3.8-27B",
            "--context-window",
            "65536",
        ],
        check=True,
    )

    pi_task = TaskSpec.model_validate(json.loads(pi_output.read_text())["task"])
    assert pi_task.agent.harness == "pi"
    assert pi_task.agent.model_name == "openai/Qwen/Qwen3.8-27B"
    assert pi_task.agent.settings == {"context_window": 65536}
    assert pi_task.builder.strategy == "per_request"
    assert "@earendil-works/pi-coding-agent@0.84.2" in pi_task.runtime.prepare[0].command
    assert 'pi --version 2>/dev/null || true' in pi_task.runtime.prepare[0].command
    pi_command = PiHarness(pi_task.agent).run_steps("fix it")[0].command
    assert 'export PATH="$HOME/.local/node/bin:$HOME/.local/bin:$PATH"' in pi_command
