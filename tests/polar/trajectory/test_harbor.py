from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from polar.gateway.node import GatewayNodeManager
from polar.runtime.base import BaseRuntime
from polar.runtime.models import ExecResult, RuntimeSpec
from polar.trajectory.evaluator.harbor import HarborEvaluator
from polar.trajectory.models import EvaluatorSpec, Trace, Trajectory


class FakeRuntime(BaseRuntime):
    def __init__(self, tmp_path: Path, reward: str) -> None:
        super().__init__(RuntimeSpec(image="fake"), "session", tmp_path / "session")
        self.reward = reward
        self.commands: list[str] = []
        self.uploads: list[tuple[str, str]] = []

    @property
    def runtime_id(self) -> str:
        return "fake"

    async def start(self) -> None: ...

    async def stop(self) -> None: ...

    async def exec(self, command: str, **kwargs) -> ExecResult:
        self.commands.append(command)
        if "reward.txt" in command:
            return ExecResult(stdout=self.reward, return_code=0)
        if "reward.json" in command:
            return ExecResult(stdout="", return_code=1)
        if command == "bash /tests/test.sh":
            return ExecResult(stdout="verifier ran\n", return_code=0)
        return ExecResult(stdout="", return_code=0)

    async def upload_file(self, local_path: str, remote_path: str) -> None: ...

    async def upload_dir(self, local_path: str, remote_path: str) -> None:
        self.uploads.append((local_path, remote_path))

    async def download_file(self, local_path: str, remote_path: str) -> None: ...

    async def download_dir(self, local_path: str, remote_path: str) -> None: ...


@pytest.mark.parametrize(("raw_reward", "expected"), [("1", 1.0), ("0.25", 0.25), ("3", 1.0)])
def test_plain_harbor_uses_final_runtime_reward(tmp_path: Path, raw_reward: str, expected: float) -> None:
    tests_dir = tmp_path / "task" / "tests"
    tests_dir.mkdir(parents=True)
    (tests_dir / "test.sh").write_text("#!/bin/bash\n")
    runtime = FakeRuntime(tmp_path, raw_reward)
    evaluator = HarborEvaluator(tests_dir=str(tests_dir), verifier_timeout=600)
    trajectory = Trajectory(status="COMPLETED", traces=[Trace(), Trace()])

    result = asyncio.run(
        evaluator.evaluate(
            trajectory,
            runtime=runtime,
            artifacts_dir=tmp_path / "artifacts",
            env={},
            timeout_seconds=30,
        )
    )

    assert result.outcome_reward == expected
    assert result.trace_rewards is None
    assert runtime.uploads == [(str(tests_dir), "/tests")]
    assert "bash /tests/test.sh" in runtime.commands
    assert (tmp_path / "artifacts" / "verifier.stdout.log").read_text() == "verifier ran\n"

    merged = GatewayNodeManager._merge_eval_result(
        trajectory,
        result,
        EvaluatorSpec(strategy="harbor", refresh_runtime=False),
    )
    assert [trace.reward for trace in merged.traces] == [expected, expected]
    assert merged.metadata["evaluation"]["strategy"] == "harbor"
