import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from polar.gateway.node import GatewayNodeManager
from polar.rollout.timer import StageTimer
from polar.trajectory.models import Trace, Trajectory


def _trace(reward=None):
    return Trace(prompt_ids=[1, 2], response_ids=[3, 4], loss_mask=[1, 1], response_logprobs=[-0.1, -0.2], reward=reward)


def test_apply_timeout_reward_marks_every_trace_as_a_failure() -> None:
    trajectory = Trajectory(status="COMPLETED", metadata={"builder": "prefix_merging"}, traces=[_trace(), _trace(0.9)])

    result = GatewayNodeManager._apply_timeout_reward(trajectory, 0.0, "session execution timeout")

    assert result.status == "TIMEOUT"
    assert result.error == "session execution timeout"
    assert [trace.reward for trace in result.traces] == [0.0, 0.0]
    assert result.metadata["builder"] == "prefix_merging"
    assert result.metadata["evaluation"] == {"strategy": "timeout_reward", "outcome_reward": 0.0, "trace_rewards": None}


def test_grant_postrun_budget_lifts_an_exhausted_deadline_but_keeps_a_larger_one() -> None:
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.postrun_grace_seconds = 300.0

    async def run() -> tuple[float, float, float]:
        now = asyncio.get_running_loop().time()
        exhausted = SimpleNamespace(execution_deadline=now - 5.0)
        generous = SimpleNamespace(execution_deadline=now + 1000.0)
        manager._grant_postrun_budget(exhausted)
        manager._grant_postrun_budget(generous)
        return now, exhausted.execution_deadline, generous.execution_deadline

    now, lifted, kept = asyncio.run(run())
    assert 299.0 <= lifted - now <= 301.0
    assert kept - now == 1000.0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure,timeout_reward,status,reward",
    [(TimeoutError, 0.0, "TIMEOUT", 0.0), (TimeoutError, None, "TIMEOUT", None), (RuntimeError, 0.0, "ERROR", None)],
)
async def test_eval_failure_preserves_trajectory_and_timeout_policy(tmp_path, failure, timeout_reward, status, reward):
    manager = GatewayNodeManager.__new__(GatewayNodeManager)
    manager.timeout_reward = timeout_reward
    manager.node_id = "gateway-test"
    manager.default_runtime = None
    manager.session_registry = SimpleNamespace(set_status=lambda *args: None)
    manager._build_trajectory = lambda request: Trajectory(status="COMPLETED", traces=[_trace()])
    evaluator = SimpleNamespace(evaluate=AsyncMock(side_effect=failure("evaluation stopped")))
    manager.evaluators = SimpleNamespace(create=lambda spec: evaluator)
    request = SimpleNamespace(
        session_id="session-test",
        task_id="task-test",
        evaluator=SimpleNamespace(strategy="harbor", config={}, env={}, refresh_runtime=False),
        runtime=None,
        metadata={},
    )
    managed = SimpleNamespace(
        request=request,
        agent_result=SimpleNamespace(status="completed", error=None),
        timer=StageTimer(),
        runtime=object(),
        session_dir=tmp_path,
        artifacts_dir=tmp_path / "artifacts",
        execution_deadline=asyncio.get_running_loop().time() + 30,
    )

    result = await manager._build_session_result(managed)

    assert result.status == status
    assert len(result.trajectory.traces) == 1
    assert result.trajectory.traces[0].reward == reward
    assert result.error
