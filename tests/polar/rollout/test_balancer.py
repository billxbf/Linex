from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from molt.trainer.rollout.polar import PolarServiceActor
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager
from polar.rollout.models import NodeRegistrationRequest, NodeStageMetrics, SessionResult, TaskRequest, TaskResult
from polar.trajectory.models import Trajectory


def _register(
    scheduler: NodeScheduler,
    node_id: str,
    *,
    max_init_workers: int = 4,
    max_run_workers: int = 2,
    max_postrun_workers: int = 2,
) -> None:
    scheduler.register_node(
        NodeRegistrationRequest(
            node_id=node_id,
            gateway_url=f"http://127.0.0.1/{node_id}",
            max_init_workers=max_init_workers,
            max_run_workers=max_run_workers,
            max_postrun_workers=max_postrun_workers,
            heartbeat_interval_seconds=30,
        )
    )


def test_acquire_prefers_lower_run_pressure_before_init_pressure() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "busy-run", max_run_workers=2)
    _register(scheduler, "free-run", max_run_workers=2)
    scheduler.heartbeat(
        "busy-run",
        metrics=NodeStageMetrics(run_inflight=1, init_queue_depth=0),
    )
    scheduler.heartbeat(
        "free-run",
        metrics=NodeStageMetrics(run_inflight=0, init_queue_depth=3),
    )

    selected = scheduler.acquire_node()

    assert selected is not None
    assert selected.node_id == "free-run"
    assert selected.dispatch_reservations == 1


def test_release_reservation_decrements_dispatch_pressure() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a")

    assert scheduler.acquire_node().dispatch_reservations == 1
    assert scheduler.release_reservation("node-a").dispatch_reservations == 0


def test_draining_node_does_not_receive_new_sessions() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a")
    _register(scheduler, "node-b")

    scheduler.acquire_node()
    drained = scheduler.drain_node("node-a")
    selected = scheduler.acquire_node()

    assert drained.draining is True
    assert selected is not None
    assert selected.node_id == "node-b"


def test_stale_nodes_become_ineligible() -> None:
    scheduler = NodeScheduler(stale_factor=2.5)
    _register(scheduler, "node-a")
    scheduler._nodes["node-a"].last_heartbeat = datetime.now(timezone.utc) - timedelta(minutes=10)

    assert scheduler.acquire_node() is None
    assert scheduler.get_node("node-a").healthy is False


def test_postrun_backlog_blocks_admission() -> None:
    scheduler = NodeScheduler()
    _register(scheduler, "node-a", max_postrun_workers=2)
    scheduler.heartbeat(
        "node-a",
        metrics=NodeStageMetrics(postrun_queue_depth=4),
    )

    assert scheduler.acquire_node() is None


def test_training_task_result_is_returned_once_without_retention() -> None:
    class Pipeline:
        calls = 0

        async def run_batch(self, sessions):
            self.calls += 1
            return [
                SessionResult(
                    session_id=session.session_id,
                    task_id=session.task_id,
                    status="COMPLETED",
                    trajectory=Trajectory(status="COMPLETED"),
                )
                for session in sessions
            ]

        def status(self):
            return {"pending_sessions": 0}

    pipeline = Pipeline()
    manager = RolloutManager(pipeline=pipeline, scheduler=NodeScheduler())
    request = TaskRequest(
        task_id="task-1",
        instruction="Do it",
        num_samples=2,
        agent={"harness": "codex"},
        sampling_params={"temperature": 1.0},
    )

    result = asyncio.run(manager.run_task(request))

    assert pipeline.calls == 1
    assert len(result.results) == 2
    assert manager.status() == {
        "pipeline": {"pending_sessions": 0},
        "nodes": NodeScheduler().stats(),
    }
    assert "_tasks" not in vars(manager)


def test_ray_service_returns_task_directly_without_http_polling(monkeypatch) -> None:
    calls = []

    class Manager:
        async def run_task(self, request):
            calls.append(request.task_id)
            return TaskResult(task_id=request.task_id, instruction=request.instruction, results=[])

    from polar.rollout import server

    monkeypatch.setattr(server, "get_state", lambda: SimpleNamespace(manager=Manager()))
    actor_class = getattr(PolarServiceActor, "__ray_actor_class__", PolarServiceActor)
    actor = object.__new__(actor_class)
    actor.service = "rollout"
    actor._client = object()  # No HTTP methods: any polling would fail this test.
    payload = TaskRequest(
        task_id="task-1",
        instruction="Do it",
        agent={"harness": "codex"},
        sampling_params={"temperature": 1.0},
    ).model_dump(mode="json")

    result = asyncio.run(actor.run_task(payload))

    assert calls == ["task-1"]
    assert result == {"task_id": "task-1", "instruction": "Do it", "results": []}
