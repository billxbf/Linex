"""Top-level task orchestration for rollout batches."""

from __future__ import annotations

import logging
import time
import uuid

from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import SessionContext, TaskRequest, TaskResult
from polar.rollout.pipeline import Pipeline

logger = logging.getLogger(__name__)


class RolloutManager:
    """Manage the lifecycle of rollout sessions for a single submitted task."""

    def __init__(
        self,
        *,
        pipeline: Pipeline,
        scheduler: NodeScheduler,
    ) -> None:
        self.pipeline = pipeline
        self.scheduler = scheduler

    async def run_task(self, request: TaskRequest) -> TaskResult:
        """Run one training task and return its result without retaining it."""
        sessions = [
            SessionContext(
                session_id=f"sk-polar-{uuid.uuid4()}",
                task_id=request.task_id,
                request=request,
                deadline_monotonic=time.monotonic() + request.timeout_seconds,
            )
            for _ in range(request.num_samples)
        ]

        ordered_results = list(await self.pipeline.run_batch(sessions))
        logger.info("Task %s completed with %d results", request.task_id, len(ordered_results))

        return TaskResult(
            task_id=request.task_id,
            instruction=request.instruction,
            results=ordered_results,
        )

    def status(self) -> dict[str, object]:
        return {
            "pipeline": self.pipeline.status(),
            "nodes": self.scheduler.stats(),
        }
