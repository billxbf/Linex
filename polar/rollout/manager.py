"""Top-level task orchestration for rollout batches."""

from __future__ import annotations

import asyncio
import logging
import threading
import time
import uuid
from dataclasses import dataclass, field

import httpx

from polar.rollout.balancer import NodeScheduler
from polar.rollout.models import (
    SessionContext,
    SessionResult,
    TaskRequest,
    TaskResult,
    TaskStatus,
)
from polar.rollout.pipeline import Pipeline

logger = logging.getLogger(__name__)

_CALLBACK_TIMEOUT_SECONDS = 10.0


@dataclass(slots=True)
class _TaskRecord:
    task_id: str
    status: str
    total_sessions: int
    completed_sessions: int = 0
    results: list[SessionResult] = field(default_factory=list)
    result_paths: list[str] = field(default_factory=list)


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
        self._tasks: dict[str, _TaskRecord] = {}
        self._lock = threading.RLock()

    async def submit_task(self, request: TaskRequest) -> str:
        """Register a task and run it in the background. Returns task_id immediately."""
        with self._lock:
            existing = self._tasks.get(request.task_id)
            if existing is not None and existing.status == "running":
                raise ValueError(f"task {request.task_id} is already running")
            self._tasks[request.task_id] = _TaskRecord(
                task_id=request.task_id,
                status="running",
                total_sessions=request.num_samples,
            )
        asyncio.create_task(self._run_task_background(request))
        return request.task_id

    async def _run_task_background(self, request: TaskRequest) -> None:
        """Execute a task in the background, updating the record on completion."""
        try:
            result = await self._execute_task(request)
            logger.info("Task %s completed with %d results", request.task_id, len(result.results))
        except Exception:
            logger.exception("Background task %s failed", request.task_id)
            return
        if request.callback_url:
            await self._post_callback(request.callback_url, result)

    async def _post_callback(self, callback_url: str, result: TaskResult) -> None:
        """Best-effort POST the terminal TaskResult to the trainer's callback URL."""
        try:
            async with httpx.AsyncClient(timeout=_CALLBACK_TIMEOUT_SECONDS) as client:
                response = await client.post(
                    callback_url,
                    json=result.model_dump(mode="json"),
                )
                response.raise_for_status()
        except Exception:
            logger.warning(
                "Callback POST to %s failed for task %s; trainer must fall back to polling",
                callback_url,
                result.task_id,
                exc_info=True,
            )

    async def _execute_task(self, request: TaskRequest) -> TaskResult:
        sessions = [
            SessionContext(
                session_id=f"sk-polar-{uuid.uuid4()}",
                task_id=request.task_id,
                request=request,
                deadline_monotonic=time.monotonic() + request.timeout_seconds,
            )
            for _ in range(request.num_samples)
        ]

        async def _on_result(result: SessionResult) -> None:
            result_path = self.pipeline.result_path_for(result.task_id, result.session_id)
            with self._lock:
                record = self._tasks[request.task_id]
                record.completed_sessions += 1
                record.results.append(result)
                if result_path is not None:
                    record.result_paths.append(result_path)

        try:
            results = await self.pipeline.run_batch(sessions, on_result=_on_result)
        except Exception:
            with self._lock:
                self._tasks[request.task_id].status = "failed"
            raise

        ordered_results = list(results)
        with self._lock:
            record = self._tasks[request.task_id]
            record.status = "completed"
            record.completed_sessions = len(ordered_results)
            record.results = ordered_results
            result_paths = list(record.result_paths)

        return TaskResult(
            task_id=request.task_id,
            status="completed",
            results=ordered_results,
            result_paths=result_paths,
        )

    def get_task(self, task_id: str) -> TaskStatus | None:
        with self._lock:
            record = self._tasks.get(task_id)
            if record is None:
                return None
            return TaskStatus(
                task_id=record.task_id,
                status=record.status,
                total_sessions=record.total_sessions,
                completed_sessions=record.completed_sessions,
                results=list(record.results),
                result_paths=list(record.result_paths),
            )

    def status(self) -> dict[str, object]:
        with self._lock:
            task_statuses = {
                task_id: record.status
                for task_id, record in self._tasks.items()
            }
        return {
            "tasks": task_statuses,
            "pipeline": self.pipeline.status(),
            "nodes": self.scheduler.stats(),
        }
