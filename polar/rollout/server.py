"""FastAPI server for rollout orchestration."""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from dataclasses import dataclass

from fastapi import FastAPI, HTTPException

from polar.config import RolloutServiceConfig, TopologyConfig
from polar.rollout.balancer import NodeScheduler
from polar.rollout.manager import RolloutManager
from polar.rollout.models import (
    GatewayNodeInfo,
    NodeHeartbeatRequest,
    NodeRegistrationRequest,
    SessionResult,
    TaskRequest,
    TaskStatus,
)
from polar.rollout.pipeline import Pipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


@dataclass(slots=True)
class RolloutState:
    topology: TopologyConfig
    rollout: RolloutServiceConfig
    scheduler: NodeScheduler
    pipeline: Pipeline
    manager: RolloutManager


_state: RolloutState | None = None
_configured_topology_path: str | None = None


def configure_server(topology_path: str = "topology.yaml") -> None:
    global _configured_topology_path, _state
    _configured_topology_path = topology_path
    _state = None


def _build_state(topology: TopologyConfig) -> RolloutState:
    rollout = topology.rollout
    scheduler = NodeScheduler(bootstrap_nodes=topology.bootstrap_nodes)
    pipeline = Pipeline(
        callback_url=f"{rollout.public_url}/callbacks/session_result",
        save_dir=rollout.save_dir,
        scheduler=scheduler,
        dispatch_poll_interval_seconds=rollout.dispatch_poll_interval_seconds,
        callback_grace_seconds=rollout.callback_grace_seconds,
    )
    manager = RolloutManager(pipeline=pipeline, scheduler=scheduler)
    return RolloutState(
        topology=topology,
        rollout=rollout,
        scheduler=scheduler,
        pipeline=pipeline,
        manager=manager,
    )


def get_state() -> RolloutState:
    global _state
    if _state is None:
        topology_path = _configured_topology_path or os.environ.get(
            "POLAR_TOPOLOGY",
            "topology.yaml",
        )
        _state = _build_state(TopologyConfig.load(topology_path))
    return _state


@asynccontextmanager
async def _lifespan(_: FastAPI):
    state = get_state()
    await state.pipeline.start()
    try:
        yield
    finally:
        await state.pipeline.close()


app = FastAPI(title="Polar Rollout", version="0.1.0", lifespan=_lifespan)


@app.get("/health")
async def health():
    state = get_state()
    return {"status": "ok", "nodes": len(state.scheduler.list_nodes())}


@app.post("/rollout/task/submit")
async def submit_task_async(request: TaskRequest):
    """Non-blocking task submission. Returns immediately with task_id.

    Poll ``GET /rollout/task/{task_id}`` until status becomes terminal.
    """
    state = get_state()
    try:
        task_id = await state.manager.submit_task(request)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    return {"task_id": task_id, "status": "running"}


@app.get("/rollout/task/{task_id}", response_model=TaskStatus)
async def get_task(task_id: str):
    task = get_state().manager.get_task(task_id)
    if task is None:
        raise HTTPException(status_code=404, detail="Task not found")
    return task


@app.get("/rollout/status")
async def rollout_status():
    return get_state().manager.status()


@app.post("/nodes/register", response_model=GatewayNodeInfo)
async def register_node(request: NodeRegistrationRequest):
    return get_state().scheduler.register_node(request)


@app.post("/nodes/{node_id}/heartbeat", response_model=GatewayNodeInfo)
async def node_heartbeat(node_id: str, request: NodeHeartbeatRequest):
    try:
        return get_state().scheduler.heartbeat(node_id, metrics=request.metrics)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/nodes", response_model=list[GatewayNodeInfo])
async def list_nodes():
    return get_state().scheduler.list_nodes()


@app.get("/nodes/{node_id}", response_model=GatewayNodeInfo)
async def get_node(node_id: str):
    node = get_state().scheduler.get_node(node_id)
    if node is None:
        raise HTTPException(status_code=404, detail="Node not found")
    return node


@app.delete("/nodes/{node_id}", response_model=GatewayNodeInfo)
async def drain_node(node_id: str):
    try:
        return get_state().scheduler.drain_node(node_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/callbacks/session_result")
async def session_result_callback(result: SessionResult):
    await get_state().pipeline.accept_callback_result(result)
    return {"status": "accepted"}


def serve(topology_path: str = "topology.yaml", *, log_level: str = "info") -> None:
    import uvicorn

    configure_server(topology_path)
    state = get_state()
    uvicorn.run(
        app,
        host=state.rollout.host,
        port=state.rollout.port,
        log_level=log_level,
    )


def main() -> None:
    serve(os.environ.get("POLAR_TOPOLOGY", "topology.yaml"))


if __name__ == "__main__":
    main()
