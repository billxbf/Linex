"""Ray-owned Polar services used by the Molt training job."""

from __future__ import annotations

import asyncio
import socket
import time

import httpx
import ray

from polar.config import TopologyConfig
from polar.rollout.models import TaskRequest


@ray.remote(num_cpus=1, max_concurrency=1000)
class PolarServiceActor:
    """Host one rollout server or gateway in its own Ray process."""

    def __init__(self, service: str, node_id: str | None = None):
        if service not in {"rollout", "gateway"}:
            raise ValueError(f"Unknown Polar service: {service}")
        if service == "gateway" and not node_id:
            raise ValueError("Polar gateways require a node id")
        self.service = service
        self.node_id = node_id
        self.host = ray.util.get_node_ip_address()
        with socket.socket() as sock:
            sock.bind((self.host, 0))
            self.port = sock.getsockname()[1]
        self.url = f"http://{self.host}:{self.port}"
        self._server = None
        self._server_task = None
        self._client = None

    def descriptor(self) -> dict[str, object]:
        return {
            "service": self.service,
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "url": self.url,
        }

    async def start(self, topology_payload: dict) -> str:
        import uvicorn

        topology = TopologyConfig.model_validate(topology_payload)
        if self.service == "rollout":
            from polar.rollout.server import app, configure_server

            configure_server(topology)
        else:
            from polar.gateway.server import app, configure_server

            configure_server(topology, node_id=self.node_id)

        self._server = uvicorn.Server(uvicorn.Config(app, host=self.host, port=self.port, log_level="info"))
        self._server_task = asyncio.create_task(self._server.serve())
        deadline = time.monotonic() + 180.0
        while not self._server.started:
            if self._server_task.done():
                await self._server_task
                raise RuntimeError(f"Polar {self.service} stopped during startup")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Polar {self.service} did not become ready")
            await asyncio.sleep(0.1)

        self._client = httpx.AsyncClient(base_url=self.url, timeout=30.0)
        response = await self._client.get("/health")
        response.raise_for_status()
        health = response.json()
        if self.service == "gateway" and health.get("inference", {}).get("status") == "error":
            raise RuntimeError(f"Polar gateway cannot reach the Molt router: {health['inference']['error']}")
        return self.url

    async def ready(self, expected_nodes: int = 0) -> str:
        if self._client is None:
            raise RuntimeError(f"Polar {self.service} has not started")
        deadline = time.monotonic() + 180.0
        while True:
            response = await self._client.get("/health")
            response.raise_for_status()
            if self.service != "rollout" or response.json().get("nodes", 0) >= expected_nodes:
                return self.url
            if self._server_task.done():
                await self._server_task
                raise RuntimeError("Polar rollout server stopped before gateway registration")
            if time.monotonic() >= deadline:
                raise RuntimeError(f"Polar registered {response.json().get('nodes', 0)} of {expected_nodes} gateways")
            await asyncio.sleep(1.0)

    async def run_task(self, payload: dict) -> dict:
        if self.service != "rollout" or self._client is None:
            raise RuntimeError("run_task is only available on a started rollout server")
        request = TaskRequest.model_validate(payload)
        response = await self._client.post("/rollout/task/submit", json=request.model_dump(mode="json"))
        response.raise_for_status()
        while True:
            response = await self._client.get(f"/rollout/task/{request.task_id}")
            response.raise_for_status()
            result = response.json()
            if result["status"] != "running":
                return result
            await asyncio.sleep(0.2)

    async def pause(self, timeout_seconds: float = 300.0) -> dict:
        if self.service != "gateway":
            raise RuntimeError("pause is only available on gateways")
        from polar.gateway.server import get_state

        return await get_state().inference.pause_generation(timeout_seconds=timeout_seconds)

    async def resume(self) -> dict:
        if self.service != "gateway":
            raise RuntimeError("resume is only available on gateways")
        from polar.gateway.server import get_state

        return await get_state().inference.resume_generation()

    async def close(self) -> None:
        if self._server is not None:
            self._server.should_exit = True
        if self._server_task is not None:
            await asyncio.gather(self._server_task, return_exceptions=True)
        if self._client is not None:
            await self._client.aclose()
