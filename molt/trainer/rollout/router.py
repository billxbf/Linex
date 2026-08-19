# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The vLLM router owned by the Molt training job."""

import socket
import time

import ray


@ray.remote(num_cpus=1)
class VllmRouterActor:
    """Run and supervise the vllm-router process for Molt's engine servers."""

    @staticmethod
    def _free_port(host: str) -> int:
        with socket.socket() as sock:
            sock.bind((host, 0))
            return sock.getsockname()[1]

    def __init__(self, worker_urls, *, policy="consistent_hash", port=None, max_payload_mb=512):
        import subprocess
        import sys

        self._host = ray.util.get_node_ip_address()
        self._port = port or self._free_port(self._host)
        self._proc = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_router.launch_router",
                "--host",
                self._host,
                "--port",
                str(self._port),
                "--prometheus-host",
                self._host,
                "--prometheus-port",
                str(self._free_port(self._host)),
                "--policy",
                policy,
                "--max-payload-size",
                str(max_payload_mb * 1024 * 1024),
                "--worker-urls",
                *[str(url) for url in worker_urls],
            ]
        )

    def url(self):
        return f"http://{self._host}:{self._port}"

    def ready(self, timeout_s=180.0):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            return_code = self._proc.poll()
            if return_code is not None:
                raise RuntimeError(
                    f"vLLM router subprocess exited ({return_code}) before binding {self._host}:{self._port}"
                )
            with socket.socket() as sock:
                sock.settimeout(1.0)
                if sock.connect_ex((self._host, self._port)) == 0:
                    return self.url()
            time.sleep(2.0)
        raise RuntimeError(f"vLLM router did not come up at {self.url()}")

    def close(self):
        import subprocess

        if self._proc.poll() is not None:
            return
        self._proc.terminate()
        try:
            self._proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            self._proc.kill()
            self._proc.wait()


def create_vllm_router(
    engines,
    *,
    policy="consistent_hash",
    port=None,
    tool_call_parser=None,
    reasoning_parser=None,
):
    """Serve each engine's OpenAI API and return the ready router actor and URL."""
    engine_urls = ray.get(
        [
            engine.serve_openai.remote(
                tool_call_parser=tool_call_parser,
                reasoning_parser=reasoning_parser,
            )
            for engine in engines
        ]
    )
    router = VllmRouterActor.remote(engine_urls, policy=policy, port=port)
    try:
        return router, ray.get(router.ready.remote())
    except Exception:
        ray.get(router.close.remote())
        raise
