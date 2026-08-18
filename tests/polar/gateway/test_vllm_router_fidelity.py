# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import ClassVar
from urllib.request import Request, urlopen

import pytest

_CHAT_RESPONSE = {
    "id": "chatcmpl-fidelity",
    "object": "chat.completion",
    "created": 1,
    "model": "policy",
    "prompt_token_ids": [7, 8],
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "OK"},
            "finish_reason": "stop",
            "token_ids": [42],
            "logprobs": {
                "content": [
                    {
                        "token": "OK",
                        "bytes": [79, 75],
                        "logprob": -0.25,
                        "top_logprobs": [],
                    }
                ]
            },
        }
    ],
    "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
}


class _FakeVllmWorker(BaseHTTPRequestHandler):
    requests: ClassVar[list[dict]] = []

    def _send_json(self, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._send_json(
                {
                    "object": "list",
                    "data": [{"id": "policy", "object": "model", "created": 1, "owned_by": "test"}],
                }
            )
        else:
            self._send_json({"status": "ok"})

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length))
        self.requests.append(payload)
        self._send_json(_CHAT_RESPONSE)

    def log_message(self, _format: str, *_args) -> None:
        pass


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _post_json(url: str, payload: dict) -> dict:
    request = Request(
        url,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-session-id": "fidelity-session"},
        method="POST",
    )
    with urlopen(request, timeout=10) as response:
        return json.load(response)


def _wait_for_router(process: subprocess.Popen, port: int, timeout: float = 30.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            output = process.stdout.read() if process.stdout is not None else ""
            pytest.fail(f"vllm-router exited during startup:\n{output}")
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.2):
                return
        except OSError:
            time.sleep(0.1)
    pytest.fail("vllm-router did not bind within 30 seconds")


@pytest.mark.integration
def test_openai_chat_route_preserves_training_fields() -> None:
    """Compare direct-worker and routed OpenAI responses without loading a model."""
    pytest.importorskip("vllm_router", reason="vllm-router is required for the fidelity test")

    _FakeVllmWorker.requests.clear()
    worker = ThreadingHTTPServer(("127.0.0.1", 0), _FakeVllmWorker)
    worker_thread = threading.Thread(target=worker.serve_forever, daemon=True)
    worker_thread.start()

    router_port = _free_port()
    router = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "vllm_router.launch_router",
            "--host",
            "127.0.0.1",
            "--port",
            str(router_port),
            "--prometheus-host",
            "127.0.0.1",
            "--prometheus-port",
            str(_free_port()),
            "--policy",
            "consistent_hash",
            "--worker-urls",
            f"http://127.0.0.1:{worker.server_port}",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )

    request = {
        "model": "policy",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 1,
        "stream": False,
        "logprobs": True,
        "top_logprobs": 0,
        "return_token_ids": True,
    }
    try:
        _wait_for_router(router, router_port)
        direct = _post_json(f"http://127.0.0.1:{worker.server_port}/v1/chat/completions", request)
        routed = _post_json(f"http://127.0.0.1:{router_port}/v1/chat/completions", request)

        assert _FakeVllmWorker.requests[-1]["return_token_ids"] is True
        assert routed["prompt_token_ids"] == direct["prompt_token_ids"]
        assert routed["choices"][0]["token_ids"] == direct["choices"][0]["token_ids"]
        assert routed["choices"][0]["logprobs"] == direct["choices"][0]["logprobs"]
    finally:
        router.terminate()
        try:
            router.wait(timeout=10)
        except subprocess.TimeoutExpired:
            router.kill()
            router.wait()
        worker.shutdown()
        worker.server_close()
        worker_thread.join(timeout=10)
