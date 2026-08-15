from __future__ import annotations

import asyncio
import json

import httpx

from polar.gateway.engine import SGLangEngine
from polar.gateway.proxy import InferenceClient


def test_sglang_router_workers_are_used_for_direct_completion() -> None:
    requests: list[tuple[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append((request.url.host or "", request.url.path))
        if request.url.path == "/workers":
            return httpx.Response(
                200,
                json={
                    "workers": [
                        {"id": "w0", "url": "http://worker-a:15000", "worker_type": "regular"}
                    ]
                },
            )
        if request.url.host == "worker-a" and request.url.path == "/v1/chat/completions":
            body = json.loads(request.content)
            assert body["return_prompt_token_ids"] is True
            assert body["return_meta_info"] is True
            return httpx.Response(
                200,
                json={
                    "choices": [
                        {
                            "prompt_token_ids": [1, 2, 3],
                            "message": {"role": "assistant", "content": "x"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [{"token": "x", "logprob": -0.1, "bytes": [120]}]
                            },
                            "meta_info": {
                                "output_token_logprobs": [[-0.1, 4, "x"]]
                            },
                        }
                    ]
                },
            )
        return httpx.Response(500, json={"error": f"unexpected {request.url}"})

    async def run() -> dict:
        client = InferenceClient("http://router:9000", SGLangEngine())
        client._client = httpx.AsyncClient(
            base_url="http://router:9000",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.completion({"messages": [{"role": "user", "content": "hi"}]})
        finally:
            await client.close()

    response = asyncio.run(run())
    choice = response["choices"][0]

    assert ("router", "/workers") in requests
    assert ("worker-a", "/v1/chat/completions") in requests
    assert ("router", "/v1/chat/completions") not in requests
    assert choice["input_token_ids"] == [1, 2, 3]
    assert choice["token_ids"] == [4]
    assert choice["logprobs"]["content"][0]["token_id"] == 4
