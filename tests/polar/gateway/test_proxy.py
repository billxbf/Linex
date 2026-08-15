from __future__ import annotations

import asyncio
import json

import httpx

from polar.gateway.proxy import InferenceClient


def test_completion_uses_vllm_training_fields_and_normalizes_response() -> None:
    forwarded: dict = {}

    def handler(request: httpx.Request) -> httpx.Response:
        forwarded.update(json.loads(request.content))
        assert request.url.path == "/v1/chat/completions"
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [1, 2, 3],
                "choices": [
                    {
                        "token_ids": [4],
                        "message": {"role": "assistant", "content": "x", "reasoning": "why"},
                        "logprobs": {"content": [{"token": "x", "logprob": -0.1}]},
                    }
                ],
            },
        )

    async def run() -> dict:
        client = InferenceClient("http://router:9000")
        client._client = httpx.AsyncClient(
            base_url="http://router:9000",
            transport=httpx.MockTransport(handler),
        )
        try:
            return await client.completion(
                {
                    "stream": True,
                    "messages": [
                        {"role": "assistant", "content": "prior", "reasoning_content": "thought"}
                    ],
                }
            )
        finally:
            await client.close()

    response = asyncio.run(run())

    assert forwarded["stream"] is False
    assert forwarded["logprobs"] is True
    assert forwarded["return_token_ids"] is True
    assert forwarded["top_logprobs"] == 0
    assert forwarded["messages"][0]["reasoning"] == "thought"
    choice = response["choices"][0]
    assert choice["message"]["reasoning_content"] == "why"
    assert choice["logprobs"]["content"][0]["token_id"] == 4


def test_generation_status_has_no_backend_selector() -> None:
    client = InferenceClient("http://router:9000/")

    async def run() -> tuple[dict, dict]:
        paused = await client.pause_generation()
        resumed = await client.resume_generation()
        return paused, resumed

    paused, resumed = asyncio.run(run())

    assert paused == {"paused": True, "inflight": 0, "base_url": "http://router:9000"}
    assert resumed == {"paused": False, "inflight": 0, "base_url": "http://router:9000"}
