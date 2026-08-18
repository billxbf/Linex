from __future__ import annotations

import asyncio
import json

import httpx
import pytest

from polar.gateway.proxy import InferenceClient, InferenceContractError

_SAMPLING = {
    "max_tokens": 8,
    "max_total_tokens": 10,
    "temperature": 0.2,
    "top_p": 1.0,
    "top_k": -1,
    "min_tokens": 1,
    "min_p": 0.0,
    "repetition_penalty": 1.0,
    "frequency_penalty": 0.0,
    "presence_penalty": 0.0,
    "seed": None,
    "skip_special_tokens": False,
    "ignore_eos": False,
    "include_stop_str_in_output": False,
    "logprobs": 1,
    "n": 1,
}


def test_completion_uses_one_direct_request_with_molt_sampling() -> None:
    forwarded: list[tuple[str, str | None, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        forwarded.append((request.url.path, request.headers.get("x-session-id"), payload))
        return httpx.Response(
            200,
            json={
                "prompt_token_ids": [1, 2, 3],
                "choices": [
                    {
                        "index": 0,
                        "token_ids": [4],
                        "message": {"role": "assistant", "content": "x", "reasoning": "why"},
                        "finish_reason": "stop",
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
                    "model": "policy",
                    "stream": True,
                    "max_tokens": 99,
                    "max_completion_tokens": 77,
                    "temperature": 0.9,
                    "top_p": 0.5,
                    "top_k": 20,
                    "skip_special_tokens": True,
                    "include_stop_str_in_output": True,
                    "logprobs": False,
                    "return_token_ids": False,
                    "top_logprobs": 5,
                    "seed": 999,
                    "messages": [{"role": "assistant", "content": "prior", "reasoning_content": "thought"}],
                },
                session_id="session-1",
                sampling_params=_SAMPLING,
            )
        finally:
            await client.close()

    response = asyncio.run(run())

    assert len(forwarded) == 1
    path, session_id, payload = forwarded[0]
    assert path == "/v1/chat/completions"
    assert session_id == "session-1"
    assert payload["stream"] is False
    assert payload["max_tokens"] == 8
    assert payload["temperature"] == 0.2
    assert payload["top_p"] == 1.0
    assert payload["top_k"] == -1
    assert payload["seed"] is None
    assert "max_total_tokens" not in payload
    assert "max_completion_tokens" not in payload
    assert payload["skip_special_tokens"] is True
    assert payload["include_stop_str_in_output"] is True
    assert payload["logprobs"] is True
    assert payload["return_token_ids"] is True
    assert payload["top_logprobs"] == 0
    assert payload["messages"][0]["reasoning"] == "thought"
    choice = response["choices"][0]
    assert response["prompt_token_ids"] == [1, 2, 3]
    assert choice["token_ids"] == [4]
    assert choice["message"]["reasoning_content"] == "why"
    assert choice["logprobs"]["content"][0]["token_id"] == 4


@pytest.mark.parametrize(
    ("response", "message"),
    [
        (
            {"choices": [{"token_ids": [4], "logprobs": {"content": [{"logprob": -0.1}]}}]},
            "prompt_token_ids",
        ),
        ({"prompt_token_ids": [1]}, r"choices\[0\]"),
        (
            {"prompt_token_ids": [1], "choices": [{"logprobs": {"content": [{"logprob": -0.1}]}}]},
            r"choices\[0\]\.token_ids",
        ),
        (
            {"prompt_token_ids": [1], "choices": [{"token_ids": [4]}]},
            r"choices\[0\]\.logprobs\.content",
        ),
        (
            {
                "prompt_token_ids": [1],
                "choices": [{"token_ids": [4, 5], "logprobs": {"content": [{"logprob": -0.1}]}}],
            },
            "2 token IDs and 1 log probabilities",
        ),
        (
            {
                "prompt_token_ids": [1],
                "choices": [{"token_ids": [4], "logprobs": {"content": [{"token": "x"}]}}],
            },
            "token without a log probability",
        ),
    ],
)
def test_completion_rejects_invalid_training_fields(response, message) -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=response)

    async def run() -> None:
        client = InferenceClient("http://router:9000")
        client._client = httpx.AsyncClient(
            base_url="http://router:9000",
            transport=httpx.MockTransport(handler),
        )
        try:
            await client.completion(
                {"model": "policy", "messages": [{"role": "user", "content": "x"}]},
                session_id="session-1",
                sampling_params=_SAMPLING,
            )
        finally:
            await client.close()

    with pytest.raises(InferenceContractError, match=message):
        asyncio.run(run())


def test_generation_status_has_no_backend_selector() -> None:
    client = InferenceClient("http://router:9000/")

    async def run() -> tuple[dict, dict]:
        paused = await client.pause_generation()
        resumed = await client.resume_generation()
        return paused, resumed

    paused, resumed = asyncio.run(run())

    assert paused == {"paused": True, "inflight": 0, "base_url": "http://router:9000"}
    assert resumed == {"paused": False, "inflight": 0, "base_url": "http://router:9000"}
