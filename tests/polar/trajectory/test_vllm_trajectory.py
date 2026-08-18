from __future__ import annotations

import asyncio

import pytest

from polar.trajectory.builder.per_request import PerRequestBuilder
from polar.trajectory.builder.prefix_merging import PrefixMergingBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession

_EOT = 99


def _record(
    completion_id: str,
    prompt_ids: list[int],
    response_ids: list[int],
    logprobs: list[float] | None,
    *,
    content: str,
    prompt_messages: list[dict],
    reasoning: str | None = None,
) -> CompletionRecord:
    message = {"role": "assistant", "content": content}
    if reasoning is not None:
        message["reasoning_content"] = reasoning
    choice = {
        "token_ids": response_ids,
        "message": message,
        "finish_reason": "stop",
    }
    if logprobs is not None:
        choice["logprobs"] = {
            "content": [
                {"token_id": token_id, "logprob": logprob}
                for token_id, logprob in zip(response_ids, logprobs)
            ]
        }
    return CompletionRecord(
        completion_id=completion_id,
        request={"messages": prompt_messages},
        response={"prompt_token_ids": prompt_ids, "choices": [choice]},
    )


def test_per_request_builder_preserves_vllm_training_fields() -> None:
    record = _record(
        "c1",
        [1, 2, 3],
        [10, 11, 12],
        [-0.1, -0.2, -0.3],
        content="4",
        reasoning="thinking",
        prompt_messages=[{"role": "user", "content": "2+2?"}],
    )

    trajectory = asyncio.run(
        PerRequestBuilder().build(CompletionSession(session_id="s", completions=[record]))
    )
    trace = trajectory.traces[0]

    assert trace.prompt_ids == [1, 2, 3]
    assert trace.response_ids == [10, 11, 12]
    assert trace.loss_mask == [1, 1, 1]
    assert trace.response_logprobs == [-0.1, -0.2, -0.3]
    assert trace.response_messages[0]["reasoning_content"] == "thinking"


def test_prefix_merging_preserves_interstitial_tokens_and_logprobs() -> None:
    user = {"role": "user", "content": "Q1"}
    assistant = {"role": "assistant", "content": "A1"}
    tool = {"role": "tool", "content": "result"}
    records = [
        _record(
            "c1",
            [1, 2, 3],
            [10, 11, _EOT],
            [-0.1, -0.2, -0.3],
            content="A1",
            prompt_messages=[user],
        ),
        _record(
            "c2",
            [1, 2, 3, 10, 11, _EOT, 50, 51],
            [20, 21, _EOT],
            [-0.5, -0.6, -0.7],
            content="A2",
            prompt_messages=[user, assistant, tool],
        ),
    ]

    trajectory = asyncio.run(
        PrefixMergingBuilder(end_of_turn_token_id=_EOT).build(
            CompletionSession(session_id="s", completions=records)
        )
    )
    trace = trajectory.traces[0]

    assert trace.response_ids == [10, 11, _EOT, 50, 51, 20, 21, _EOT]
    assert trace.loss_mask == [1, 1, 1, 0, 0, 1, 1, 1]
    assert trace.response_logprobs == [-0.1, -0.2, -0.3, 0.0, 0.0, -0.5, -0.6, -0.7]


def test_prefix_merging_rejects_incomplete_trainable_logprobs() -> None:
    user = {"role": "user", "content": "Q1"}
    records = [
        _record(
            "c1",
            [1, 2, 3],
            [10, 11, _EOT],
            [-0.1, -0.2, -0.3],
            content="A1",
            prompt_messages=[user],
        ),
        _record(
            "c2",
            [1, 2, 3, 10, 11, _EOT, 50, 51],
            [20, 21, _EOT],
            None,
            content="A2",
            prompt_messages=[
                user,
                {"role": "assistant", "content": "A1"},
                {"role": "tool", "content": "result"},
            ],
        ),
    ]

    with pytest.raises(ValueError, match="trainable response tokens require aligned response_logprobs"):
        asyncio.run(
            PrefixMergingBuilder(end_of_turn_token_id=_EOT).build(
                CompletionSession(session_id="s", completions=records)
            )
        )
