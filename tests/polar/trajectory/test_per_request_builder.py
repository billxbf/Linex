from __future__ import annotations

import asyncio

import pytest

from polar.trajectory.builder.per_request import PerRequestBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession


def test_per_request_builder_returns_error_for_empty_session() -> None:
    session = CompletionSession(
        session_id="session-1",
        metadata={"group_id": "g1"},
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.status == "ERROR"
    assert trajectory.error == "no completions"
    assert trajectory.metadata["builder"] == "per_request"
    assert trajectory.metadata["group_id"] == "g1"
    assert trajectory.traces == []


def test_per_request_builder_emits_one_trace_per_completion() -> None:
    session = CompletionSession(
        session_id="session-1",
        task_id="task-1",
        model_requested="requested",
        model_used="served",
        api_type="openai_chat",
        completions=[
            CompletionRecord(
                completion_id="completion-1",
                request={
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "tools": [{"type": "function", "function": {"name": "lookup"}}],
                },
                response={
                    "prompt_token_ids": [1, 2],
                    "choices": [
                        {
                            "token_ids": [3, 4],
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token_id": 3, "logprob": -0.1},
                                    {"token_id": 4, "logprob": -0.2},
                                ]
                            },
                        }
                    ]
                },
                metadata={"completion_metadata": True},
            )
        ],
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.status == "COMPLETED"
    assert trajectory.metadata["record_count"] == 1
    assert trajectory.metadata["trace_count"] == 1
    trace = trajectory.traces[0]
    assert trace.prompt_ids == [1, 2]
    assert trace.response_ids == [3, 4]
    assert trace.loss_mask == [1, 1]
    assert trace.prompt_messages == [{"role": "user", "content": "Say hi"}]
    assert trace.response_messages == [{"role": "assistant", "content": "Hi"}]
    assert trace.tools == [{"type": "function", "function": {"name": "lookup"}}]
    assert trace.response_logprobs == [-0.1, -0.2]


def test_per_request_builder_rejects_unaligned_token_sources() -> None:
    session = CompletionSession(
        session_id="session-1",
        completions=[
            CompletionRecord(
                completion_id="completion-1",
                request={"messages": [{"role": "user", "content": "Say hi"}]},
                response={
                    "prompt_token_ids": [1, 2],
                    "choices": [
                        {
                            "token_ids": [3, 4, 5],
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token_id": 3, "logprob": -0.1},
                                    {"token_id": 4, "logprob": -0.2},
                                ]
                            },
                        }
                    ]
                },
            )
        ],
    )

    with pytest.raises(ValueError, match="trainable response tokens require aligned response_logprobs"):
        asyncio.run(PerRequestBuilder().build(session))
