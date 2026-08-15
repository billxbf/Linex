from __future__ import annotations

import asyncio

from polar.trajectory.builder.per_request import PerRequestBuilder
from polar.trajectory.models import CompletionRecord, CompletionSession


def test_per_request_builder_returns_error_for_empty_session() -> None:
    session = CompletionSession(
        session_id="session-1",
        metadata={"group_id": "g1", "policy_version": 7},
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))

    assert trajectory.status == "ERROR"
    assert trajectory.error == "no completions"
    assert trajectory.metadata["builder"] == "per_request"
    assert trajectory.metadata["group_id"] == "g1"
    assert trajectory.metadata["policy_version"] == 7
    assert trajectory.traces == []


def test_per_request_builder_emits_one_trace_per_completion() -> None:
    session = CompletionSession(
        session_id="session-1",
        task_id="task-1",
        model_requested="requested",
        model_used="served",
        api_type="openai_chat",
        metadata={"rollout_step": 3},
        completions=[
            CompletionRecord(
                completion_id="completion-1",
                request={
                    "messages": [{"role": "user", "content": "Say hi"}],
                    "tools": [{"type": "function", "function": {"name": "lookup"}}],
                },
                response={
                    "choices": [
                        {
                            "input_token_ids": [1, 2],
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
    assert trajectory.metadata["rollout_step"] == 3
    trace = trajectory.traces[0]
    assert trace.prompt_ids == [1, 2]
    assert trace.response_ids == [3, 4]
    assert trace.loss_mask == [1, 1]
    assert trace.prompt_messages == [{"role": "user", "content": "Say hi"}]
    assert trace.response_messages == [{"role": "assistant", "content": "Hi"}]
    assert trace.tools == [{"type": "function", "function": {"name": "lookup"}}]
    assert trace.response_logprobs == [-0.1, -0.2]


def test_per_request_builder_accepts_raw_source_sglang_token_fields() -> None:
    session = CompletionSession(
        session_id="session-1",
        completions=[
            CompletionRecord(
                completion_id="completion-1",
                request={"messages": [{"role": "user", "content": "Say hi"}]},
                response={
                    "choices": [
                        {
                            "prompt_token_ids": [1, 2],
                            "message": {"role": "assistant", "content": "Hi"},
                            "finish_reason": "stop",
                            "logprobs": {
                                "content": [
                                    {"token": "H", "logprob": -0.1},
                                    {"token": "i", "logprob": -0.2},
                                ]
                            },
                            "meta_info": {
                                "output_token_logprobs": [
                                    [-0.1, 3, "H"],
                                    [-0.2, 4, "i"],
                                ]
                            },
                        }
                    ]
                },
            )
        ],
    )

    trajectory = asyncio.run(PerRequestBuilder().build(session))
    trace = trajectory.traces[0]

    assert trace.prompt_ids == [1, 2]
    assert trace.response_ids == [3, 4]
    assert trace.response_logprobs == [-0.1, -0.2]


def test_per_request_builder_does_not_mix_unaligned_token_sources() -> None:
    session = CompletionSession(
        session_id="session-1",
        completions=[
            CompletionRecord(
                completion_id="completion-1",
                request={"messages": [{"role": "user", "content": "Say hi"}]},
                response={
                    "choices": [
                        {
                            "input_token_ids": [1, 2],
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

    trajectory = asyncio.run(PerRequestBuilder().build(session))
    trace = trajectory.traces[0]

    assert trace.response_ids == [3, 4, 5]
    assert trace.response_logprobs is None
