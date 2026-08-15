"""Helpers for converting completion records into trajectory traces."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from polar.trajectory.models import CompletionRecord, Trace


def _coerce_int_list(value: Any) -> list[int] | None:
    if not isinstance(value, list):
        return None
    extracted: list[int] = []
    for item in value:
        try:
            extracted.append(int(item))
        except (TypeError, ValueError):
            return None
    return extracted


def _paired_tokens_from_logprobs_content(
    choice: dict[str, Any],
) -> tuple[list[int], list[float]] | None:
    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        return None
    content = logprobs.get("content")
    if not isinstance(content, list) or not content:
        return None

    token_ids: list[int] = []
    token_logprobs: list[float] = []
    for item in content:
        if not isinstance(item, dict):
            return None
        token_id = item.get("token_id")
        logprob = item.get("logprob")
        if token_id is None or logprob is None:
            return None
        try:
            token_ids.append(int(token_id))
            token_logprobs.append(float(logprob))
        except (TypeError, ValueError):
            return None
    return token_ids, token_logprobs


def _extract_response_tokens(
    response: dict[str, Any],
    choice: dict[str, Any],
) -> tuple[list[int], list[float] | None]:
    """Extract response ids and logprobs without mixing unaligned sources."""
    token_ids = choice.get("token_ids", response.get("token_ids"))
    response_ids = _coerce_int_list(token_ids)
    paired = _paired_tokens_from_logprobs_content(choice)

    if response_ids is not None:
        if paired is not None:
            paired_ids, paired_logprobs = paired
            if paired_ids == response_ids:
                return response_ids, paired_logprobs
        return response_ids, None

    return paired if paired is not None else ([], None)


def _extract_prompt_messages(request: dict[str, Any]) -> list[dict[str, Any]]:
    messages = request.get("messages")
    if not isinstance(messages, list):
        return []
    return [deepcopy(message) for message in messages if isinstance(message, dict)]


def _extract_tools(request: dict[str, Any]) -> list[dict[str, Any]] | None:
    tools = request.get("tools")
    if not isinstance(tools, list) or not tools:
        return None
    extracted = [deepcopy(tool) for tool in tools if isinstance(tool, dict)]
    return extracted or None


def build_trace_from_completion(completion: CompletionRecord) -> Trace:
    """Normalize one stored completion record into a trajectory trace."""

    request = completion.request if isinstance(completion.request, dict) else {}
    response = completion.response if isinstance(completion.response, dict) else {}
    choices = response.get("choices")
    first_choice = (
        choices[0]
        if isinstance(choices, list) and choices and isinstance(choices[0], dict)
        else {}
    )
    prompt_ids = response.get("prompt_token_ids")
    response_message = first_choice.get("message")
    finish_reason = first_choice.get("finish_reason")

    response_ids, response_logprobs = _extract_response_tokens(response, first_choice)

    return Trace(
        prompt_ids=list(prompt_ids) if isinstance(prompt_ids, list) else [],
        response_ids=response_ids,
        loss_mask=[1] * len(response_ids),
        prompt_messages=_extract_prompt_messages(request),
        response_messages=[deepcopy(response_message)] if isinstance(response_message, dict) else [],
        tools=_extract_tools(request),
        finish_reason=str(finish_reason) if finish_reason is not None else None,
        response_logprobs=response_logprobs,
        metadata=deepcopy(completion.metadata),
    )
