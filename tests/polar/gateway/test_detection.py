from __future__ import annotations

from polar.gateway.detection import APIType, detect, extract_model


def test_detect_prefers_path_over_body_shape() -> None:
    assert (
        detect(
            "/v1/messages",
            headers={},
            body={"contents": [{"parts": [{"text": "hello"}]}]},
        )
        == APIType.ANTHROPIC
    )
    assert detect("/v1/chat/completions", {}, {"input": "x", "instructions": "y"}) == APIType.OPENAI_CHAT
    assert detect("/v1/responses", {}, {"messages": []}) == APIType.OPENAI_RESPONSES
    assert detect("/v1beta/models/gemini:generateContent", {}, {}) == APIType.GOOGLE


def test_detect_falls_back_to_headers_and_body_shape() -> None:
    assert detect("/unknown", {"Anthropic-Version": "2023-06-01"}, {}) == APIType.ANTHROPIC
    assert detect("/unknown", {}, {"contents": []}) == APIType.GOOGLE
    assert detect("/unknown", {}, {"input": "x", "instructions": "y"}) == APIType.OPENAI_RESPONSES
    assert detect("/unknown", {}, {}) == APIType.OPENAI_CHAT


def test_extract_model_defaults_by_api_family() -> None:
    assert extract_model(APIType.OPENAI_CHAT, {"model": "requested-model"}) == "requested-model"
    assert extract_model(APIType.GOOGLE, {}) == "gemini-pro"
    assert extract_model(APIType.ANTHROPIC, {}) == "unknown"
