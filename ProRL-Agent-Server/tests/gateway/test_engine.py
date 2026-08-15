from __future__ import annotations

import pytest

from polar.gateway.engine import SGLangEngine, VLLMEngine, get_engine


def test_get_engine_returns_the_right_strategy() -> None:
    assert isinstance(get_engine("sglang"), SGLangEngine)
    assert isinstance(get_engine("vllm"), VLLMEngine)


def test_get_engine_rejects_unknown_name() -> None:
    with pytest.raises(ValueError, match="Unknown inference engine"):
        get_engine("tgi")


def test_sglang_engine_requests_source_token_extensions() -> None:
    engine = SGLangEngine()
    request = {"messages": []}
    out = engine.prepare_request(request)
    assert out is request and out["logprobs"] is True
    assert out["return_prompt_token_ids"] is True
    assert out["return_meta_info"] is True


def test_sglang_normalize_source_response_to_training_shape() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "prompt_token_ids": [1, 2],
                "logprobs": {
                    "content": [
                        {"token": "h", "logprob": -0.1},
                        {"token": "i", "logprob": -0.2},
                    ]
                },
                "meta_info": {
                    "output_token_logprobs": [
                        [-0.1, 10, "h"],
                        [-0.2, 11, "i"],
                    ]
                },
            }
        ]
    }

    out = SGLangEngine().normalize_response(response)
    choice = out["choices"][0]

    assert choice["input_token_ids"] == [1, 2]
    assert choice["token_ids"] == [10, 11]
    assert [entry["token_id"] for entry in choice["logprobs"]["content"]] == [10, 11]
    assert "meta_info" not in choice


def test_vllm_prepare_request_requests_token_ids_and_logprobs() -> None:
    out = VLLMEngine().prepare_request({"messages": [], "logprobs": True})
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True
    assert out["top_logprobs"] == 0


def test_vllm_prepare_request_keeps_explicit_top_logprobs() -> None:
    out = VLLMEngine().prepare_request({"logprobs": True, "top_logprobs": 5})
    assert out["top_logprobs"] == 5


def test_vllm_prepare_request_forces_logprobs_when_absent() -> None:
    out = VLLMEngine().prepare_request({"messages": []})
    assert out["logprobs"] is True
    assert out["return_token_ids"] is True
    assert out["top_logprobs"] == 0


def test_vllm_normalize_renames_reasoning_to_reasoning_content() -> None:
    response = {
        "choices": [
            {"message": {"role": "assistant", "content": "a", "reasoning": "because"}}
        ]
    }
    message = VLLMEngine().normalize_response(response)["choices"][0]["message"]
    assert message["reasoning_content"] == "because"
    assert "reasoning" not in message


def test_vllm_normalize_keeps_existing_reasoning_content() -> None:
    response = {"choices": [{"message": {"reasoning": "new", "reasoning_content": "kept"}}]}
    message = VLLMEngine().normalize_response(response)["choices"][0]["message"]
    assert message["reasoning_content"] == "kept"


def test_vllm_normalize_without_reasoning_is_noop() -> None:
    response = {"choices": [{"message": {"role": "assistant", "content": "hi"}}]}
    out = VLLMEngine().normalize_response(response)
    assert out["choices"][0]["message"] == {"role": "assistant", "content": "hi"}


def test_vllm_normalize_stamps_token_ids_onto_logprobs() -> None:
    response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "hi"},
                "token_ids": [10, 11],
                "logprobs": {
                    "content": [
                        {"token": "h", "logprob": -0.1},
                        {"token": "i", "logprob": -0.2},
                    ]
                },
            }
        ]
    }
    content = VLLMEngine().normalize_response(response)["choices"][0]["logprobs"]["content"]
    assert [entry["token_id"] for entry in content] == [10, 11]


def test_vllm_normalize_skips_token_id_stamp_on_length_mismatch() -> None:
    response = {
        "choices": [
            {
                "token_ids": [10, 11, 12],
                "logprobs": {"content": [{"token": "h", "logprob": -0.1}]},
            }
        ]
    }
    content = VLLMEngine().normalize_response(response)["choices"][0]["logprobs"]["content"]
    assert "token_id" not in content[0]
