from __future__ import annotations

from slime_bridge._messages import flatten_content, messages_to_text, prompt_to_instruction_text


def test_flatten_content_handles_openai_content_shapes() -> None:
    assert flatten_content(None) == ""
    assert flatten_content("hello") == "hello"
    assert (
        flatten_content(
            [
                {"type": "text", "text": "hello"},
                {"text": " world"},
                "ignored",
            ]
        )
        == "hello world"
    )
    assert flatten_content(123) == "123"


def test_prompt_to_instruction_text_preserves_single_role_prompt_content() -> None:
    assert (
        prompt_to_instruction_text(
            [
                {"role": "user", "content": [{"type": "text", "text": "Fix it"}]},
                {"role": "user", "content": "Then test it"},
            ]
        )
        == "Fix it\n\nThen test it"
    )


def test_prompt_to_instruction_text_labels_multi_role_prompts() -> None:
    assert (
        prompt_to_instruction_text(
            [
                {"role": "system", "content": "Be precise"},
                {"role": "user", "content": "Fix it"},
            ]
        )
        == "[system] Be precise\n\n[user] Fix it"
    )


def test_messages_to_text_renders_role_blocks() -> None:
    assert (
        messages_to_text(
            [
                {"role": "assistant", "content": "Done"},
                {"role": "tool", "content": [{"type": "text", "text": "ok"}]},
            ]
        )
        == "[assistant] Done\n\n[tool] ok"
    )
