"""Tests for the gateway CompletionWriter."""

from __future__ import annotations

import asyncio
import base64
import io
import json
from pathlib import Path

from PIL import Image

from polar.gateway.completion_writer import CompletionWriter, _truncate_value
from polar.gateway.session import SessionRegistry
from polar.gateway.storage import SessionStore
from polar.rollout.models import SessionResult
from polar.trajectory.models import Trajectory


def test_truncate_value_string() -> None:
    long = "a" * 100
    truncated = _truncate_value(long, max_bytes=20)
    assert isinstance(truncated, str)
    assert len(truncated.encode("utf-8")) <= 24  # plus ellipsis


def test_truncate_value_under_budget() -> None:
    short = {"foo": "bar"}
    assert _truncate_value(short, max_bytes=1024) == short


def test_writer_persists_records(tmp_path: Path) -> None:
    async def run() -> None:
        writer = CompletionWriter(save_dir=tmp_path, queue_size=8)
        await writer.start()
        for i in range(3):
            writer.enqueue(
                task_id="t1",
                session_id="sess1",
                completion_id=f"id{i}",
                record={"completion_id": f"id{i}", "payload": {"i": i}},
            )
        await asyncio.sleep(0.2)
        await writer.close()

    asyncio.run(run())

    out_dir = tmp_path / "task_t1" / "sessions" / "sess1" / "completions"
    files = sorted(out_dir.glob("*.json"))
    assert len(files) == 3
    first = json.loads(files[0].read_text())
    assert first["payload"]["i"] == 0


def test_writer_disabled_when_no_save_dir() -> None:
    async def run() -> bool:
        writer = CompletionWriter(save_dir=None, enabled=True)
        await writer.start()
        ok = writer.enqueue(task_id="t", session_id="s", completion_id="c", record={})
        await writer.close()
        return ok

    assert asyncio.run(run()) is False


def test_writer_requires_task_id(tmp_path: Path) -> None:
    async def run() -> bool:
        writer = CompletionWriter(save_dir=tmp_path)
        await writer.start()
        ok = writer.enqueue(task_id=None, session_id="s", completion_id="c", record={})
        await writer.close()
        return ok

    assert asyncio.run(run()) is False


def test_session_store_moves_media_out_of_completion_payload(tmp_path: Path) -> None:
    image_buffer = io.BytesIO()
    Image.new("RGB", (2, 2), "yellow").save(image_buffer, format="PNG")
    image_url = "data:image/png;base64," + base64.b64encode(image_buffer.getvalue()).decode()
    response = {
        "prompt_token_ids": [1, 2],
        "choices": [
            {
                "token_ids": [3, 4],
                "logprobs": {"content": [{"token_id": 3, "logprob": -0.1}, {"token_id": 4, "logprob": -0.2}]},
            }
        ],
    }
    store = SessionStore(artifact_root=tmp_path)

    request = {"messages": [{"role": "user", "content": [{"type": "image_url", "image_url": {"url": image_url}}]}]}
    store.save_message(
        "session-1",
        request,
        response,
        task_id="task-1",
    )
    store.save_message("session-1", request, response, task_id="task-1")
    record, repeated = store.load_completion_session("session-1").completions

    assert len(record.media_paths) == 1 and Path(record.media_paths[0]).read_bytes() == image_buffer.getvalue()
    assert repeated.media_paths == record.media_paths
    assert len(list((tmp_path / "task_task-1" / "sessions" / "session-1" / "artifacts").glob("media-*"))) == 1
    assert record.request["messages"][0]["content"][0]["image_url"]["url"].startswith("file://")
    assert image_url not in record.model_dump_json()


def test_delivered_gateway_result_releases_trajectory_payload() -> None:
    registry = SessionRegistry()
    registry.register("session-1", task_id="task-1")
    registry.set_result(
        "session-1",
        SessionResult(
            session_id="session-1",
            task_id="task-1",
            status="COMPLETED",
            trajectory=Trajectory(status="COMPLETED"),
        ),
    )

    registry.clear_result_payload("session-1")

    info = registry.get("session-1")
    assert info.status == "COMPLETED"
    assert info.result is None
