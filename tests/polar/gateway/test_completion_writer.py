"""Tests for the gateway CompletionWriter."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from polar.gateway.completion_writer import CompletionWriter, _truncate_value


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
