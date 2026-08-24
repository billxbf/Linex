from __future__ import annotations

import json
from pathlib import Path

from examples.skill2env_sft.export_sft import export_sessions


def _session(
    *,
    status: str = "COMPLETED",
    builder: str = "per_request",
    traces: list[dict] | None = None,
    reward: float | None = 0.0,
) -> dict:
    if traces is None:
        traces = [
            {
                "prompt_ids": [],
                "response_ids": [],
                "loss_mask": [],
                "prompt_messages": [{"role": "user", "content": "task"}],
                "response_messages": [{"role": "assistant", "content": "answer"}],
                "finish_reason": "stop",
                "reward": reward,
            }
        ]
    return {
        "session_id": "session-1",
        "task_id": "task-1",
        "status": status,
        "trajectory": {
            "metadata": {
                "builder": builder,
                "evaluation": {"reward": reward},
            },
            "traces": traces,
        },
    }


def _write(root: Path, name: str, payload: dict | str) -> None:
    path = root / f"task_{name}" / f"ses_{name}.json"
    path.parent.mkdir(parents=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload))


def test_export_sessions_keeps_valid_per_request_traces_and_media_order(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    traces = _session()["trajectory"]["traces"] * 2
    traces[0] = {**traces[0], "media_paths": ["second.png", "first.png"]}
    _write(rollout_dir, "complete", _session(traces=traces, reward=0.0))
    _write(rollout_dir, "error", _session(status="ERROR"))
    _write(rollout_dir, "timeout", _session(status="TIMEOUT"))
    _write(rollout_dir, "empty", _session(traces=[]))
    _write(rollout_dir, "wrong", _session(builder="prefix_merging"))
    _write(rollout_dir, "broken", "{")
    _write(rollout_dir, "array", "[]")
    malformed_traces = _session()["trajectory"]["traces"] * 2
    malformed_traces[1] = {**malformed_traces[1]}
    malformed_traces[1]["response_messages"] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"function": {"name": "write", "arguments": "{"}}],
        }
    ]
    _write(rollout_dir, "malformed", _session(traces=malformed_traces))
    no_assistant = _session()
    no_assistant["trajectory"]["traces"][0]["response_messages"] = [{"role": "tool", "content": "output"}]
    _write(rollout_dir, "no_assistant", no_assistant)

    output = tmp_path / "teacher_sft.jsonl"
    summary = export_sessions(rollout_dir, output)
    rows = [json.loads(line) for line in output.read_text().splitlines()]

    assert summary["sessions_seen"] == 9
    assert summary["rows_written"] == 3
    assert summary["status_counts"] == {"COMPLETED": 5, "ERROR": 1, "MALFORMED": 2, "TIMEOUT": 1}
    assert summary["reward_counts"]["0.0"] == 7
    assert summary["skip_reasons"] == {
        "empty_trajectory": 1,
        "malformed_json": 1,
        "malformed_session": 1,
        "malformed_tool_calls": 1,
        "no_assistant_response": 1,
        "status_not_completed": 2,
        "wrong_builder": 1,
    }
    assert [row["trace_index"] for row in rows] == [0, 1, 0]
    assert all(row["reward"] == 0.0 for row in rows)
    assert rows[0]["images"] == ["second.png", "first.png"]
    assert "images" not in rows[1]
    assert "prompt_ids" not in rows[0] and "loss_mask" not in rows[0]


def test_rejection_sampling_keeps_only_positive_reward_traces(tmp_path: Path) -> None:
    rollout_dir = tmp_path / "rollouts"
    _write(rollout_dir, "pass", _session(reward=1.0))
    _write(rollout_dir, "fail", _session(reward=0.0))
    _write(rollout_dir, "missing", _session(reward=None))

    output = tmp_path / "teacher_sft.jsonl"

    kept_all = export_sessions(rollout_dir, output)
    assert kept_all["rows_written"] == 3
    assert "rejected_by_reward" not in kept_all["skip_reasons"]

    kept_positive = export_sessions(rollout_dir, output, rejection_sampling=True)
    assert kept_positive["rows_written"] == 1
    assert kept_positive["skip_reasons"]["rejected_by_reward"] == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[0]["reward"] == 1.0
