"""Export complete Polar sessions as Skill2Env SFT chat JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from pydantic import ValidationError

from polar.rollout.models import SessionResult, SessionStatus


def export_sessions(
    rollout_dir: Path, output_path: Path, *, rejection_sampling: bool = False
) -> dict[str, object]:
    """Write eligible persisted sessions and return export counts.

    ``rejection_sampling`` drops otherwise-healthy traces whose teacher session
    did not succeed (reward missing or <= 0). Off by default: every well-formed
    teacher trace is exported regardless of outcome, so the student imitates
    full teacher behavior rather than only its successes.
    """
    paths = sorted(rollout_dir.glob("task_*/ses_*.json"))
    if not paths:
        raise ValueError(f"No persisted session results found under {rollout_dir}")

    rows: list[dict[str, object]] = []
    statuses: Counter[str] = Counter()
    rewards: Counter[str] = Counter()
    skipped: Counter[str] = Counter()

    for path in paths:
        try:
            payload = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            statuses["MALFORMED"] += 1
            rewards["unknown"] += 1
            skipped["malformed_json"] += 1
            continue
        if not isinstance(payload, dict):
            statuses["MALFORMED"] += 1
            rewards["unknown"] += 1
            skipped["malformed_session"] += 1
            continue

        status = str(payload.get("status", "MISSING"))
        statuses[status] += 1
        trajectory_payload = payload.get("trajectory")
        trajectory_metadata = trajectory_payload.get("metadata") if isinstance(trajectory_payload, dict) else None
        evaluation = trajectory_metadata.get("evaluation", {}) if isinstance(trajectory_metadata, dict) else {}
        reward = evaluation.get("reward", evaluation.get("outcome_reward")) if isinstance(evaluation, dict) else None
        traces_payload = trajectory_payload.get("traces") if isinstance(trajectory_payload, dict) else None
        if (
            reward is None
            and isinstance(traces_payload, list)
            and traces_payload
            and isinstance(traces_payload[0], dict)
        ):
            reward = traces_payload[0].get("reward")
        rewards[json.dumps(reward, separators=(",", ":"))] += 1

        if status != SessionStatus.COMPLETED:
            skipped["status_not_completed"] += 1
            continue
        if not isinstance(trajectory_payload, dict):
            skipped["malformed_session"] += 1
            continue

        # Persistence avoids duplicating terminal status inside the trajectory.
        # Restore it only for validation against the canonical wire model.
        trajectory_payload["status"] = status
        trajectory_payload["error"] = payload.get("error")
        try:
            result = SessionResult.model_validate(payload)
        except ValidationError:
            skipped["malformed_session"] += 1
            continue

        metadata = result.trajectory.metadata
        if metadata.get("builder") != "per_request":
            skipped["wrong_builder"] += 1
            continue
        if not result.trajectory.traces:
            skipped["empty_trajectory"] += 1
            continue

        for trace_index, trace in enumerate(result.trajectory.traces):
            if not any(message.get("role") == "assistant" for message in trace.response_messages):
                skipped["no_assistant_response"] += 1
                continue

            invalid_reason = None
            for message in trace.prompt_messages + trace.response_messages:
                if message.get("role") not in {"system", "user", "assistant", "tool"}:
                    invalid_reason = "malformed_messages"
                    break
                content = message.get("content")
                if content is not None and not isinstance(content, (str, list)):
                    invalid_reason = "malformed_messages"
                    break
                if isinstance(content, list) and not all(isinstance(item, dict) for item in content):
                    invalid_reason = "malformed_messages"
                    break
                calls = message.get("tool_calls")
                if calls is None:
                    continue
                if not isinstance(calls, list):
                    invalid_reason = "malformed_tool_calls"
                    break
                for call in calls:
                    function = call.get("function") if isinstance(call, dict) else None
                    if not isinstance(function, dict) or not isinstance(function.get("name"), str):
                        invalid_reason = "malformed_tool_calls"
                        break
                    arguments = function.get("arguments")
                    try:
                        arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
                    except json.JSONDecodeError:
                        invalid_reason = "malformed_tool_calls"
                        break
                    if not isinstance(arguments, dict):
                        invalid_reason = "malformed_tool_calls"
                        break
                if invalid_reason:
                    break
            if invalid_reason:
                skipped[invalid_reason] += 1
                continue
            if rejection_sampling and not (trace.reward is not None and trace.reward > 0):
                skipped["rejected_by_reward"] += 1
                continue

            row: dict[str, object] = {
                "prompt_messages": trace.prompt_messages,
                "response_messages": trace.response_messages,
                "task_id": result.task_id,
                "session_id": result.session_id,
                "trace_index": trace_index,
                "session_status": result.status.value,
                "reward": trace.reward,
                "finish_reason": trace.finish_reason,
            }
            if trace.tools:
                row["tools"] = trace.tools
            if trace.media_paths:
                row["images"] = trace.media_paths
            rows.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp")
    with temporary.open("w") as output:
        for row in rows:
            output.write(json.dumps(row, separators=(",", ":")) + "\n")
    temporary.replace(output_path)

    return {
        "sessions_seen": len(paths),
        "rows_written": len(rows),
        "status_counts": dict(sorted(statuses.items())),
        "reward_counts": dict(sorted(rewards.items())),
        "skip_reasons": dict(sorted(skipped.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("rollout_dir", type=Path, help="Directory containing task_*/ses_*.json")
    parser.add_argument("output", type=Path, help="Destination JSONL path")
    parser.add_argument(
        "--rejection-sampling",
        action="store_true",
        help="Keep only traces whose teacher session reward > 0. Default keeps every "
        "well-formed trace regardless of outcome.",
    )
    args = parser.parse_args()

    try:
        summary = export_sessions(
            args.rollout_dir.expanduser().resolve(),
            args.output.expanduser().resolve(),
            rejection_sampling=args.rejection_sampling,
        )
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
