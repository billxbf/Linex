"""Eval-parity checks: presets must configure each agent the way eval runs do."""

import json

from polar.agent.models import AgentSpec, MCPServerSpec
from polar.agent.presets.claude_code import ClaudeCodeHarness
from polar.agent.presets.codex import CodexHarness
from polar.agent.presets.gemini_cli import GeminiCliHarness
from polar.agent.presets.hermes import HermesHarness
from polar.agent.presets.opencode import OpenCodeHarness
from polar.agent.presets.qwen_code import QwenCodeHarness

_MCP = [
    MCPServerSpec(name="files", transport="stdio", command="mcp-files", args=["--root", "/w"]),
    MCPServerSpec(name="web", transport="streamable-http", url="http://mcp.local/http"),
]


def test_hermes_config_matches_eval_settings() -> None:
    harness = HermesHarness(AgentSpec(harness="hermes", model_name="Qwen3.5-9B"))
    config = harness._build_config()
    assert config["compression"] == {"enabled": True, "threshold": 0.85}
    assert config["delegation"] == {"max_iterations": 50}
    assert config["agent"] == {"max_turns": 90}


def test_claude_code_uses_permission_mode_and_background_tasks() -> None:
    harness = ClaudeCodeHarness(AgentSpec(harness="claude_code", model_name="policy"))
    step = harness.run_steps("do it")[0]
    assert "--permission-mode=bypassPermissions" in step.command
    assert "--dangerously-skip-permissions" not in step.command
    assert "set -o pipefail" in step.command
    assert step.env["FORCE_AUTO_BACKGROUND_TASKS"] == "1"
    assert step.env["ENABLE_BACKGROUND_TASKS"] == "1"


def test_codex_writes_base_url_before_mcp_tables() -> None:
    harness = CodexHarness(
        AgentSpec(harness="codex", model_name="openai/gpt-5.5", mcp_servers=_MCP)
    )
    config_step, run_step = harness.run_steps("do it")
    # Root key emission must come before the [mcp_servers.*] heredoc, or TOML
    # parses openai_base_url as a key of the last server table.
    assert config_step.command.index("openai_base_url") < config_step.command.index(
        'mcp_servers."files"'
    )
    assert 'type = "streamable-http"' not in config_step.command
    assert "-c model_reasoning_effort=high" in run_step.command
    assert "set -o pipefail" in run_step.command


def test_gemini_settings_pin_auth_and_use_object_mcp_shape() -> None:
    harness = GeminiCliHarness(
        AgentSpec(
            harness="gemini_cli",
            model_name="gemini-2.5-pro",
            settings={"reasoning_effort": "high"},
            mcp_servers=_MCP,
        )
    )
    config = harness._config
    assert config["security"]["auth"]["selectedType"] == "gemini-api-key"
    assert config["experimental"] == {"skills": True}
    assert config["mcpServers"]["files"] == {"command": "mcp-files", "args": ["--root", "/w"]}
    assert config["mcpServers"]["web"] == {"httpUrl": "http://mcp.local/http"}
    alias = "polar-gemini-2.5-pro-high"
    thinking = config["modelConfigs"]["customAliases"][alias]["modelConfig"]
    assert thinking["generateContentConfig"]["thinkingConfig"]["thinkingLevel"] == "HIGH"
    assert f"--model={alias}" in harness.run_steps("go")[0].command


def test_qwen_pins_headless_openai_auth() -> None:
    harness = QwenCodeHarness(AgentSpec(harness="qwen_code", model_name="openai/qwen3-coder"))
    step = harness.run_steps("go")[0]
    assert "--auth-type openai" in step.command
    assert '--openai-base-url "$OPENAI_BASE_URL"' in step.command
    assert step.env["OPENAI_MODEL"] == "qwen3-coder"


def test_opencode_registers_gateway_endpoint_and_local_remote_mcp() -> None:
    harness = OpenCodeHarness(
        AgentSpec(harness="opencode", model_name="openai/gpt-5.4", mcp_servers=_MCP)
    )
    step = harness.run_steps("go")[0]
    config_json = step.command.split("printf '%s' ")[1].split(" | sed")[0]
    config = json.loads(config_json.replace("'\\''", "'")[1:-1])
    assert config["provider"]["openai"]["options"]["baseURL"] == "__POLAR_GATEWAY_BASE_URL__"
    assert config["mcp"]["files"] == {"type": "local", "command": ["mcp-files", "--root", "/w"]}
    assert config["mcp"]["web"] == {"type": "remote", "url": "http://mcp.local/http"}
    assert "--dangerously-skip-permissions" in step.command
    assert '"permission"' not in step.command
