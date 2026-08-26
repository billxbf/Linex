"""OpenCode harness — https://github.com/opencode-ai/opencode"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import RUNTIME_AGENT_LOG_DIR, BaseRuntime
from polar.runtime.models import ExecInput

# Substituted with $OPENAI_BASE_URL at exec time (the gateway env is only
# present during run steps, not setup), keeping the JSON static.
_BASE_URL_PLACEHOLDER = "__POLAR_GATEWAY_BASE_URL__"


class OpenCodeHarness(BaseHarness):
    """Run OpenCode CLI in non-interactive mode.

    The model is registered in ``opencode.json`` with the endpoint under
    ``provider.<name>.options.baseURL`` (opencode reads it from provider
    options, not the environment), MCP servers use opencode's
    ``local``/``remote`` types, and permissions are bypassed with
    ``--dangerously-skip-permissions`` instead of enumerating permission keys.
    """

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._config_dir = "$HOME/.config/opencode"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._config_dir}")

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._config_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._config_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        model = self.model_name or "openai/gpt-5.4"
        provider, model_id = model.split("/", 1) if "/" in model else ("openai", model)

        config: dict = {
            "provider": {
                provider: {
                    "models": {model_id: {}},
                    "options": {"baseURL": _BASE_URL_PLACEHOLDER},
                }
            },
        }
        if self.mcp_servers:
            mcp_config: dict = {}
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    command = [server.command, *server.args] if server.command else []
                    mcp_config[server.name] = {"type": "local", "command": command}
                else:  # sse or streamable-http
                    mcp_config[server.name] = {"type": "remote", "url": server.url}
            config["mcp"] = mcp_config
        config_json = json.dumps(config, indent=2)

        env: dict[str, str] = {
            **self.env,
            "OPENCODE_FAKE_VCS": "git",
        }

        return [
            ExecInput(
                command=(
                    f"mkdir -p {self._config_dir} && "
                    # baseURL placeholder -> $OPENAI_BASE_URL at exec time.
                    f"printf '%s' {shlex.quote(config_json)} "
                    f'| sed "s|{_BASE_URL_PLACEHOLDER}|$OPENAI_BASE_URL|g" '
                    f"> {self._config_dir}/opencode.json && "
                    f"set -o pipefail && opencode --model={shlex.quote(model)} run "
                    f"--format=json --dangerously-skip-permissions -- {escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/opencode.txt"
                ),
                env=env,
            )
        ]
