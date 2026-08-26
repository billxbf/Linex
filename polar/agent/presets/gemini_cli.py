"""Gemini CLI harness — https://github.com/google/gemini-cli"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import RUNTIME_AGENT_LOG_DIR, BaseRuntime
from polar.runtime.models import ExecInput


class GeminiCliHarness(BaseHarness):
    """Run Google Gemini CLI in non-interactive mode.

    Config matches the eval-side gemini-cli setup so trained behavior
    transfers: same settings.json shape (auth pinning, mcpServers object,
    experimental.skills, thinking-level model alias) and the same run flags.
    """

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._gemini_dir = "$HOME/.gemini"
        # Headless `gemini --prompt` cannot show the auth-method dialog, and its
        # auto-detection breaks when a custom GOOGLE_GEMINI_BASE_URL is set (the
        # gateway case), so pin API-key auth.
        self._config: dict = {"security": {"auth": {"selectedType": "gemini-api-key"}}}
        self._run_model = self.model_name

        if self.mcp_servers:
            servers: dict[str, dict] = {}
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    servers[server.name] = {"command": server.command, "args": server.args}
                elif server.transport == "streamable-http":
                    servers[server.name] = {"httpUrl": server.url}
                else:  # sse
                    servers[server.name] = {"url": server.url}
            self._config["mcpServers"] = servers

        # Thinking level rides on a custom model alias, the CLI's only
        # non-interactive way to set a thinkingConfig.
        reasoning_effort = self.settings.get("reasoning_effort")
        if self.model_name and reasoning_effort:
            effort = str(reasoning_effort)
            self._run_model = f"polar-{self.model_name}-{effort}"
            self._config["modelConfigs"] = {
                "customAliases": {
                    self._run_model: {
                        "modelConfig": {
                            "model": self.model_name,
                            "generateContentConfig": {
                                "thinkingConfig": {
                                    "includeThoughts": True,
                                    "thinkingLevel": effort.upper(),
                                },
                            },
                        }
                    }
                }
            }

        self._config["experimental"] = {"skills": True}

    async def setup(self, runtime: BaseRuntime) -> None:
        config_json = json.dumps(self._config, indent=2)
        await runtime.exec(
            f"mkdir -p {self._gemini_dir} && "
            f"cat > {self._gemini_dir}/settings.json << 'POLARCFG'\n{config_json}\nPOLARCFG"
        )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._gemini_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._gemini_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {
            "GEMINI_CLI_TRUST_WORKSPACE": "true",
            **self.env,
        }

        flags: list[str] = ["--yolo"]
        if self._run_model:
            flags.append(f"--model={shlex.quote(self._run_model)}")
        if self.settings.get("sandbox") is True:
            flags.append("--sandbox")

        flags_str = " ".join(flags)
        return [
            ExecInput(
                command=(
                    # The gateway injects GOOGLE_API_KEY / GOOGLE_API_URL; the
                    # Gemini CLI reads GEMINI_API_KEY / GOOGLE_GEMINI_BASE_URL,
                    # so map one onto the other to route calls at the proxy.
                    'export GEMINI_API_KEY="$GOOGLE_API_KEY" '
                    'GOOGLE_GEMINI_BASE_URL="$GOOGLE_API_URL" && '
                    f"set -o pipefail && gemini {flags_str} --prompt={escaped} "
                    f"2>&1 </dev/null | tee {RUNTIME_AGENT_LOG_DIR}/gemini-cli.txt"
                ),
                env=env,
            )
        ]
