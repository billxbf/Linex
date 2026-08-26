"""Qwen Code harness — https://github.com/QwenLM/qwen-code"""

from __future__ import annotations

import json
import shlex

from polar.agent.base import BaseHarness
from polar.agent.models import AgentSpec
from polar.runtime.base import RUNTIME_AGENT_LOG_DIR, BaseRuntime
from polar.runtime.models import ExecInput


class QwenCodeHarness(BaseHarness):
    """Run Qwen Code CLI in non-interactive mode.

    Matches the eval-side qwen-code setup: MCP servers as a name-keyed
    ``mcpServers`` object and explicit headless auth
    (``--auth-type openai --openai-api-key/--openai-base-url``), pointed at the
    gateway env the run step injects.
    """

    def __init__(self, agent_spec: AgentSpec) -> None:
        super().__init__(agent_spec)
        self._qwen_dir = "$HOME/.qwen"

    async def setup(self, runtime: BaseRuntime) -> None:
        await runtime.exec(f"mkdir -p {self._qwen_dir}")

        # Register MCP servers (object keyed by server name, the CLI's schema).
        if self.mcp_servers:
            servers: dict[str, dict] = {}
            for server in self.mcp_servers:
                if server.transport == "stdio":
                    servers[server.name] = {"command": server.command, "args": server.args}
                elif server.transport == "streamable-http":
                    servers[server.name] = {"httpUrl": server.url}
                else:  # sse
                    servers[server.name] = {"url": server.url}
            config_json = json.dumps({"mcpServers": servers}, indent=2)
            await runtime.exec(
                f"cat > {self._qwen_dir}/settings.json << 'POLARCFG'\n{config_json}\nPOLARCFG"
            )

        # Copy skills
        if self.skills_path:
            await runtime.exec(
                f"mkdir -p {self._qwen_dir}/skills && "
                f"cp -r {shlex.quote(self.skills_path)}/* {self._qwen_dir}/skills/ 2>/dev/null || true"
            )

    def run_steps(self, instruction: str) -> list[ExecInput]:
        escaped = shlex.quote(instruction)
        env: dict[str, str] = {**self.env}
        # qwen-code reads the model from OPENAI_MODEL; passing both an env var
        # and a --model CLI flag created conflicts on proxied backends, so only
        # the env var form is used. The provider prefix is stripped.
        if self.model_name:
            env["OPENAI_MODEL"] = self.model_name.split("/", 1)[-1]

        return [
            ExecInput(
                command=(
                    # Pin headless auth to the OpenAI-compatible gateway; the
                    # auth dialog cannot render under --prompt.
                    "set -o pipefail && qwen --yolo --auth-type openai "
                    '--openai-api-key "$OPENAI_API_KEY" '
                    '--openai-base-url "$OPENAI_BASE_URL" '
                    f"--prompt={escaped} "
                    f"2>&1 | tee {RUNTIME_AGENT_LOG_DIR}/qwen-code.txt"
                ),
                env=env,
            )
        ]
