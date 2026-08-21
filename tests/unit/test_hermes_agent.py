from polar.agent.models import AgentSpec
from polar.agent.presets.hermes import HermesHarness


def test_hermes_home_is_isolated_per_runtime_session() -> None:
    harness = HermesHarness(
        AgentSpec(harness="hermes", model_name="Qwen3.5-9B", settings={"toolsets": "terminal,file"})
    )

    step = harness.run_steps("Fix the workspace.")[0]

    assert "mkdir -p /polar/session/hermes" in step.command
    assert "> /polar/session/hermes/config.yaml" in step.command
    assert "set -o pipefail && hermes" in step.command
    assert "--toolsets terminal,file" in step.command
    assert step.env["HERMES_HOME"] == "/polar/session/hermes"
