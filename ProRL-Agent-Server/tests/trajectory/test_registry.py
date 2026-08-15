from __future__ import annotations

import asyncio

import pytest

from polar.trajectory.builder.base import BaseTrajectoryBuilder
from polar.trajectory.models import CompletionSession, StrategySpec, Trajectory
from polar.trajectory.registry import StrategyRegistry, default_builder_registry


class DummyBuilder(BaseTrajectoryBuilder):
    def __init__(self, *, status: str = "COMPLETED") -> None:
        self.status = status

    async def build(self, session: CompletionSession) -> Trajectory:
        return Trajectory(status=self.status, metadata={"session_id": session.session_id})


class NotABuilder:
    pass


def test_strategy_registry_creates_fresh_instances_with_config() -> None:
    registry: StrategyRegistry[BaseTrajectoryBuilder] = StrategyRegistry(BaseTrajectoryBuilder)
    registry.register("dummy", DummyBuilder)

    builder = registry.create(StrategySpec(strategy="dummy", config={"status": "TIMEOUT"}))
    trajectory = asyncio.run(builder.build(CompletionSession(session_id="session-1")))

    assert trajectory.status == "TIMEOUT"
    assert trajectory.metadata["session_id"] == "session-1"


def test_strategy_registry_rejects_non_subclasses() -> None:
    registry: StrategyRegistry[BaseTrajectoryBuilder] = StrategyRegistry(BaseTrajectoryBuilder)

    with pytest.raises(TypeError, match="subclass"):
        registry.register("bad", NotABuilder)


def test_default_builder_registry_exposes_built_in_builders() -> None:
    registry = default_builder_registry()

    assert registry.list_strategies() == ["per_request", "prefix_merging"]
    assert registry.create(StrategySpec(strategy="per_request")).__class__.__name__ == "PerRequestBuilder"


def test_unknown_strategy_raises_clear_error() -> None:
    registry: StrategyRegistry[BaseTrajectoryBuilder] = StrategyRegistry(BaseTrajectoryBuilder)

    with pytest.raises(KeyError, match="Unknown strategy"):
        registry.create(StrategySpec(strategy="missing"))
