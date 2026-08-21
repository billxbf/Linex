from __future__ import annotations

import asyncio

from polar.config import TopologyConfig
from polar.gateway.server import _build_state


def test_gateway_sessions_use_shared_rollout_directory(tmp_path) -> None:
    topology = TopologyConfig.model_validate(
        {
            "rollout": {"public_url": "http://rollout:8080", "save_dir": str(tmp_path)},
            "gateway": {
                "rollout_server_url": "http://rollout:8080",
                "nodes": [{"id": "node-1", "public_url": "http://gateway:8081"}],
            },
        }
    )

    state = _build_state(topology, "node-1")
    expected = tmp_path / "sessions" / "node-1"
    assert state.node_manager._session_base_dir == str(expected)
    assert expected.is_dir()
    asyncio.run(state.inference.close())
    state.storage.close()
