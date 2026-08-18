from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from polar.config import TopologyConfig


def _write_yaml(path: Path, data: dict) -> Path:
    path.write_text(yaml.safe_dump(data))
    return path


def test_topology_defaults_public_urls_and_rollout_url(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "topology.yaml",
        {
            "rollout": {
                "host": "0.0.0.0",
                "port": 8080,
            },
            "gateway": {
                "nodes": [
                    {
                        "id": "node-a",
                        "host": "0.0.0.0",
                        "port": 8100,
                    }
                ],
            },
        },
    )

    topology = TopologyConfig.load(path)

    assert topology.rollout.public_url == "http://127.0.0.1:8080"
    assert topology.gateway.nodes[0].public_url == "http://127.0.0.1:8100"
    assert topology.gateway.rollout_server_url == topology.rollout.public_url


def test_topology_rejects_unknown_keys(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "topology.yaml",
        {
            "rollout": {"unexpected": True},
            "gateway": {
                "nodes": [
                    {
                        "id": "node-a",
                        "public_url": "http://127.0.0.1:8100",
                    }
                ],
            },
        },
    )

    with pytest.raises(ValueError, match="unexpected"):
        TopologyConfig.load(path)


def test_select_gateway_requires_node_id_for_multi_node_topology(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "topology.yaml",
        {
            "gateway": {
                "nodes": [
                    {"id": "node-a", "public_url": "http://127.0.0.1:8100"},
                    {"id": "node-b", "public_url": "http://127.0.0.1:8101"},
                ],
            },
        },
    )
    topology = TopologyConfig.load(path)

    with pytest.raises(ValueError, match="--node-id"):
        topology.select_gateway_node()

    assert topology.select_gateway_node("node-b").port == 8081


def test_inference_block_selects_base_url(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "topology.yaml",
        {
            "gateway": {
                "nodes": [
                    {
                        "id": "node-a",
                        "public_url": "http://127.0.0.1:8100",
                        "inference": {"base_url": "http://127.0.0.1:8000"},
                    }
                ],
            },
        },
    )
    node = TopologyConfig.load(path).gateway.nodes[0]
    assert node.inference_base_url == "http://127.0.0.1:8000"


def test_inference_defaults_when_block_omitted(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "topology.yaml",
        {"gateway": {"nodes": [{"id": "node-a", "public_url": "http://127.0.0.1:8100"}]}},
    )
    node = TopologyConfig.load(path).gateway.nodes[0]
    assert node.inference_base_url == "http://127.0.0.1:8000"


def test_inference_backend_selector_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "topology.yaml",
        {
            "gateway": {
                "nodes": [
                    {
                        "id": "node-a",
                        "public_url": "http://127.0.0.1:8100",
                        "inference": {"engine": "vllm", "base_url": "http://127.0.0.1:8000"},
                    }
                ],
            },
        },
    )
    with pytest.raises(ValueError, match="engine"):
        TopologyConfig.load(path)


def test_invalid_inference_base_url_is_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "topology.yaml",
        {
            "gateway": {
                "nodes": [
                    {
                        "id": "node-a",
                        "public_url": "http://127.0.0.1:8100",
                        "inference": {"base_url": "not-a-url"},
                    }
                ],
            },
        },
    )
    with pytest.raises(ValueError, match="inference.base_url"):
        TopologyConfig.load(path)


def test_duplicate_gateway_node_ids_are_rejected(tmp_path: Path) -> None:
    path = _write_yaml(
        tmp_path / "topology.yaml",
        {
            "gateway": {
                "nodes": [
                    {"id": "node-a", "public_url": "http://127.0.0.1:8100"},
                    {"id": "node-a", "public_url": "http://127.0.0.1:8101"},
                ],
            },
        },
    )

    with pytest.raises(ValueError, match="Duplicate gateway node id"):
        TopologyConfig.load(path)


def test_servers_accept_generated_topology_without_phantom_nodes(monkeypatch) -> None:
    from polar.gateway import server as gateway_server
    from polar.rollout import server as rollout_server

    topology = TopologyConfig.model_validate(
        {
            "rollout": {"public_url": "http://rollout:8080"},
            "gateway": {
                "rollout_server_url": "http://rollout:8080",
                "nodes": [
                    {
                        "id": "node-a",
                        "public_url": "http://gateway:8081",
                        "inference": {"base_url": "http://router:8000"},
                    }
                ],
            },
        }
    )

    rollout_state = rollout_server._build_state(topology)
    assert rollout_state.scheduler.list_nodes() == []

    monkeypatch.setattr(gateway_server, "_state", None)
    monkeypatch.setattr(gateway_server, "_configured_topology", None)
    monkeypatch.setattr(gateway_server, "_configured_topology_path", None)
    monkeypatch.setattr(gateway_server, "_configured_node_id", None)
    monkeypatch.setattr(gateway_server, "_build_state", lambda value, node_id: (value, node_id))
    gateway_server.configure_server(topology, node_id="node-a")
    assert gateway_server.get_state() == (topology, "node-a")
