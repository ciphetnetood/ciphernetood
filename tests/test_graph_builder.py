from __future__ import annotations

from ciphernet_ood.data.graph_builder import session_to_graph
from ciphernet_ood.data.packet_features import PacketRecord


def _config():
    return {
        "session": {"max_packets": 4, "max_payload_bytes": 8, "byte_normalization": 255.0},
        "graph": {
            "temporal_edges": True,
            "direction_edges": True,
            "burst_edges": True,
            "size_similarity_edges": True,
            "time_similarity_edges": True,
            "size_similarity_threshold": 0.2,
            "time_similarity_threshold": 0.2,
            "add_self_loops": True,
        },
    }


def test_session_to_graph_shapes():
    packets = [
        PacketRecord(b"abcdef", 100, 1, 0.0, 0),
        PacketRecord(b"123", 110, -1, 0.1, 1),
    ]
    graph = session_to_graph(packets, label=1, config=_config())
    assert graph.x.shape == (4, 12)
    assert graph.mask.sum().item() == 2
    assert graph.edge_index.shape[0] == 2
    assert graph.y.item() == 1
