from __future__ import annotations

import itertools

import numpy as np
import torch
from torch_geometric.data import Data

from ciphernet_ood.data.packet_features import PacketRecord, encode_packet_features


def _add_edge(edges: list[tuple[int, int]], i: int, j: int) -> None:
    if i != j:
        edges.append((i, j))
        edges.append((j, i))


def build_edge_index(meta: np.ndarray, mask: np.ndarray, config: dict) -> torch.Tensor:
    graph_cfg = config["graph"]
    valid = np.flatnonzero(mask > 0)
    edges: list[tuple[int, int]] = []

    if graph_cfg.get("temporal_edges", True):
        for i, j in zip(valid[:-1], valid[1:]):
            _add_edge(edges, int(i), int(j))

    if graph_cfg.get("direction_edges", True):
        for i, j in itertools.combinations(valid, 2):
            if np.sign(meta[i, 1]) == np.sign(meta[j, 1]):
                _add_edge(edges, int(i), int(j))

    if graph_cfg.get("burst_edges", True):
        burst_start = 0
        for idx in range(1, len(valid) + 1):
            boundary = idx == len(valid) or np.sign(meta[valid[idx], 1]) != np.sign(meta[valid[idx - 1], 1])
            if boundary:
                for i, j in itertools.combinations(valid[burst_start:idx], 2):
                    _add_edge(edges, int(i), int(j))
                burst_start = idx

    if graph_cfg.get("size_similarity_edges", True):
        threshold = float(graph_cfg.get("size_similarity_threshold", 0.1))
        for i, j in itertools.combinations(valid, 2):
            if abs(float(meta[i, 0] - meta[j, 0])) <= threshold:
                _add_edge(edges, int(i), int(j))

    if graph_cfg.get("time_similarity_edges", True):
        threshold = float(graph_cfg.get("time_similarity_threshold", 0.1))
        for i, j in itertools.combinations(valid, 2):
            if abs(float(meta[i, 2] - meta[j, 2])) <= threshold:
                _add_edge(edges, int(i), int(j))

    if graph_cfg.get("add_self_loops", True):
        edges.extend((int(i), int(i)) for i in valid)

    if not edges:
        edges = [(0, 0)]
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def session_to_graph(packets: list[PacketRecord], label: int, config: dict) -> Data:
    sess_cfg = config["session"]
    x, mask, meta = encode_packet_features(
        packets=packets,
        max_packets=int(sess_cfg["max_packets"]),
        max_payload_bytes=int(sess_cfg["max_payload_bytes"]),
        byte_normalization=float(sess_cfg["byte_normalization"]),
    )
    edge_index = build_edge_index(meta, mask, config)
    return Data(
        x=torch.tensor(x, dtype=torch.float32),
        edge_index=edge_index,
        y=torch.tensor(label, dtype=torch.long),
        mask=torch.tensor(mask, dtype=torch.float32),
    )
