from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PacketRecord:
    payload: bytes
    size: int
    direction: int
    timestamp: float
    position: int


def encode_packet_features(
    packets: list[PacketRecord],
    max_packets: int,
    max_payload_bytes: int,
    byte_normalization: float = 255.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_dim = max_payload_bytes + 4
    features = np.zeros((max_packets, feature_dim), dtype=np.float32)
    mask = np.zeros(max_packets, dtype=np.float32)
    meta = np.zeros((max_packets, 4), dtype=np.float32)

    previous_time: float | None = None
    for out_idx, packet in enumerate(packets[:max_packets]):
        payload = np.frombuffer(packet.payload[:max_payload_bytes], dtype=np.uint8).astype(np.float32)
        payload = payload / byte_normalization
        features[out_idx, : len(payload)] = payload

        interarrival = 0.0 if previous_time is None else max(packet.timestamp - previous_time, 0.0)
        previous_time = packet.timestamp

        size_value = np.log1p(max(packet.size, 0)) / 16.0
        time_value = np.log1p(interarrival)
        direction_value = 1.0 if packet.direction >= 0 else -1.0
        position_value = packet.position / max(max_packets - 1, 1)

        behavior = np.array([size_value, direction_value, time_value, position_value], dtype=np.float32)
        features[out_idx, max_payload_bytes:] = behavior
        meta[out_idx] = behavior
        mask[out_idx] = 1.0

    return features, mask, meta
