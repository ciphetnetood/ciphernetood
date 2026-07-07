from __future__ import annotations

import torch


def perturb_graph(data, kind: str, amount: float):
    data = data.clone()
    behavior_start = data.x.size(1) - 4
    if kind == "packet_size_jitter":
        data.x[:, behavior_start] *= 1.0 + amount * torch.randn_like(data.x[:, behavior_start])
    elif kind == "timing_jitter":
        data.x[:, behavior_start + 2] *= 1.0 + amount * torch.randn_like(data.x[:, behavior_start + 2])
    elif kind == "direction_masking":
        mask = torch.rand_like(data.x[:, behavior_start + 1]) < amount
        data.x[mask, behavior_start + 1] = 0.0
    return data
