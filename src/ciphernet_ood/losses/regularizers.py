from __future__ import annotations

import torch


def intra_prototype_loss(z: torch.Tensor, labels: torch.Tensor, assignments: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
    target_proto = prototypes[labels, assignments]
    return torch.square(z - target_proto).sum(dim=-1).mean()


def inter_prototype_loss(prototypes: torch.Tensor, delta: float) -> torch.Tensor:
    y, c, d = prototypes.shape
    flat = prototypes.reshape(y * c, d)
    class_ids = torch.arange(y, device=prototypes.device).repeat_interleave(c)
    sim = flat @ flat.t()
    different = class_ids[:, None] != class_ids[None, :]
    penalties = torch.relu(sim[different] - delta)
    if penalties.numel() == 0:
        return torch.zeros((), device=prototypes.device)
    return penalties.mean()
