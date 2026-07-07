from __future__ import annotations

import torch
from torch import nn


class TrafficModePrototypes(nn.Module):
    def __init__(self, num_classes: int, num_prototypes: int, embedding_dim: int):
        super().__init__()
        weights = torch.randn(num_classes, num_prototypes, embedding_dim)
        self.weights = nn.Parameter(torch.nn.functional.normalize(weights, p=2, dim=-1))

    def normalized(self) -> torch.Tensor:
        return torch.nn.functional.normalize(self.weights, p=2, dim=-1)

    def similarities(self, z: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bd,ycd->byc", torch.nn.functional.normalize(z, p=2, dim=-1), self.normalized())

    def normalize_(self) -> None:
        with torch.no_grad():
            self.weights.copy_(self.normalized())
