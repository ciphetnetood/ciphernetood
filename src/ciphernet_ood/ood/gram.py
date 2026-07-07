from __future__ import annotations

import torch
from torch_geometric.utils import to_dense_batch


def graph_hidden_batches(hidden: torch.Tensor, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return to_dense_batch(hidden, batch)


def gram_vectors(hidden: torch.Tensor, batch: torch.Tensor, eps: float = 1e-8, upper_triangle: bool = True) -> torch.Tensor:
    dense, mask = graph_hidden_batches(hidden, batch)
    dense = dense * mask.unsqueeze(-1)
    grams = torch.bmm(dense.transpose(1, 2), dense)
    grams = grams / (torch.linalg.matrix_norm(grams, ord="fro", dim=(1, 2), keepdim=True) + eps)
    if upper_triangle:
        idx = torch.triu_indices(grams.size(1), grams.size(2), device=grams.device)
        return grams[:, idx[0], idx[1]]
    return grams.flatten(start_dim=1)
