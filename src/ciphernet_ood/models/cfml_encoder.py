from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool

from ciphernet_ood.models.graph_transformer import GraphTransformerEncoder


class CFMLEncoder(nn.Module):
    def __init__(self, input_dim: int, config: dict):
        super().__init__()
        model_cfg = config["model"]
        hidden_dim = int(model_cfg["hidden_dim"])
        projection_dim = int(model_cfg["projection_dim"])
        self.readout = model_cfg.get("readout", "mean")
        self.normalize_embeddings = bool(model_cfg.get("normalize_embeddings", True))
        self.encoder = GraphTransformerEncoder(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=int(model_cfg["num_layers"]),
            num_heads=int(model_cfg.get("num_heads", 4)),
            dropout=float(model_cfg.get("dropout", 0.1)),
        )
        self.projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim),
        )

    def pool(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        if self.readout == "max":
            return global_max_pool(h, batch)
        if self.readout == "sum":
            return global_add_pool(h, batch)
        return global_mean_pool(h, batch)

    def forward(self, data) -> tuple[torch.Tensor, list[torch.Tensor]]:
        hidden_states = self.encoder(data.x, data.edge_index)
        pooled = self.pool(hidden_states[-1], data.batch)
        z = self.projector(pooled)
        if self.normalize_embeddings:
            z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z, hidden_states
