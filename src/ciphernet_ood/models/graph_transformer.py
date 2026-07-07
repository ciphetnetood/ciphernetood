from __future__ import annotations

import torch
from torch import nn
from torch_geometric.nn import TransformerConv


class GraphTransformerEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_layers: int, num_heads: int, dropout: float):
        super().__init__()
        self.input = nn.Linear(input_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [
                TransformerConv(
                    in_channels=hidden_dim,
                    out_channels=hidden_dim // num_heads,
                    heads=num_heads,
                    dropout=dropout,
                    beta=True,
                )
                for _ in range(num_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> list[torch.Tensor]:
        h = self.input(x)
        hidden_states = []
        for conv, norm in zip(self.layers, self.norms):
            residual = h
            h = conv(h, edge_index)
            h = norm(residual + self.dropout(torch.relu(h)))
            hidden_states.append(h)
        return hidden_states
