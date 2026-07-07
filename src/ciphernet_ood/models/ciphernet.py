from __future__ import annotations

import torch
from torch import nn

from ciphernet_ood.models.cfml_encoder import CFMLEncoder
from ciphernet_ood.models.prototypes import TrafficModePrototypes


class CipherNetOOD(nn.Module):
    def __init__(self, input_dim: int, num_classes: int, config: dict):
        super().__init__()
        self.encoder = CFMLEncoder(input_dim=input_dim, config=config)
        self.prototypes = TrafficModePrototypes(
            num_classes=num_classes,
            num_prototypes=int(config["model"]["num_prototypes"]),
            embedding_dim=int(config["model"]["projection_dim"]),
        )

    def forward(self, data) -> dict[str, torch.Tensor | list[torch.Tensor]]:
        z, hidden_states = self.encoder(data)
        similarities = self.prototypes.similarities(z)
        logits = similarities.max(dim=-1).values
        return {"z": z, "hidden_states": hidden_states, "similarities": similarities, "logits": logits}

    @torch.no_grad()
    def predict(self, data) -> torch.Tensor:
        return self.forward(data)["logits"].argmax(dim=-1)
