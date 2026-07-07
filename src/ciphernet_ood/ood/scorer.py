from __future__ import annotations

import torch

from ciphernet_ood.ood.gram import gram_vectors


def mahalanobis(x: torch.Tensor, mean: torch.Tensor, precision: torch.Tensor) -> torch.Tensor:
    diff = x - mean.to(x.device)
    return torch.einsum("bi,ij,bj->b", diff, precision.to(x.device), diff)


@torch.no_grad()
def cpg_scores(model, data, signatures: dict, config: dict) -> tuple[torch.Tensor, torch.Tensor]:
    outputs = model(data)
    z = outputs["z"]
    similarities = outputs["similarities"]
    flat_sim = similarities.flatten(start_dim=1)
    posterior = torch.softmax(float(config["ood"]["prototype_temperature"]) * flat_sim, dim=-1)
    y_count, c_count = similarities.shape[1], similarities.shape[2]
    layer_weights = [float(v) for v in config["ood"]["layer_weights"]]
    selected_layers = [int(i) - 1 for i in config["ood"]["selected_layers"]]
    eps = float(config["ood"]["gram_normalization_eps"])

    distances = torch.zeros(z.size(0), y_count * c_count, device=z.device)
    for local_idx, layer_idx in enumerate(selected_layers):
        vectors = gram_vectors(outputs["hidden_states"][layer_idx], data.batch, eps=eps)
        for y in range(y_count):
            for c in range(c_count):
                key = (layer_idx, y, c)
                if key not in signatures["means"]:
                    distances[:, y * c_count + c] += 1e6
                    continue
                d = mahalanobis(vectors, signatures["means"][key], signatures["precisions"][key])
                distances[:, y * c_count + c] += layer_weights[local_idx] * d
    score = -torch.log(torch.sum(posterior * torch.exp(-distances), dim=-1).clamp_min(1e-12))
    pred = similarities.max(dim=-1).values.argmax(dim=-1)
    return score, pred
