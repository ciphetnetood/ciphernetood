from __future__ import annotations

import torch
from tqdm import tqdm

from ciphernet_ood.ood.gram import gram_vectors


@torch.no_grad()
def fit_cpg_signatures(model, loader, config: dict, device: torch.device) -> dict:
    model.eval()
    selected_layers = [int(i) - 1 for i in config["ood"]["selected_layers"]]
    eps = float(config["ood"]["gram_normalization_eps"])
    reg = float(config["ood"]["covariance_regularization"])
    buckets: dict[tuple[int, int, int], list[torch.Tensor]] = {}

    for data in tqdm(loader, desc="Fitting CPG signatures"):
        data = data.to(device)
        outputs = model(data)
        labels = data.y.view(-1)
        assignments = outputs["similarities"][torch.arange(labels.numel(), device=device), labels].argmax(dim=-1)
        for layer_idx in selected_layers:
            vectors = gram_vectors(outputs["hidden_states"][layer_idx], data.batch, eps=eps)
            for row, label, proto in zip(vectors.cpu(), labels.cpu(), assignments.cpu()):
                buckets.setdefault((layer_idx, int(label), int(proto)), []).append(row)

    signatures = {"means": {}, "precisions": {}, "threshold": None, "config": config["ood"]}
    for key, rows in buckets.items():
        mat = torch.stack(rows)
        mean = mat.mean(dim=0)
        centered = mat - mean
        if mat.size(0) > 1:
            cov = centered.t().matmul(centered) / (mat.size(0) - 1)
        else:
            cov = torch.eye(mat.size(1)) * reg
        cov = cov + reg * torch.eye(cov.size(0))
        signatures["means"][key] = mean
        signatures["precisions"][key] = torch.linalg.pinv(cov)
    return signatures
