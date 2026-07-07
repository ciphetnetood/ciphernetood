from __future__ import annotations

import torch
from tqdm import tqdm

from ciphernet_ood.evaluation.metrics import ood_metrics
from ciphernet_ood.ood.scorer import cpg_scores


@torch.no_grad()
def evaluate_open_world(model, id_loader, ood_loader, signatures: dict, config: dict, device: torch.device) -> dict:
    model.eval()
    labels, scores = [], []
    for is_ood, loader in [(0, id_loader), (1, ood_loader)]:
        for data in tqdm(loader, desc="Open-world evaluation"):
            data = data.to(device)
            score, _ = cpg_scores(model, data, signatures, config)
            labels.extend([is_ood] * score.numel())
            scores.extend(score.cpu().tolist())
    return ood_metrics(labels, scores, float(signatures["threshold"]))
