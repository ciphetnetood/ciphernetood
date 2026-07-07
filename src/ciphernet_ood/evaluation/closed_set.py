from __future__ import annotations

import torch
from tqdm import tqdm

from ciphernet_ood.evaluation.metrics import classification_metrics, confusion, imbalance_metrics


@torch.no_grad()
def evaluate_closed_set(model, loader, device: torch.device) -> dict:
    model.eval()
    y_true, y_pred = [], []
    for data in tqdm(loader, desc="Closed-set evaluation"):
        data = data.to(device)
        pred = model.predict(data)
        y_true.extend(data.y.view(-1).cpu().tolist())
        y_pred.extend(pred.cpu().tolist())
    metrics = classification_metrics(y_true, y_pred)
    metrics.update(imbalance_metrics(y_true, y_pred))
    metrics["confusion_matrix"] = confusion(y_true, y_pred)
    return metrics
