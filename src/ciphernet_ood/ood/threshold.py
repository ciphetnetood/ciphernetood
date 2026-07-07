from __future__ import annotations

import numpy as np
import torch
from tqdm import tqdm

from ciphernet_ood.ood.scorer import cpg_scores


@torch.no_grad()
def calibrate_threshold(model, loader, signatures: dict, config: dict, device: torch.device) -> float:
    scores = []
    for data in tqdm(loader, desc="Calibrating OOD threshold"):
        data = data.to(device)
        score, _ = cpg_scores(model, data, signatures, config)
        scores.extend(score.cpu().tolist())
    return float(np.percentile(np.asarray(scores), float(config["ood"]["threshold_percentile"])))
