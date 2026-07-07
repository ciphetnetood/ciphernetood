from __future__ import annotations

import torch
from torch.nn import functional as F

from ciphernet_ood.losses.regularizers import inter_prototype_loss, intra_prototype_loss
from ciphernet_ood.losses.traffic_margin import batch_dispersion, class_confusability, traffic_conditioned_margin


def cfml_loss(outputs: dict, labels: torch.Tensor, class_frequency: torch.Tensor, prototypes: torch.Tensor, config: dict) -> dict[str, torch.Tensor]:
    z = outputs["z"]
    similarities = outputs["similarities"]
    num_classes, num_prototypes = prototypes.shape[:2]
    cfg = config["cfml"]
    scale = float(cfg["feature_scale"])

    target_sims = similarities[torch.arange(labels.numel(), device=labels.device), labels]
    assignments = target_sims.argmax(dim=-1)
    dispersion = batch_dispersion(z, labels, assignments, num_classes, num_prototypes)
    confusability = class_confusability(z, labels, num_classes, float(cfg["kappa"]))
    margins = traffic_conditioned_margin(class_frequency, dispersion, confusability, config)

    logits = similarities.max(dim=-1).values * scale
    target_cos = target_sims[torch.arange(labels.numel(), device=labels.device), assignments].clamp(-1 + 1e-7, 1 - 1e-7)
    target_theta = torch.acos(target_cos)
    target_margin = margins[labels, assignments]
    logits[torch.arange(labels.numel(), device=labels.device), labels] = torch.cos(target_theta + target_margin) * scale

    cls = F.cross_entropy(logits, labels)
    intra = intra_prototype_loss(z, labels, assignments, prototypes)
    inter = inter_prototype_loss(prototypes, float(cfg["delta"]))
    total = cls + float(cfg["lambda_intra"]) * intra + float(cfg["lambda_inter"]) * inter
    return {"loss": total, "cls": cls.detach(), "intra": intra.detach(), "inter": inter.detach()}
