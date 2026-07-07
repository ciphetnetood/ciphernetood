from __future__ import annotations

import torch


def normalized_class_frequency(labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    counts = torch.bincount(labels.detach().cpu(), minlength=num_classes).float().clamp_min(1.0)
    freq = counts / counts.max()
    return freq.to(labels.device)


def batch_dispersion(z: torch.Tensor, labels: torch.Tensor, assignments: torch.Tensor, num_classes: int, num_prototypes: int) -> torch.Tensor:
    result = torch.zeros(num_classes, num_prototypes, device=z.device)
    for y in range(num_classes):
        for c in range(num_prototypes):
            idx = (labels == y) & (assignments == c)
            if idx.any():
                samples = z[idx]
                mean = samples.mean(dim=0, keepdim=True)
                result[y, c] = torch.norm(samples - mean, dim=-1).mean()
    return result


def class_confusability(z: torch.Tensor, labels: torch.Tensor, num_classes: int, kappa: float) -> torch.Tensor:
    means = []
    for y in range(num_classes):
        idx = labels == y
        if idx.any():
            means.append(torch.nn.functional.normalize(z[idx].mean(dim=0), p=2, dim=0))
        else:
            means.append(torch.zeros(z.size(-1), device=z.device))
    means_t = torch.stack(means)
    sim = means_t @ means_t.t()
    eye = torch.eye(num_classes, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, 0.0)
    return torch.relu(sim - kappa).sum(dim=1) / max(num_classes - 1, 1)


def traffic_conditioned_margin(
    class_frequency: torch.Tensor,
    dispersion: torch.Tensor,
    confusability: torch.Tensor,
    config: dict,
) -> torch.Tensor:
    cfg = config["cfml"]
    margin = (
        float(cfg["base_margin"])
        * (1.0 / class_frequency.clamp_min(1e-6)).pow(float(cfg["beta"]))[:, None]
        * (1.0 + float(cfg["gamma"]) * dispersion)
        * (1.0 + float(cfg["eta"]) * confusability[:, None])
    )
    return margin.clamp(float(cfg["margin_min"]), float(cfg["margin_max"]))
