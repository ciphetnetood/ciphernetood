from __future__ import annotations

from pathlib import Path

import torch
from tqdm import tqdm

from ciphernet_ood.evaluation.closed_set import evaluate_closed_set
from ciphernet_ood.losses.cfml_loss import cfml_loss
from ciphernet_ood.losses.traffic_margin import normalized_class_frequency
from ciphernet_ood.training.checkpointing import save_checkpoint
from ciphernet_ood.training.hooks import EarlyStopping


def collect_labels(loader) -> torch.Tensor:
    labels = []
    for data in loader:
        labels.extend(data.y.view(-1).tolist())
    return torch.tensor(labels, dtype=torch.long)


def train_cfml(model, train_loader, val_loader, config: dict, device: torch.device, checkpoint_dir: str | Path) -> dict:
    model.to(device)
    opt = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    labels = collect_labels(train_loader).to(device)
    num_classes = model.prototypes.weights.size(0)
    class_frequency = normalized_class_frequency(labels, num_classes)
    stopper = EarlyStopping(int(config["training"]["early_stopping_patience"]))
    best_metrics: dict = {}
    checkpoint_dir = Path(checkpoint_dir)

    for epoch in range(1, int(config["training"]["max_epochs"]) + 1):
        model.train()
        losses = []
        for data in tqdm(train_loader, desc=f"Epoch {epoch}"):
            data = data.to(device)
            opt.zero_grad(set_to_none=True)
            outputs = model(data)
            loss_dict = cfml_loss(outputs, data.y.view(-1), class_frequency, model.prototypes.normalized(), config)
            loss_dict["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            opt.step()
            model.prototypes.normalize_()
            losses.append(float(loss_dict["loss"].detach().cpu()))

        metrics = evaluate_closed_set(model, val_loader, device)
        metrics["train_loss"] = sum(losses) / max(len(losses), 1)
        monitor = float(metrics.get(config["training"].get("monitor_metric", "macro_f1"), metrics["macro_f1"]))
        if monitor >= stopper.best:
            best_metrics = metrics
            save_checkpoint(checkpoint_dir / "best.pt", model, opt, epoch, metrics, config, class_frequency.cpu())
        if stopper.step(monitor):
            break
    return best_metrics
