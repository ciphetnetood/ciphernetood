from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ciphernet_ood.config import ensure_dirs, load_config
from ciphernet_ood.data.dataset import make_loader
from ciphernet_ood.models.ciphernet import CipherNetOOD
from ciphernet_ood.utils.io import load_json, save_json


def parse_base_args() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--dataset", required=True)
    return parser


def load_project(args) -> dict:
    config = load_config(args.config, args.dataset)
    ensure_dirs(config, args.dataset)
    return config


def graph_files(config: dict, dataset: str) -> list[Path]:
    root = Path(config["data"]["processed_dir"]) / dataset
    return sorted(root.glob("*/*.pt")) + sorted(root.glob("*.pt"))


def load_splits(config: dict, dataset: str) -> dict:
    path = Path(config["data"]["split_dir"]) / dataset / "closed_set.json"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}. Run build_graphs.py first.")
    return load_json(path)


def save_splits(config: dict, dataset: str, splits: dict) -> None:
    save_json(splits, Path(config["data"]["split_dir"]) / dataset / "closed_set.json")


def infer_shapes(files: list[str | Path]) -> tuple[int, int]:
    if not files:
        raise RuntimeError("No graph files found.")
    sample = torch.load(files[0], map_location="cpu")
    labels = []
    for file in files:
        labels.append(int(torch.load(file, map_location="cpu").y.item()))
    return int(sample.x.size(-1)), max(labels) + 1


def build_model(config: dict, files: list[str | Path], device: torch.device) -> CipherNetOOD:
    input_dim, num_classes = infer_shapes(files)
    return CipherNetOOD(input_dim=input_dim, num_classes=num_classes, config=config).to(device)


def loaders_from_splits(config: dict, splits: dict):
    tr_cfg = config["training"]
    batch = int(tr_cfg["batch_size"])
    workers = int(tr_cfg.get("num_workers", 0))
    return {
        name: make_loader(files, batch_size=batch, shuffle=(name == "train"), num_workers=workers)
        for name, files in splits.items()
    }
