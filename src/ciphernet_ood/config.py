from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = value
    return result


def load_yaml(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def load_config(config_path: str | Path, dataset: str | None = None) -> dict[str, Any]:
    config_path = Path(config_path)
    config = load_yaml(config_path)
    if dataset:
        dataset_path = config_path.parent / "datasets" / f"{dataset}.yaml"
        if dataset_path.exists():
            config = deep_update(config, load_yaml(dataset_path))
        config.setdefault("dataset", {})["name"] = dataset
    return config


def ensure_dirs(config: dict[str, Any], dataset: str | None = None) -> None:
    names = [
        config["project"]["output_dir"],
        config["project"]["checkpoint_dir"],
        config["data"]["processed_dir"],
        config["data"]["split_dir"],
        config["data"]["session_dir"],
    ]
    for name in names:
        path = Path(name)
        if dataset and name in {config["project"]["checkpoint_dir"], config["data"]["processed_dir"], config["data"]["split_dir"]}:
            path = path / dataset
        path.mkdir(parents=True, exist_ok=True)
