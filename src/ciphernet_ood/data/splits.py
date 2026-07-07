from __future__ import annotations

import random
from collections import defaultdict
from pathlib import Path

import torch


def stratified_split(files: list[Path], train: float, val: float, seed: int) -> dict[str, list[str]]:
    rng = random.Random(seed)
    by_label: dict[int, list[Path]] = defaultdict(list)
    for file in files:
        data = torch.load(file, map_location="cpu")
        by_label[int(data.y.item())].append(file)

    result = {"train": [], "val": [], "test": []}
    for label_files in by_label.values():
        rng.shuffle(label_files)
        n = len(label_files)
        n_train = int(n * train)
        n_val = int(n * val)
        result["train"].extend(str(p) for p in label_files[:n_train])
        result["val"].extend(str(p) for p in label_files[n_train : n_train + n_val])
        result["test"].extend(str(p) for p in label_files[n_train + n_val :])
    return result
