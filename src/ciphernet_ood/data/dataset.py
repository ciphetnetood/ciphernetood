from __future__ import annotations

from pathlib import Path

import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class GraphFileDataset(torch.utils.data.Dataset):
    def __init__(self, files: list[str | Path]):
        self.files = [Path(file) for file in files]

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Data:
        return torch.load(self.files[index], map_location="cpu")


def make_loader(files: list[str | Path], batch_size: int, shuffle: bool, num_workers: int = 0) -> DataLoader:
    return DataLoader(GraphFileDataset(files), batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
