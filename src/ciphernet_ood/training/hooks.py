from __future__ import annotations


class EarlyStopping:
    def __init__(self, patience: int):
        self.patience = patience
        self.best = float("-inf")
        self.bad_epochs = 0

    def step(self, value: float) -> bool:
        if value > self.best:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience
