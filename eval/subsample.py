"""Data-fraction wrapper for GEO-Bench-2 datamodules."""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader, Subset


class SubsampledDataModule:
    """Wraps a GEO-Bench-2 datamodule to subsample the training set.

    Only the training set is subsampled. Validation and test sets remain unchanged.
    Subsampling is deterministic via a fixed seed.
    """

    def __init__(self, datamodule, data_pct: int, seed: int = 42):
        self.datamodule = datamodule
        self.data_pct = data_pct
        self.seed = seed

    def __getattr__(self, name):
        return getattr(self.datamodule, name)

    def setup(self, stage=None):
        self.datamodule.setup(stage=stage)
        if stage in ("fit",) and self.data_pct < 100:
            dataset = self.datamodule.train_dataset
            n = len(dataset)
            k = max(1, int(n * self.data_pct / 100))
            g = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(n, generator=g)[:k].tolist()
            self.datamodule.train_dataset = Subset(dataset, indices)

    def prepare_data(self):
        self.datamodule.prepare_data()

    def train_dataloader(self) -> DataLoader:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.datamodule.test_dataloader()

    def on_after_batch_transfer(self, batch, dataloader_idx):
        return self.datamodule.on_after_batch_transfer(batch, dataloader_idx)
