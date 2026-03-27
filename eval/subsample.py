"""Data-fraction wrapper for GEO-Bench-2 datamodules."""

from __future__ import annotations

from typing import Any, cast

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Subset


class SubsampledDataModule(LightningDataModule):
    """Wraps a GEO-Bench-2 datamodule to subsample the training set.

    Only the training set is subsampled. Validation and test sets remain unchanged.
    Subsampling is deterministic via a fixed seed.
    """

    def __init__(self, datamodule: LightningDataModule, data_pct: int, seed: int = 42):
        super().__init__()
        self.datamodule = datamodule
        self.data_pct = data_pct
        self.seed = seed

    def __getattr__(self, name: str):
        return getattr(self.datamodule, name)

    def setup(self, stage: str | None = None) -> None:
        stage_arg = stage if stage is not None else "fit"
        self.datamodule.setup(stage=stage_arg)
        if stage in ("fit",) and self.data_pct < 100:
            wrapped_datamodule = cast(Any, self.datamodule)
            dataset = wrapped_datamodule.train_dataset
            n = len(dataset)
            k = max(1, int(n * self.data_pct / 100))
            g = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(n, generator=g)[:k].tolist()
            wrapped_datamodule.train_dataset = Subset(dataset, indices)

    def prepare_data(self) -> None:
        self.datamodule.prepare_data()

    def train_dataloader(self) -> DataLoader:
        return self.datamodule.train_dataloader()

    def val_dataloader(self) -> DataLoader:
        return self.datamodule.val_dataloader()

    def test_dataloader(self) -> DataLoader:
        return self.datamodule.test_dataloader()

    def on_after_batch_transfer(self, batch, dataloader_idx: int):
        return self.datamodule.on_after_batch_transfer(batch, dataloader_idx)
