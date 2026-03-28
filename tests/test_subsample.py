"""Tests for data-fraction subsampling."""

from unittest.mock import MagicMock

import torch
from torch.utils.data import TensorDataset

from eval.subsample import SubsampledDataModule


def _mock_datamodule(n_train=100):
    """Create a minimal mock datamodule with a train dataset."""
    dm = MagicMock()
    dm.train_dataset = TensorDataset(torch.randn(n_train, 3, 32, 32))
    return dm


class TestSubsample:
    def test_50pct_halves_dataset(self):
        dm = _mock_datamodule(100)
        sub = SubsampledDataModule(dm, data_pct=50)
        sub.setup(stage="fit")
        assert len(dm.train_dataset) == 50

    def test_1pct_minimum_one_sample(self):
        dm = _mock_datamodule(10)
        sub = SubsampledDataModule(dm, data_pct=1)
        sub.setup(stage="fit")
        assert len(dm.train_dataset) >= 1

    def test_100pct_no_subsample(self):
        dm = _mock_datamodule(100)
        sub = SubsampledDataModule(dm, data_pct=100)
        sub.setup(stage="fit")
        # 100% should not wrap in Subset
        assert len(dm.train_dataset) == 100

    def test_deterministic_same_seed(self):
        dm1 = _mock_datamodule(100)
        dm2 = _mock_datamodule(100)
        # Same underlying data
        dm2.train_dataset = dm1.train_dataset.__class__(*dm1.train_dataset.tensors)

        sub1 = SubsampledDataModule(dm1, data_pct=10, seed=42)
        sub2 = SubsampledDataModule(dm2, data_pct=10, seed=42)
        sub1.setup(stage="fit")
        sub2.setup(stage="fit")

        idx1 = dm1.train_dataset.indices
        idx2 = dm2.train_dataset.indices
        assert idx1 == idx2

    def test_different_seed_different_indices(self):
        dm1 = _mock_datamodule(100)
        dm2 = _mock_datamodule(100)
        dm2.train_dataset = dm1.train_dataset.__class__(*dm1.train_dataset.tensors)

        sub1 = SubsampledDataModule(dm1, data_pct=50, seed=1)
        sub2 = SubsampledDataModule(dm2, data_pct=50, seed=2)
        sub1.setup(stage="fit")
        sub2.setup(stage="fit")

        idx1 = set(dm1.train_dataset.indices)
        idx2 = set(dm2.train_dataset.indices)
        assert idx1 != idx2

    def test_delegates_val_dataloader(self):
        dm = _mock_datamodule()
        sub = SubsampledDataModule(dm, data_pct=50)
        sub.val_dataloader()
        dm.val_dataloader.assert_called_once()

    def test_delegates_test_dataloader(self):
        dm = _mock_datamodule()
        sub = SubsampledDataModule(dm, data_pct=50)
        sub.test_dataloader()
        dm.test_dataloader.assert_called_once()
