"""Tests for dataset registry completeness and configuration consistency."""

import pytest

from eval.datasets import DATASET_CONFIGS, DatasetConfig
from eval.scoring import NORMALIZER

EXPECTED_DATASETS = [
    "spacenet2",
    "spacenet7",
    "flair2",
    "dynamic_earthnet",
    "treesatai",
    "everwatch",
    "nzcattle",
]


class TestDatasetRegistry:
    def test_all_7_datasets_registered(self):
        assert set(DATASET_CONFIGS.keys()) == set(EXPECTED_DATASETS)

    @pytest.mark.parametrize("name", EXPECTED_DATASETS)
    def test_config_is_dataclass(self, name):
        assert isinstance(DATASET_CONFIGS[name], DatasetConfig)

    @pytest.mark.parametrize("name", EXPECTED_DATASETS)
    def test_valid_task_type(self, name):
        assert DATASET_CONFIGS[name].task_type in ("segmentation", "classification", "detection")

    @pytest.mark.parametrize("name", EXPECTED_DATASETS)
    def test_metric_direction_valid(self, name):
        assert DATASET_CONFIGS[name].metric_direction in ("max", "min")

    @pytest.mark.parametrize("name", EXPECTED_DATASETS)
    def test_num_classes_positive(self, name):
        assert DATASET_CONFIGS[name].num_classes > 0

    @pytest.mark.parametrize("name", EXPECTED_DATASETS)
    def test_normalizer_range_exists(self, name):
        assert name in NORMALIZER
        mn, mx = NORMALIZER[name]
        assert mn < mx


class TestTaskTypes:
    def test_segmentation_datasets(self):
        seg = [n for n, c in DATASET_CONFIGS.items() if c.task_type == "segmentation"]
        assert set(seg) == {"spacenet2", "spacenet7", "flair2", "dynamic_earthnet"}

    def test_classification_datasets(self):
        cls = [n for n, c in DATASET_CONFIGS.items() if c.task_type == "classification"]
        assert cls == ["treesatai"]

    def test_detection_datasets(self):
        det = [n for n, c in DATASET_CONFIGS.items() if c.task_type == "detection"]
        assert set(det) == {"everwatch", "nzcattle"}


class TestMetricKeys:
    """Verify terratorch metric key conventions."""

    def test_seg_metric_key_format(self):
        for name in ["spacenet2", "spacenet7", "flair2", "dynamic_earthnet"]:
            assert DATASET_CONFIGS[name].metric_key == "val/mIoU"

    def test_cls_metric_key_format(self):
        assert DATASET_CONFIGS["treesatai"].metric_key == "val/Multilabel_F1_Score"

    def test_det_metric_key_format(self):
        for name in ["everwatch", "nzcattle"]:
            # Detection uses underscore, no slash
            assert DATASET_CONFIGS[name].metric_key == "val_map"


class TestBandOrder:
    """Verify band_order format matches dataset requirements."""

    def test_list_format_datasets(self):
        for name in ["spacenet2", "flair2"]:
            bo = DATASET_CONFIGS[name].band_order
            assert isinstance(bo, list), f"{name} should use list band_order"

    def test_dict_format_datasets(self):
        for name in ["dynamic_earthnet", "treesatai"]:
            bo = DATASET_CONFIGS[name].band_order
            assert isinstance(bo, dict), f"{name} should use dict band_order"

    def test_dict_datasets_have_stacked_image(self):
        for name in ["dynamic_earthnet", "treesatai"]:
            cfg = DATASET_CONFIGS[name]
            assert cfg.datamodule_kwargs.get("return_stacked_image") is True
