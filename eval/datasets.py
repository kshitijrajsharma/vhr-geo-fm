"""Dataset configurations and datamodule factory for GEO-Bench-2 VHR datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from geobench_v2.datamodules.dynamic_earthnet import GeoBenchDynamicEarthNetDataModule
from geobench_v2.datamodules.everwatch import GeoBenchEverWatchDataModule
from geobench_v2.datamodules.flair2 import GeoBenchFLAIR2DataModule
from geobench_v2.datamodules.nzcattle import GeoBenchNZCattleDataModule
from geobench_v2.datamodules.spacenet2 import GeoBenchSpaceNet2DataModule
from geobench_v2.datamodules.spacenet7 import GeoBenchSpaceNet7DataModule
from geobench_v2.datamodules.treesatai import GeoBenchTreeSatAIDataModule


@dataclass
class DatasetConfig:
    name: str
    task_type: str  # segmentation | classification | detection
    metric_key: str  # terratorch metric key used for val monitoring
    metric_leaderboard: str  # leaderboard metric name for CSV export
    metric_direction: str  # max | min
    num_classes: int
    img_size: int
    loss: str
    band_order: Any  # list or dict passed to datamodule
    datamodule_class: type
    datamodule_kwargs: dict = field(default_factory=dict)
    normalizer_range: tuple[float, float] = (0.0, 1.0)


DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "spacenet2": DatasetConfig(
        name="spacenet2",
        task_type="segmentation",
        metric_key="val/mIoU",
        metric_leaderboard="Multiclass_Jaccard_Index",
        metric_direction="max",
        num_classes=3,
        img_size=512,
        loss="ce",
        band_order=["red", "green", "blue"],
        datamodule_class=GeoBenchSpaceNet2DataModule,
        normalizer_range=(0.8431392908096313, 0.8844738006591797),
    ),
    "spacenet7": DatasetConfig(
        name="spacenet7",
        task_type="segmentation",
        metric_key="val/mIoU",
        metric_leaderboard="Multiclass_Jaccard_Index",
        metric_direction="max",
        num_classes=3,
        img_size=512,
        loss="ce",
        band_order=("red", "green", "blue"),
        datamodule_class=GeoBenchSpaceNet7DataModule,
        normalizer_range=(0.3912872672080993, 0.6402554512023926),
    ),
    "flair2": DatasetConfig(
        name="flair2",
        task_type="segmentation",
        metric_key="val/mIoU",
        metric_leaderboard="Multiclass_Jaccard_Index",
        metric_direction="max",
        num_classes=13,
        img_size=512,
        loss="ce",
        band_order=["red", "green", "blue"],
        datamodule_class=GeoBenchFLAIR2DataModule,
        normalizer_range=(0.4062314331531524, 0.5503402948379517),
    ),
    "dynamic_earthnet": DatasetConfig(
        name="dynamic_earthnet",
        task_type="segmentation",
        metric_key="val/mIoU",
        metric_leaderboard="Multiclass_Jaccard_Index",
        metric_direction="max",
        num_classes=7,
        img_size=512,
        loss="ce",
        band_order={"planet": ("r", "g", "b")},
        datamodule_class=GeoBenchDynamicEarthNetDataModule,
        datamodule_kwargs={"return_stacked_image": True, "temporal_setting": "single"},
        normalizer_range=(0.1660756915807724, 0.3561779260635376),
    ),
    "treesatai": DatasetConfig(
        name="treesatai",
        task_type="classification",
        metric_key="val/Multilabel_F1_Score",
        metric_leaderboard="Multilabel_F1_Score",
        metric_direction="max",
        num_classes=15,
        img_size=304,
        loss="bce",
        band_order={"aerial": ["red", "green", "blue"]},
        datamodule_class=GeoBenchTreeSatAIDataModule,
        datamodule_kwargs={"return_stacked_image": True},
        normalizer_range=(0.5388144254684448, 0.6719674468040466),
    ),
    "everwatch": DatasetConfig(
        name="everwatch",
        task_type="detection",
        metric_key="val_map",
        metric_leaderboard="Mean_Average_Precision",
        metric_direction="max",
        num_classes=9,
        img_size=512,
        loss="",
        band_order=("red", "green", "blue"),
        datamodule_class=GeoBenchEverWatchDataModule,
        normalizer_range=(0.2177008986473083, 0.3155497610569),
    ),
    "nzcattle": DatasetConfig(
        name="nzcattle",
        task_type="detection",
        metric_key="val_map",
        metric_leaderboard="Mean_Average_Precision",
        metric_direction="max",
        num_classes=2,
        img_size=512,
        loss="",
        band_order=("red", "green", "blue"),
        datamodule_class=GeoBenchNZCattleDataModule,
        normalizer_range=(0.2819505035877228, 0.401744931936264),
    ),
}


def create_datamodule(
    config: DatasetConfig,
    batch_size: int,
    data_root: str,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """Create a GEO-Bench-2 datamodule from config."""
    kwargs: dict[str, Any] = {
        "img_size": config.img_size,
        "band_order": config.band_order,
        "batch_size": batch_size,
        "eval_batch_size": batch_size * 2,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "root": data_root,
        **config.datamodule_kwargs,
    }
    return config.datamodule_class(**kwargs)
