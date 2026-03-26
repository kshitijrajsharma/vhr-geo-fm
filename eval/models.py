"""Task factory for GEO-Bench-2 evaluation using terratorch built-in tasks."""

from __future__ import annotations

from terratorch.tasks import (
    MultiLabelClassificationTask,
    ObjectDetectionTask,
    SemanticSegmentationTask,
)

from eval.datasets import DatasetConfig


def _seg_model_args(backbone: str, num_classes: int, pretrained: bool) -> dict:
    """Model args for semantic segmentation: UNet decoder."""
    return {
        "backbone": backbone,
        "backbone_pretrained": pretrained,
        "decoder": "UNetDecoder",
        "decoder_channels": [512, 256, 128, 64],
        "backbone_out_indices": [1, 2, 3, 4],
        "num_classes": num_classes,
        "head_dropout": 0.1,
    }


def _cls_model_args(backbone: str, num_classes: int, pretrained: bool) -> dict:
    """Model args for classification: IdentityDecoder + linear head (paper Section 3.3)."""
    return {
        "backbone": backbone,
        "backbone_pretrained": pretrained,
        "decoder": "IdentityDecoder",
        "backbone_out_indices": [-1],
        "num_classes": num_classes,
        "head_dropout": 0.1,
    }


def _det_model_args(backbone: str, num_classes: int, pretrained: bool) -> dict:
    """Model args for object detection: Faster R-CNN + FPN."""
    return {
        "backbone": backbone,
        "backbone_pretrained": pretrained,
        "num_classes": num_classes,
        "framework": "faster-rcnn",
        "in_channels": 3,
    }


def create_task(
    config: DatasetConfig,
    backbone: str,
    lr: float,
    weight_decay: float = 0.01,
    frozen: bool = True,
    pretrained: bool = True,
):
    """Create a terratorch task matching GEO-Bench-2 protocol."""
    freeze_backbone = frozen
    freeze_decoder = False

    if config.task_type == "segmentation":
        model_args = _seg_model_args(backbone, config.num_classes, pretrained)
        return SemanticSegmentationTask(
            model_args=model_args,
            model_factory="EncoderDecoderFactory",
            loss=config.loss,
            ignore_index=-100,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
            lr=lr,
            optimizer="AdamW",
            optimizer_hparams={"weight_decay": weight_decay},
            scheduler="ReduceLROnPlateau",
            scheduler_hparams={"mode": "min", "factor": 0.5, "patience": 5},
        )

    if config.task_type == "classification":
        model_args = _cls_model_args(backbone, config.num_classes, pretrained)
        return MultiLabelClassificationTask(
            model_args=model_args,
            model_factory="EncoderDecoderFactory",
            loss=config.loss,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
            lr=lr,
            optimizer="AdamW",
            optimizer_hparams={"weight_decay": weight_decay},
            scheduler="ReduceLROnPlateau",
            scheduler_hparams={"mode": "min", "factor": 0.5, "patience": 5},
            class_names=[str(i) for i in range(config.num_classes)],
        )

    if config.task_type == "detection":
        model_args = _det_model_args(backbone, config.num_classes, pretrained)
        return ObjectDetectionTask(
            model_factory="ObjectDetectionModelFactory",
            model_args=model_args,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
            lr=lr,
            optimizer="AdamW",
            optimizer_hparams={"weight_decay": weight_decay},
            scheduler="ReduceLROnPlateau",
            scheduler_hparams={"mode": "min", "factor": 0.5, "patience": 5},
            boxes_field="bbox_xyxy",
            labels_field="label",
        )

    msg = f"Unknown task type: {config.task_type}"
    raise ValueError(msg)
