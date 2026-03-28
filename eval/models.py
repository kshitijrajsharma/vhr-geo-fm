"""Task factory -- builds terratorch tasks matching GEO-Bench-2 protocol."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
from typing import Any

import torch
from terratorch.tasks import (
    MultiLabelClassificationTask,
    ObjectDetectionTask,
    SemanticSegmentationTask,
)
from torch import nn

from eval.datasets import DatasetConfig

MODEL_REGISTRY_PATH = Path(__file__).resolve().parent.parent / "models.json"

# Backbones that don't support out_indices (select post-hoc instead)
_SKIP_OUT_INDICES = {
    "terratorch_terramind_v1_large",
    "terratorch_dinov3_vitl16",
    "terratorch_dinov3_convnext_large",
}

_DEFAULT_INFO: dict[str, Any] = {
    "arch": "cnn",
    "backbone_kwargs": {},
    "seg_out_indices": [1, 2, 3, 4],
    "cls_out_indices": [-1],
}


def _load_registry() -> dict[str, dict]:
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)
    return {
        backbone: info
        for cat in registry["categories"].values()
        for backbone, info in cat["models"].items()
    }


_REGISTRY = _load_registry()


def _info(backbone: str) -> dict:
    return _REGISTRY.get(backbone, _DEFAULT_INFO)


def _out_indices_key(task_type: str) -> str:
    return "seg_out_indices" if task_type in ("segmentation", "detection") else "cls_out_indices"


def _backbone_kwargs(
    backbone: str, pretrained: bool, task_type: str, img_size: int
) -> dict[str, Any]:
    info = _info(backbone)
    bk: dict[str, Any] = {"pretrained": pretrained, **info.get("backbone_kwargs", {})}

    if backbone not in _SKIP_OUT_INDICES:
        key = _out_indices_key(task_type)
        bk["out_indices"] = info.get(key, _DEFAULT_INFO[key])

    # DINOv3: gated checkpoints via env vars
    _CKPT_ENV = {
        "terratorch_dinov3_vitl16": "DINOV3_VITL16_CKPT",
        "terratorch_dinov3_convnext_large": "DINOV3_CONVNEXT_CKPT",
    }
    if backbone in _CKPT_ENV:
        ckpt = os.environ.get(_CKPT_ENV[backbone])
        if ckpt:
            bk["ckpt_path"] = ckpt

    # ViT img_size must match terratorch pad_images (doubles patch_size)
    if "dofa" in backbone or backbone == "timm_clay_v1_base":
        ps = 16 if "dofa" in backbone else 8
        bk["img_size"] = math.ceil(img_size / (ps * 2)) * (ps * 2)

    return bk


def _build_backbone(backbone: str, pretrained: bool, task_type: str, img_size: int) -> nn.Module:
    from terratorch.registry import BACKBONE_REGISTRY

    model = BACKBONE_REGISTRY.build(
        backbone, **_backbone_kwargs(backbone, pretrained, task_type, img_size)
    )
    info = _info(backbone)
    key = _out_indices_key(task_type)

    # Post-hoc layer selection for backbones without out_indices
    if backbone in _SKIP_OUT_INDICES:
        model = _IndexSelectWrapper(model, info.get(key, [5, 11, 17, 23]))

    # Fix out_channels when backbone reports all layers but out_indices filters
    expected = info.get(key, _DEFAULT_INFO[key])
    if hasattr(model, "out_channels") and len(model.out_channels) != len(expected):
        total = len(model.out_channels)
        resolved = [i if i >= 0 else total + i for i in expected]
        model.out_channels = [model.out_channels[i] for i in resolved]

    return model


class _IndexSelectWrapper(nn.Module):
    """Selects specific feature indices from backbones returning all layers."""

    def __init__(self, backbone: nn.Module, indices: list[int]):
        super().__init__()
        self.backbone = backbone
        self._indices = indices
        total = len(backbone.out_channels)
        resolved = [i if i >= 0 else total + i for i in indices]
        self.out_channels = [backbone.out_channels[i] for i in resolved]

    def forward(self, x: torch.Tensor, **kw) -> list[torch.Tensor]:
        feats = self.backbone(x, **kw)
        n = len(feats)
        return [feats[i if i >= 0 else n + i] for i in self._indices]


def _necks(backbone: str, task_type: str) -> list[dict]:
    """Terratorch neck chain per GEO-Bench-2 Section 3.3."""
    arch = _info(backbone).get("arch", "cnn")
    if arch == "cnn":
        return []

    spatial = task_type in ("segmentation", "detection")

    if arch == "vit":
        if spatial:
            return [
                {"name": "ReshapeTokensToImage", "remove_cls_token": True},
                {"name": "LearnedInterpolateToPyramidal"},
            ]
        return [{"name": "AggregateTokens", "pooling": "mean", "drop_cls": True}]

    if arch == "swin":
        base = [{"name": "PermuteDims", "new_order": [0, 3, 1, 2]}]
        if not spatial:
            base.append({"name": "AggregateTokens", "pooling": "mean"})
        return base

    return []


def _model_args(
    backbone: str, num_classes: int, pretrained: bool, img_size: int, task_type: str
) -> dict:
    info = _info(backbone)
    args: dict[str, Any] = {
        "backbone": _build_backbone(backbone, pretrained, task_type, img_size),
        "necks": _necks(backbone, task_type),
        "num_classes": num_classes,
        "head_dropout": 0.1,
    }
    if task_type == "segmentation":
        args["decoder"] = "UNetDecoder"
        args["decoder_channels"] = info.get("seg_decoder_channels", [512, 256, 128, 64])
    else:
        args["decoder"] = "IdentityDecoder"
    return args


def create_task(
    config: DatasetConfig,
    backbone: str,
    lr: float,
    weight_decay: float = 0.01,
    frozen: bool = True,
    pretrained: bool = True,
):
    """Create terratorch task. Constant LR per GEO-Bench-2 protocol (no scheduler)."""
    common = {
        "freeze_backbone": frozen,
        "freeze_decoder": False,
        "lr": lr,
        "optimizer": "AdamW",
        "optimizer_hparams": {"weight_decay": weight_decay},
    }

    if config.task_type == "segmentation":
        return SemanticSegmentationTask(
            model_args=_model_args(
                backbone, config.num_classes, pretrained, config.img_size, "segmentation"
            ),
            model_factory="EncoderDecoderFactory",
            loss=config.loss,
            ignore_index=-100,
            **common,
        )

    if config.task_type == "classification":
        return MultiLabelClassificationTask(
            model_args=_model_args(
                backbone, config.num_classes, pretrained, config.img_size, "classification"
            ),
            model_factory="EncoderDecoderFactory",
            loss=config.loss,
            class_names=[str(i) for i in range(config.num_classes)],
            **common,
        )

    if config.task_type == "detection":
        det_args = {
            "backbone": backbone,
            "backbone_pretrained": pretrained,
            "num_classes": config.num_classes,
            "framework": "faster-rcnn",
            "in_channels": 3,
        }
        return ObjectDetectionTask(
            model_factory="ObjectDetectionModelFactory",
            model_args=det_args,
            boxes_field="bbox_xyxy",
            labels_field="label",
            **common,
        )

    raise ValueError(f"Unknown task type: {config.task_type}")
