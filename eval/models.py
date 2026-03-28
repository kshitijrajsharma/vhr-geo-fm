"""Task factory for GEO-Bench-2 evaluation using terratorch built-in tasks."""

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


def _load_model_registry() -> dict[str, dict]:
    """Load the model registry and flatten to backbone → config mapping."""
    with open(MODEL_REGISTRY_PATH) as f:
        registry = json.load(f)
    flat: dict[str, dict] = {}
    for cat in registry["categories"].values():
        for backbone, info in cat["models"].items():
            flat[backbone] = info
    return flat


_MODEL_REGISTRY = _load_model_registry()

# Backbones that don't accept out_indices in their constructor
_SKIP_OUT_INDICES = {
    "terratorch_terramind_v1_large",
    "terratorch_dinov3_vitl16",
    "terratorch_dinov3_convnext_large",
}


def _get_model_info(backbone: str) -> dict:
    """Get model info from registry, with defaults for unknown models."""
    if backbone in _MODEL_REGISTRY:
        return _MODEL_REGISTRY[backbone]
    return {
        "arch": "cnn",
        "backbone_kwargs": {},
        "seg_out_indices": [1, 2, 3, 4],
        "cls_out_indices": [-1],
    }


def _build_backbone_kwargs(
    backbone: str, pretrained: bool, task_type: str, img_size: int,
) -> dict[str, Any]:
    """Build backbone-specific kwargs from the model registry."""
    info = _get_model_info(backbone)
    bk: dict[str, Any] = {}
    bk["pretrained"] = pretrained

    # Add model-specific backbone kwargs (model_bands, bands, num_frames, etc.)
    for k, v in info.get("backbone_kwargs", {}).items():
        bk[k] = v

    if backbone not in _SKIP_OUT_INDICES:
        if task_type in ("segmentation", "detection"):
            bk["out_indices"] = info.get("seg_out_indices", [1, 2, 3, 4])
        else:
            bk["out_indices"] = info.get("cls_out_indices", [-1])

    # DINOv3 models require ckpt_path for pretrained weights (gated by Meta)
    if backbone == "terratorch_dinov3_vitl16":
        ckpt = os.environ.get("DINOV3_VITL16_CKPT")
        if ckpt:
            bk["ckpt_path"] = ckpt
    elif backbone == "terratorch_dinov3_convnext_large":
        ckpt = os.environ.get("DINOV3_CONVNEXT_CKPT")
        if ckpt:
            bk["ckpt_path"] = ckpt

    # ViT models that need img_size for position embedding interpolation.
    # terratorch pads images to multiples of 2*patch_size (see pad_images in models/utils.py),
    # so we must match the padded size for pos_embed initialization.
    if "dofa" in backbone or backbone == "timm_clay_v1_base":
        patch_size = 16 if "dofa" in backbone else 8  # Clay uses 8x8 patches
        effective_patch = patch_size * 2  # terratorch doubles for decoder compatibility
        padded_size = math.ceil(img_size / effective_patch) * effective_patch
        bk["img_size"] = padded_size

    return bk


def _build_backbone(backbone: str, pretrained: bool, task_type: str, img_size: int) -> nn.Module:
    """Build backbone module with necessary wrappers for out_channels correctness."""
    from terratorch.registry import BACKBONE_REGISTRY

    bk = _build_backbone_kwargs(backbone, pretrained, task_type, img_size)
    model = BACKBONE_REGISTRY.build(backbone, **bk)

    info = _get_model_info(backbone)

    # For TerraMind/DINOv3 (no out_indices), select specific layers post-hoc
    if backbone in _SKIP_OUT_INDICES:
        if task_type in ("segmentation", "detection"):
            indices = info.get("seg_out_indices", [5, 11, 17, 23])
        else:
            indices = info.get("cls_out_indices", [-1])
        model = _IndexSelectWrapper(model, indices)

    # Fix out_channels metadata when backbone reports all layers but out_indices filters
    # (e.g. Prithvi reports 32 out_channels but only returns 4 features with out_indices)
    if task_type in ("segmentation", "detection"):
        expected_indices = info.get("seg_out_indices", [1, 2, 3, 4])
    else:
        expected_indices = info.get("cls_out_indices", [-1])
    if hasattr(model, "out_channels") and len(model.out_channels) != len(expected_indices):
        total = len(model.out_channels)
        resolved = [i if i >= 0 else total + i for i in expected_indices]
        model.out_channels = [model.out_channels[i] for i in resolved]

    return model


class _IndexSelectWrapper(nn.Module):
    """Select specific feature indices from a backbone that returns all layers."""

    def __init__(self, backbone: nn.Module, indices: list[int]):
        super().__init__()
        self.backbone = backbone
        self._indices = indices
        total = len(backbone.out_channels)
        resolved = [i if i >= 0 else total + i for i in indices]
        self.out_channels = [backbone.out_channels[i] for i in resolved]

    def forward(self, x: torch.Tensor, **kwargs) -> list[torch.Tensor]:
        features = self.backbone(x, **kwargs)
        total = len(features)
        resolved = [i if i >= 0 else total + i for i in self._indices]
        return [features[i] for i in resolved]


def _build_necks(backbone: str, task_type: str) -> list[dict]:
    """Build the neck configuration for the model following GEO-Bench-2 protocol.

    Paper Section 3.3: "For transformer-based models, LearnedFeatureInterpolation,
    available in TerraTorch, is applied to hierarchically structure the encoder output
    before passing it to the UNet."

    This maps to terratorch's neck chain:
      - ReshapeTokensToImage: (B, N, C) → (B, C, H, W) for ViT outputs
      - LearnedInterpolateToPyramidal: creates multi-scale pyramid for UNet decoder
      - PermuteDims: NHWC → NCHW for Swin outputs
      - SelectIndices + AggregateTokens: for classification with ViT
    """
    info = _get_model_info(backbone)
    arch = info.get("arch", "cnn")

    if arch == "cnn":
        return []

    if task_type in ("segmentation", "detection"):
        if arch == "vit":
            return [
                {"name": "ReshapeTokensToImage", "remove_cls_token": True},
                {"name": "LearnedInterpolateToPyramidal"},
            ]
        if arch == "swin":
            # Swin outputs NHWC, permute to NCHW. Already pyramidal, no interpolation needed.
            return [
                {"name": "PermuteDims", "new_order": [0, 3, 1, 2]},
            ]
    else:
        # Classification: aggregate tokens to a single vector
        if arch == "vit":
            return [
                {"name": "AggregateTokens", "pooling": "mean", "drop_cls": True},
            ]
        if arch == "swin":
            # Swin outputs NHWC; permute to NCHW before AggregateTokens (which assumes NCHW)
            return [
                {"name": "PermuteDims", "new_order": [0, 3, 1, 2]},
                {"name": "AggregateTokens", "pooling": "mean"},
            ]

    return []


def _seg_model_args(backbone: str, num_classes: int, pretrained: bool, img_size: int) -> dict:
    """Model args for semantic segmentation: UNet decoder (paper Section 3.3)."""
    info = _get_model_info(backbone)
    decoder_channels = info.get("seg_decoder_channels", [512, 256, 128, 64])

    backbone_module = _build_backbone(backbone, pretrained, "segmentation", img_size)
    necks = _build_necks(backbone, "segmentation")

    model_args: dict[str, Any] = {
        "backbone": backbone_module,
        "decoder": "UNetDecoder",
        "decoder_channels": decoder_channels,
        "necks": necks,
        "num_classes": num_classes,
        "head_dropout": 0.1,
    }
    return model_args


def _cls_model_args(backbone: str, num_classes: int, pretrained: bool, img_size: int) -> dict:
    """Model args for classification: IdentityDecoder + linear head (paper Section 3.3)."""
    backbone_module = _build_backbone(backbone, pretrained, "classification", img_size)
    necks = _build_necks(backbone, "classification")

    model_args: dict[str, Any] = {
        "backbone": backbone_module,
        "decoder": "IdentityDecoder",
        "necks": necks,
        "num_classes": num_classes,
        "head_dropout": 0.1,
    }
    return model_args


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
        model_args = _seg_model_args(backbone, config.num_classes, pretrained, config.img_size)
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
            # GEO-Bench-2 protocol: constant LR (no scheduler).
            # Paper Section 6.3.8-6.3.9 ablated cosine/warmup, found "inconclusive".
        )

    if config.task_type == "classification":
        model_args = _cls_model_args(backbone, config.num_classes, pretrained, config.img_size)
        return MultiLabelClassificationTask(
            model_args=model_args,
            model_factory="EncoderDecoderFactory",
            loss=config.loss,
            freeze_backbone=freeze_backbone,
            freeze_decoder=freeze_decoder,
            lr=lr,
            optimizer="AdamW",
            optimizer_hparams={"weight_decay": weight_decay},
            # GEO-Bench-2 protocol: constant LR (no scheduler).
            # Paper Section 6.3.8-6.3.9 ablated cosine/warmup, found "inconclusive".
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
            # GEO-Bench-2 protocol: constant LR (no scheduler).
            # Paper Section 6.3.8-6.3.9 ablated cosine/warmup, found "inconclusive".
            boxes_field="bbox_xyxy",
            labels_field="label",
        )

    msg = f"Unknown task type: {config.task_type}"
    raise ValueError(msg)
