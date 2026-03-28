"""Tests for model registry, neck configuration, and task factory logic."""

import json

import pytest

from eval.models import (
    _SKIP_OUT_INDICES,
    MODEL_REGISTRY_PATH,
    _backbone_kwargs,
    _info,
    _load_registry,
    _necks,
    _out_indices_key,
)


class TestRegistry:
    def test_registry_file_exists(self):
        assert MODEL_REGISTRY_PATH.exists()

    def test_registry_valid_json(self):
        data = json.loads(MODEL_REGISTRY_PATH.read_text())
        assert "categories" in data

    def test_all_models_have_required_fields(self):
        registry = _load_registry()
        for backbone, info in registry.items():
            assert "arch" in info, f"{backbone} missing arch"
            assert info["arch"] in ("vit", "swin", "cnn"), (
                f"{backbone} invalid arch: {info['arch']}"
            )
            assert "seg_out_indices" in info, f"{backbone} missing seg_out_indices"
            assert "cls_out_indices" in info, f"{backbone} missing cls_out_indices"

    def test_three_categories(self):
        data = json.loads(MODEL_REGISTRY_PATH.read_text())
        assert set(data["categories"].keys()) == {"VHR-FM", "CV-FM", "LowRes-FM"}

    def test_11_models_total(self):
        registry = _load_registry()
        assert len(registry) == 11

    def test_info_unknown_backbone_returns_defaults(self):
        info = _info("nonexistent_backbone")
        assert info["arch"] == "cnn"
        assert "seg_out_indices" in info


class TestOutIndicesKey:
    @pytest.mark.parametrize(
        "task,expected",
        [
            ("segmentation", "seg_out_indices"),
            ("detection", "seg_out_indices"),
            ("classification", "cls_out_indices"),
        ],
    )
    def test_mapping(self, task, expected):
        assert _out_indices_key(task) == expected


class TestNecks:
    """Verify neck chains match GEO-Bench-2 Section 3.3."""

    def test_cnn_seg_no_necks(self):
        assert _necks("timm_resnet50", "segmentation") == []

    def test_cnn_cls_no_necks(self):
        assert _necks("timm_resnet50", "classification") == []

    def test_vit_seg_reshape_and_interpolate(self):
        necks = _necks("terratorch_dofa_large_patch16_224", "segmentation")
        assert len(necks) == 2
        assert necks[0]["name"] == "ReshapeTokensToImage"
        assert necks[1]["name"] == "LearnedInterpolateToPyramidal"

    def test_vit_cls_aggregate(self):
        necks = _necks("terratorch_dofa_large_patch16_224", "classification")
        assert len(necks) == 1
        assert necks[0]["name"] == "AggregateTokens"
        assert necks[0]["pooling"] == "mean"

    def test_swin_seg_permute(self):
        necks = _necks("terratorch_satlas_swin_b_naip_si_rgb", "segmentation")
        assert len(necks) == 1
        assert necks[0]["name"] == "PermuteDims"

    def test_swin_cls_permute_and_aggregate(self):
        necks = _necks("terratorch_satlas_swin_b_naip_si_rgb", "classification")
        assert len(necks) == 2
        assert necks[0]["name"] == "PermuteDims"
        assert necks[1]["name"] == "AggregateTokens"

    def test_detection_uses_spatial_necks(self):
        # Detection should use same necks as segmentation (spatial output)
        for backbone in [
            "terratorch_dofa_large_patch16_224",
            "terratorch_satlas_swin_b_naip_si_rgb",
        ]:
            assert _necks(backbone, "detection") == _necks(backbone, "segmentation")


class TestBackboneKwargs:
    def test_skip_out_indices_not_in_kwargs(self):
        for backbone in _SKIP_OUT_INDICES:
            bk = _backbone_kwargs(
                backbone, pretrained=False, task_type="segmentation", img_size=512
            )
            assert "out_indices" not in bk

    def test_regular_backbone_has_out_indices(self):
        bk = _backbone_kwargs(
            "timm_resnet50", pretrained=False, task_type="segmentation", img_size=512
        )
        assert "out_indices" in bk
        assert bk["out_indices"] == [1, 2, 3, 4]

    def test_dofa_img_size_padded(self):
        bk = _backbone_kwargs(
            "terratorch_dofa_large_patch16_224",
            pretrained=False,
            task_type="segmentation",
            img_size=512,
        )
        # ceil(512 / (16*2)) * (16*2) = 512
        assert bk["img_size"] == 512

    def test_dofa_img_size_non_aligned(self):
        bk = _backbone_kwargs(
            "terratorch_dofa_large_patch16_224",
            pretrained=False,
            task_type="segmentation",
            img_size=300,
        )
        # ceil(300 / 32) * 32 = 320
        assert bk["img_size"] == 320

    def test_clay_img_size_padded(self):
        bk = _backbone_kwargs(
            "timm_clay_v1_base",
            pretrained=False,
            task_type="segmentation",
            img_size=304,
        )
        # ps=8, ceil(304 / 16) * 16 = 304
        assert bk["img_size"] == 304

    def test_dinov3_ckpt_from_env(self, monkeypatch):
        monkeypatch.setenv("DINOV3_VITL16_CKPT", "/path/to/ckpt.pth")
        bk = _backbone_kwargs(
            "terratorch_dinov3_vitl16",
            pretrained=False,
            task_type="segmentation",
            img_size=512,
        )
        assert bk["ckpt_path"] == "/path/to/ckpt.pth"

    def test_dinov3_no_ckpt_without_env(self, monkeypatch):
        monkeypatch.delenv("DINOV3_VITL16_CKPT", raising=False)
        bk = _backbone_kwargs(
            "terratorch_dinov3_vitl16",
            pretrained=False,
            task_type="segmentation",
            img_size=512,
        )
        assert "ckpt_path" not in bk
