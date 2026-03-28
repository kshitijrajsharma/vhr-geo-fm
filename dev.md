# Development Guide

## Setup

```sh
uv sync              # install deps
just download        # download GEO-Bench-2 VHR datasets to ./data/
```

## Running Experiments

```sh
just frozen                                               # frozen, all seg/cls datasets
just finetune                                             # finetune, all datasets incl. detection
just frozen model.backbone=terratorch_dofa_large_patch16_224  # specific model
just finetune data.data_pct=5                             # finetune, 5% data fraction
just run training.mode=frozen data.data_pct=10            # raw hydra overrides
```

Each recipe maps to a Hydra config preset. Extra overrides are passed through.

## Config Presets

```
conf/
  config.yaml     Base defaults (full protocol: 16 HPO trials, 5 seeds, 50 epochs)
  frozen.yaml     Frozen encoder — seg + cls datasets only
  finetune.yaml   Full finetune — all 7 datasets incl. detection
  verify.yaml     Quick verify — 1% data, 1 trial, 1 seed, 1 epoch
```

Presets inherit from `config.yaml` via Hydra defaults and only override what differs.

## Architecture

```
eval/
  run.py          Entry point: HPO -> seed evaluation -> scoring
  models.py       Task factory: backbone + neck + decoder via terratorch
  datasets.py     Dataset registry (7 VHR datasets) + datamodule factory
  scoring.py      Score normalization, IQM, bootstrap, CSV export
  subsample.py    Data-fraction wrapper (train set only)
models.json       Model registry: backbone kwargs, arch type, out_indices
```

## Pipeline

```
HPO (Optuna TPE)  ->  Seed Evaluation  ->  Scoring
  lr: [1e-6, 1e-3]     5 seeds             min-max normalize
  bs: {8, 16, 32}      EarlyStopping(10)   IQM per dataset
  16 trials             max 50 epochs       bootstrap aggregate
```

## Model Categories

| Category  | Models                                           | Pretraining       |
|-----------|--------------------------------------------------|-------------------|
| VHR-FM    | DINOv3-ViT-L-SAT, DOFA, Clay, Satlas-NAIP       | VHR imagery (<10m)|
| CV-FM     | ConvNext-XL, DINOv3-ConvNext-WEB, ResNet50       | ImageNet/web      |
| LowRes-FM | TerraMind, Prithvi, Satlas-S2, DeCUR            | Sentinel (>=10m)  |

## Neck Configuration (GEO-Bench-2 Section 3.3)

| Arch | Segmentation                                              | Classification            |
|------|-----------------------------------------------------------|---------------------------|
| CNN  | direct to UNet                                            | direct to linear head     |
| ViT  | ReshapeTokensToImage -> LearnedInterpolateToPyramidal    | AggregateTokens (mean)    |
| Swin | PermuteDims (NHWC->NCHW)                                 | PermuteDims -> Aggregate  |

## DINOv3 Setup

Both DINOv3 models require downloading gated checkpoints:

```sh
# Direct URLs (no auth needed)
export DINOV3_VITL16_CKPT="https://dl.fbaipublicfiles.com/dinov3/dinov3_vitl16/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth"
export DINOV3_CONVNEXT_CKPT="https://dl.fbaipublicfiles.com/dinov3/dinov3_convnext_large/dinov3_convnext_large_pretrain_lvd1689m-61fa432d.pth"
```

## Testing & CI

```sh
just test                                        # run pytest suite
just smoke                                       # quick: ResNet50 x spacenet2 (1% data, 1 epoch)
just smoke model.backbone=timm_clay_v1_base      # quick: specific model
just verify                                      # full: 9 models x 5 datasets
just ci                                          # lint + typecheck + test
```

Pre-commit hooks (ruff + ty) run on every commit. CI runs lint, typecheck, and pytest on push/PR to master.

## Key Protocol Decisions

- **Constant LR** -- paper ablated cosine/warmup, found inconclusive (Sections 6.3.8-9)
- **UNet** for segmentation, **linear head** for classification, **Faster R-CNN** for detection
- **Frozen mode** = backbone frozen, decoder trained; detection skipped in frozen mode
- **Data fractions** (5%, 10%) are our addition -- not in the original GEO-Bench-2 protocol

## Output

Results are saved to `results/<backbone>/<mode>/<pct>pct/`:
- `results.csv` -- per-seed metrics + hyperparams + system metadata
- `hpo_params.json` -- best HPO parameters per dataset
