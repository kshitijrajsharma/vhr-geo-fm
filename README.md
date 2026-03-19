## Efficient Geo FM Models for VHR

### Research Objective

Investigate whether existing geospatial foundation models (GeoFMs) are sufficient for Very High Resolution (VHR) imagery tasks, or if VHR-specific pretraining is required. The primary focus is on commercial VHR satellite and aerial imagery (GSD < 10m), since many current models rely heavily on public Sentinel images which are considered insufficient for this resolution scope.

**Key research questions:**
- Do existing models fail on VHR tasks if they were not pretrained on VHR data?
- Do VHR tasks actually require VHR pretraining, or can models trained on natural images adapt well enough?
- If existing models are insufficient, should we adapt an existing FM or train from scratch efficiently?

**Scope decisions:**
- NOT focusing on VHR SAR
- NOT focusing on temporal VHR
- Primary interest: RGB and RGB+NIR combinations at sub-10m resolution
- Downstream tasks: semantic segmentation, classification, object detection, change detection
- Benchmarking through GEO-Bench-2 and PanGea (PANGAEA) as standardized evaluation frameworks

**Key early findings (from GEO-Bench-2 analysis):**
- Models trained only on Sentinel-1/2 (e.g., TerraMind, Prithvi) fall 30+ points behind on VHR tasks
- Models trained on natural images (ConvNext, DINOv3) adapt surprisingly well to VHR and even surpass some native VHR GeoFMs like Clay and DOFA
- DINOv3-ViT-L-SAT leads because it has seen both satellite and natural images
- However, natural-image models fail when multispectral information is needed
- Existing VHR benchmark datasets are biased towards Europe and North America

**References:**
- [GEO-Bench-2 paper](https://arxiv.org/pdf/2511.15658)
- [PANGAEA paper](https://arxiv.org/abs/2412.04204) · [GitHub](https://github.com/VMarsocci/pangaea-bench)
- [GEO-Bench-2 Leaderboard](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard)
- [Awesome Remote Sensing Foundation Models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)


## Datasets

### GEO-Bench-2 VHR Datasets

Source: [GEO-Bench-2](https://arxiv.org/pdf/2511.15658)

**Classification**

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [TreeSatAI](https://essd.copernicus.org/articles/15/681/2023/) | S2 TS (benchmark); full archive also has S1 SAR + 0.2m aerial CIR | 10 m (S2) / 0.2 m (aerial) | Forestry, 13 tree species | Satellite + Aerial | [Paper](https://essd.copernicus.org/articles/15/681/2023/) · [Zenodo](https://doi.org/10.5281/zenodo.6598390) |

- 50,381 image triplets, Lower Saxony, Germany. Multi-label, 20 species / 15 genera. GEO-Bench-2 uses S2 TS only. CC-BY-4.0.

**Semantic Segmentation**

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [DynamicEarthNet](https://github.com/aysim/dynnet) | Planet PlanetScope (RGB+NIR) | 3 m | LULC change, 7 classes | Satellite VHR | [Paper](https://arxiv.org/abs/2203.12560) · [Data](https://mediatum.ub.tum.de/1650201) · [GitHub](https://github.com/aysim/dynnet) |
| [FLAIR #2](https://ignf.github.io/FLAIR/) | Aerial RGBN (0.2 m) + S2 TS (aux) | 0.2 m (aerial) | Land cover, 13 classes | Aerial VHR | [Project](https://ignf.github.io/FLAIR/) · [Paper](https://arxiv.org/abs/2310.13336) · [GitHub](https://github.com/IGNF/FLAIR-2) · [HuggingFace](https://huggingface.co/datasets/IGNF/FLAIR-1-2) |
| [SpaceNet 2](https://spacenet.ai/spacenet-buildings-dataset-v2/) | WorldView-2/3 (8-band pansharpened) | 0.3 m (WV-3) / 0.5 m (WV-2) | Urban, building footprints, 2 classes | Satellite VHR | [Challenge](https://spacenet.ai/spacenet-buildings-dataset-v2/) · [AWS](https://registry.opendata.aws/spacenet/) · [Paper](https://arxiv.org/abs/1807.01232) |
| [SpaceNet 7](https://spacenet.ai/sn7-challenge/) | Planet Dove (RGB+NIR) | 4 m | Urban, multi-temporal buildings, 2 classes | Satellite VHR | [Challenge](https://spacenet.ai/sn7-challenge/) · [Paper](https://arxiv.org/abs/2102.04420) · [AWS](https://registry.opendata.aws/spacenet/) |

- **DynamicEarthNet**: 75 AOIs, 6 continents, daily imagery (2018-2019), monthly labels. CC-BY-4.0.
- **FLAIR #2**: 20B+ annotated pixels, metropolitan France, 55 spatial domains. Open Licence 2.0.
- **SpaceNet 2**: 5 cities (Rio, Las Vegas, Shanghai, Khartoum, Paris). CC-BY-SA-4.0.
- **SpaceNet 7**: 100+ geographies, 24 monthly time steps, 11M+ building annotations. CC-BY-SA-4.0.

**Object Detection**

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [EverWatch](https://zenodo.org/records/10811969) | UAS/drone aerial RGB | 0.1 m | Ecology, wading birds, 9 classes | Aerial VHR (drone) | [Zenodo](https://zenodo.org/records/10811969) · [Project](https://everglades.weecology.org/data/uav/) · [GitHub](https://github.com/weecology/everwatch-workflow) |
| [NZ Cattle](https://zenodo.org/records/5908869) | Aerial RGB (LINZ) | 0.1 m | Agriculture, cattle detection, 2 classes | Aerial VHR | [Zenodo](https://zenodo.org/records/5908869) · [GitHub](https://github.com/diababuaiadah/Cattle-Detection) |

- **EverWatch**: 5,125 images, Everglades, Florida. CC0.
- **NZ Cattle**: 655 images, 29,803 annotated cows, New Zealand. CC-BY-4.0.

Note: GEO-Bench-2 capability section (Under 10m Resolution) includes SpaceNet2, TreeSatAI, FLAIR2, DynamicEarthNet, SpaceNet7 but excludes EverWatch and NZ Cattle (object detection excluded from capability score).

### PanGea (PANGAEA) VHR Datasets

Source: [PANGAEA](https://arxiv.org/abs/2412.04204) · [GitHub](https://github.com/VMarsocci/pangaea-bench)

**Core VHR Datasets (< 10 m)**

| Dataset | Sensor | Resolution | Domain | Task | Category | Resource |
|---------|--------|------------|--------|------|----------|----------|
| [xView2 (xBD)](https://xview2.org/dataset) | Maxar WorldView-3 (RGB) | 0.3-0.5 m | HADR, building damage, 5 classes | Change Detection | Satellite VHR | [Dataset](https://xview2.org/dataset) · [Paper](https://arxiv.org/abs/1911.09296) |
| [Five Billion Pixels](https://x-ytong.github.io/project/Five-Billion-Pixels.html) | Gaofen-2 (B/G/R/NIR) | 4 m | Land cover / urban, 24 classes | Semantic Seg. | Satellite VHR | [Project](https://x-ytong.github.io/project/Five-Billion-Pixels.html) · [Paper](https://arxiv.org/abs/2209.00727) |
| [DynamicEarthNet](https://github.com/aysim/dynnet) | Planet PlanetScope (RGB+NIR) | 3 m | LULC change, 7 classes | Semantic Seg. | Satellite VHR | [Paper](https://arxiv.org/abs/2203.12560) |
| [SpaceNet 7](https://spacenet.ai/sn7-challenge/) | Planet Dove (RGB+NIR) | 4 m | Urban, building tracking, 2 classes | Change Detection | Satellite VHR | [Challenge](https://spacenet.ai/sn7-challenge/) |

**Community-Contributed VHR Datasets**

| Dataset | Sensor | Resolution | Domain | Task | Category | Resource |
|---------|--------|------------|--------|------|----------|----------|
| [ISPRS Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | Aerial camera (NIR/R/G/B + DSM) | 0.05 m | Urban land cover, 6 classes | Semantic Seg. | Aerial VHR | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) |
| [Open-Canopy](https://huggingface.co/datasets/AI4Forest/Open-Canopy) | SPOT 6-7 + aerial LiDAR | 1.5 m | Forestry, canopy height | Regression | Satellite VHR | [HuggingFace](https://huggingface.co/datasets/AI4Forest/Open-Canopy) · [Paper](https://arxiv.org/abs/2407.09392) |

### Additional VHR Datasets (Not in GEO-Bench-2 or PanGea core)

From Pierre's GeoFM survey, filtered to GSD <= 10m, publicly licensed:

| Dataset | Task | Domain | Sensor | Resolution | Resource |
|---------|------|--------|--------|------------|----------|
| [Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | Semantic Seg. | Urban | Aerial (IR, R, G, DSM) | 0.09 m | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) |
| [DOTA V1.0](https://captain-whu.github.io/DOTA/) | Object Detection | Aerial Objects | Google Earth, GF-2, JL-1 | 0.1-4.5 m | [Project](https://captain-whu.github.io/DOTA/) |
| [xView2](https://xview2.org/dataset) | Change Detection | Disaster / HADR | Maxar (WV-3) | 0.3 m | [Dataset](https://xview2.org/dataset) |
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | Change Detection | Building Change | Google Earth | 0.5 m | [Project](https://justchenhao.github.io/LEVIR/) |
| [UCMerced](http://weegee.vision.ucmerced.edu/datasets/landuse.html) | Classification | Land Use | Aerial RGB | 0.3 m | [Dataset](http://weegee.vision.ucmerced.edu/datasets/landuse.html) |
| [fMoW](https://github.com/fMoW/dataset) | Classification | Land Use (temporal) | QuickBird, GeoEye, WV | 0.5 m | [GitHub](https://github.com/fMoW/dataset) |
| [NWPU-RESISC45](https://arxiv.org/abs/1703.00121) | Classification | Scene | Google Earth | 0.2-30 m | [Paper](https://arxiv.org/abs/1703.00121) |

6 of these 15 total VHR datasets are not included in either PanGea or GEO-Bench-2.

### Dataset Overlap Between Benchmarks

| Dataset | GEO-Bench-2 Task | PanGea Task |
|---------|------------------|-------------|
| DynamicEarthNet | Semantic Segmentation | Semantic Segmentation |
| SpaceNet 7 | Semantic Segmentation | Change Detection |

### Dataset Summary by Category

**Satellite VHR** (commercial satellite sensors, GSD < 10m)

| Dataset | Sensor | Resolution | Benchmark |
|---------|--------|------------|-----------|
| SpaceNet 2 | WorldView-2/3 | 0.3-0.5 m | GEO-Bench-2 |
| xView2 (xBD) | Maxar WorldView-3 | 0.3-0.5 m | PanGea |
| Open-Canopy | SPOT 6-7 | 1.5 m | PanGea (community) |
| DynamicEarthNet | Planet PlanetScope | 3 m | GEO-Bench-2, PanGea |
| SpaceNet 7 | Planet Dove | 4 m | GEO-Bench-2, PanGea |
| Five Billion Pixels | Gaofen-2 | 4 m | PanGea |

**Aerial VHR** (airborne / drone sensors)

| Dataset | Sensor | Resolution | Benchmark |
|---------|--------|------------|-----------|
| ISPRS Potsdam | Aerial camera (NIRGB+DSM) | 0.05 m | PanGea (community) |
| EverWatch | UAS/drone RGB | 0.1 m | GEO-Bench-2 |
| NZ Cattle | Aerial RGB (LINZ) | 0.1 m | GEO-Bench-2 |
| FLAIR #2 | Aerial RGBN + S2 (aux) | 0.2 m | GEO-Bench-2 |

**Mixed (Satellite + Aerial)**

| Dataset | Sensor | Resolution | Benchmark |
|---------|--------|------------|-----------|
| TreeSatAI | Sentinel-1/2 + aerial CIR | 10 m (S2) / 0.2 m (aerial) | GEO-Bench-2 |


## Models

Models evaluated in GEO-Bench-2, categorized by pretraining data type. Source: [GEO-Bench-2 Table 4](https://arxiv.org/pdf/2511.15658)

### Category 1: Native VHR GeoFM

Models pretrained on VHR aerial/satellite images (GSD < 10m). These have seen high-resolution imagery during pretraining.

| Model | Backbone | Params | Technique | Pretraining Data | Resolution | Pretrain Samples | License |
|-------|----------|--------|-----------|------------------|------------|------------------|---------|
| DINOv3-ViT-L-SAT | ViT | 300M | Distillation | Maxar RGB | 0.6 m | 493M | DINO V3 |
| DOFA-ViT 300M | ViT | 300M | MAE | S1, S2, EnbMap, Gaofen, Landsat | 1-30 m | 8M | CC-BY-4.0 |
| Clay-V1 ViT-B | ViT | 86M | MAE | Landsat 8/9, S1, S2, NAIP, LINZ, MODIS | 1-30 m | 70M | Apache 2.0 |
| Satlas-SwinB-NAIP | Swin | 88M | Supervised | NAIP (USGS) | 1 m | NA | ODC-BY |

### Category 2: General CV FM

Models pretrained on natural / web images only (ImageNet, LVD). No aerial or satellite images in pretraining.

| Model | Backbone | Params | Technique | Pretraining Data | License |
|-------|----------|--------|-----------|------------------|---------|
| ConvNext-XLarge-ImageNet | ConvNext | 390M | Supervised | ImageNet-22k | Apache 2.0 |
| DINOv3-ConvNext-Large-WEB | ConvNext | 230M | Distillation | LVD-1689M | DINO V3 |
| ConvNext-Large-ImageNet | ConvNext | 230M | Supervised | ImageNet-22k | Apache 2.0 |
| ResNet50-ImageNet | ResNet-50 | 25M | Supervised | ImageNet-22k | Apache 2.0 |

### Category 3: GeoFM (Low-Res)

Models pretrained on publicly available low-resolution satellite data (>= 10m), primarily Sentinel-1/2.

| Model | Backbone | Params | Technique | Pretraining Data | Resolution | Pretrain Samples | License |
|-------|----------|--------|-----------|------------------|------------|------------------|---------|
| TerraMind-V1-Large | ViT | 300M | Correlation | S1, S2, LULC, DEM, NDVI | 10 m | 9M | Apache 2.0 |
| TerraMind-V1-Base | ViT | 86M | Correlation | S1, S2, LULC, DEM, NDVI | 10 m | 9M | Apache 2.0 |
| Prithvi-EO-V2-600M-TL | ViT | 600M | MAE | HLS (Harmonized) | 30 m | 4.2M | Apache 2.0 |
| Prithvi-EO-V2-300M-TL | ViT | 300M | MAE | HLS | 30 m | 4.2M | Apache 2.0 |
| Satlas-SwinB-Sentinel2 | Swin | 88M | Supervised | Sentinel-2 | 10 m | NA | ODC-BY |
| Satlas-Swin-100M | Swin | 100M | Supervised | Sentinel-2 | 10 m | NA | ODC-BY |
| ResNet50-DeCUR | ResNet-50 | 25M | Contrastive | Sentinel-2 | 10 m | 1M | Apache 2.0 |

### Model Performance on VHR Datasets (Under 10m Resolution)

Rankings from GEO-Bench-2 after full fine-tuning (normalized IQM scores). Includes object detection datasets (EverWatch, NZCattle) via patched evaluation.

| Rank | Model | Category | Params | Score |
|------|-------|----------|--------|-------|
| 1 | ConvNext-XLarge-ImageNet | CV-General-FM | 390M | 86.9 +/- 0.1 |
| 2 | DINOv3-ConvNext-Large-WEB | CV-General-FM | 230M | 80.1 +/- 0.1 |
| 3 | DINOv3-ViT-L-SAT | Native-VHR-GeoFM | 300M | 77.8 +/- 0.2 |
| 4 | ConvNext-Large-ImageNet | CV-General-FM | 230M | 77.0 +/- 0.1 |
| 5 | DOFA-ViT 300M | Native-VHR-GeoFM | 300M | 62.7 +/- 0.2 |
| 6 | Clay-V1 ViT-B | Native-VHR-GeoFM | 100M | 60.7 +/- 0.2 |
| 7 | TerraMind-V1-Large | GeoFM-LowRes | 300M | 55.7 +/- 0.1 |
| 8 | Satlas-SwinB-NAIP | Native-VHR-GeoFM | 100M | 53.4 +/- 0.1 |
| 9 | Satlas-Swin-100M | GeoFM-LowRes | 100M | 50.8 +/- 0.2 |
| 10 | Prithvi-EO-V2-600M-TL | GeoFM-LowRes | 600M | 47.8 +/- 0.1 |
| 11 | Prithvi-EO-V2-300M-TL | GeoFM-LowRes | 300M | 43.8 +/- 0.2 |
| 12 | TerraMind-V1-Base | GeoFM-LowRes | 100M | 42.1 +/- 0.1 |
| 13 | ResNet50-ImageNet | CV-General-FM | 25M | 36.1 +/- 0.1 |
| 14 | ResNet50-DeCUR | GeoFM-LowRes | 25M | 24.3 +/- 0.1 |

**Key observations:**
- CV-General-FM models (ConvNext, DINOv3-WEB) top the VHR leaderboard, outperforming native VHR GeoFMs
- Native-VHR-GeoFM models (DOFA, Clay) perform moderately, with DINOv3-ViT-L-SAT (seen both satellite + natural images) being the strongest
- GeoFM-LowRes models (TerraMind, Prithvi) fall 30+ points behind, confirming that Sentinel-only pretraining is insufficient for VHR
- Natural-image pretraining transfers well to VHR RGB tasks but fails on multispectral tasks
