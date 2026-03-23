## Efficient Geo FM Models for VHR

### Research Objective

Investigate whether existing geospatial foundation models (GeoFMs) are sufficient for Very High Resolution (VHR) imagery tasks, or if VHR-specific pretraining is required. The primary focus is on commercial VHR satellite and aerial imagery (GSD < 10m), since many current models rely heavily on public Sentinel images Hypothesis is :  they might be insufficient for this resolution scope.

**Key research questions:**

- Do existing models fail on VHR tasks if they were not pretrained on VHR data?
- Do VHR tasks actually require VHR pretraining, or can models trained on natural/public low res sentinel images adapt well enough?
- If existing models are insufficient, should we adapt an existing FM or train from scratch efficiently?

**Scope decisions:**

- NOT focusing on VHR SAR
- NOT focusing on temporal VHR ( As our theme is disaster getting temporal data on VHR might not be feasible )
- Primary interest: RGB and RGB+NIR combinations at sub-10m resolution
- Downstream tasks: semantic segmentation, classification, object detection, change detection
- Benchmarking through GEO-Bench-2 and PanGea (PANGAEA) as standardized evaluation frameworks

**Key early findings:** Sentinel-only models lag significantly on VHR; natural-image models adapt surprisingly well to VHR RGB but fail on multispectral; VHR pretraining matters most when encoders are frozen. See [fine-tuned](#model-performance-on-vhr-datasets-fine-tuned) and [frozen encoder](#model-performance-on-vhr-datasets-frozen-encoder) results for details.

**References:**

- [GEO-Bench-2 paper](https://arxiv.org/pdf/2511.15658)
- [PANGAEA paper](https://arxiv.org/abs/2412.04204) · [GitHub](https://github.com/VMarsocci/pangaea-bench)
- [GEO-Bench-2 Leaderboard](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard)
- [Awesome Remote Sensing Foundation Models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)

### Definitions

**Satellite VHR**: Imagery acquired from orbital satellite platforms at sub-10m ground sampling distance (GSD). Modern commercial VHR satellites reach 30cm panchromatic natively, with pansharpened products down to 15cm. Key commercial VHR satellite sensors:

| Sensor | Operator | Panchromatic | Multispectral | Bands | Notes |
|--------|----------|-------------|---------------|-------|-------|
| WorldView-3 | [Maxar](https://earth.esa.int/eogateway/missions/worldview-3) | 0.31 m | 1.24 m | 8 VNIR + 8 SWIR | Highest spectral diversity |
| WorldView Legion | [Maxar](https://resources.maxar.com/data-sheets/worldview-legion) | 0.29 m | 1.16 m | VNIR | Constellation for high revisit |
| Pleiades Neo 3/4 | [Airbus](https://space-solutions.airbus.com/imagery/our-optical-and-radar-satellite-imagery/pleiades-neo/) | 0.30 m | 1.2 m | 6 (incl. Deep Blue, Red-Edge) | HD15 enhanced product |
| WorldView-2 | [Maxar](https://earth.esa.int/eogateway/missions/worldview-2) | 0.46 m | 1.84 m | 8 VNIR | First 8-band VHR satellite |
| GeoEye-1 | [Maxar](https://earth.esa.int/eogateway/missions/geoeye-1) | 0.41 m | 1.64 m | 4 VNIR | High geolocation accuracy |
| Pleiades-1A/1B | [Airbus](https://earth.esa.int/eogateway/missions/pleiades) | 0.50 m | 2.0 m | 4 (RGB+NIR) | Twin constellation |
| SkySat | [Planet](https://docs.planet.com/data/imagery/skysat/) | 0.50 m | 0.50 m (ortho) | 4 MS + 1 Pan | Super-resolution ortho product |
| SPOT 6/7 | [Airbus](https://earth.esa.int/eogateway/missions/spot-6) | 1.5 m | 6.0 m | 4 (RGB+NIR) | Wide swath (60km) |
| Gaofen-2 | [CNSA](https://earth.esa.int/eogateway/missions/gaofen-2) | 0.81 m | 3.24 m | 4 (RGB+NIR) | Chinese civilian VHR |
| PlanetScope (Dove) | [Planet](https://docs.planet.com/data/imagery/planetscope/) | N/A | 3-5 m | 4-8 bands | Daily global coverage |

Satellite VHR images cover large areas but may have atmospheric distortion, off-nadir angles, and varying revisit depending on constellation. 

**Aerial VHR**: Imagery acquired from airborne platforms (manned aircraft, drones/UAS) at very fine resolution, typically sub-1m. Examples: NAIP (~1m, US national coverage), LINZ (New Zealand aerial surveys), IGN aerial campaigns (0.2m, France), drone/UAS imagery (0.1m or finer). These images are captured under controlled conditions (ideal weather, nadir view) and typically offer only RGB or RGB+NIR bands with limited spectral diversity compared to satellite sensors.

## Datasets

### GEO-Bench-2 VHR Datasets

Source: [GEO-Bench-2](https://arxiv.org/pdf/2511.15658)

**Classification**

| Dataset | Sensor | Resolution | Domain | Category | License |
|---------|--------|------------|--------|----------|---------|
| [TreeSatAI](https://essd.copernicus.org/articles/15/681/2023/) | S2 TS (benchmark); full archive also has S1 SAR + 0.2m aerial CIR | 10 m (S2) / 0.2 m (aerial) | Forestry, 13 tree species | Satellite + Aerial | CC-BY-4.0 |

- 50,381 image triplets, Lower Saxony, Germany. Multi-label, 20 species / 15 genera. GEO-Bench-2 uses S2 TS only. [Zenodo](https://doi.org/10.5281/zenodo.6598390)

**Semantic Segmentation**

| Dataset | Sensor | Resolution | Domain | Category | License |
|---------|--------|------------|--------|----------|---------|
| [DynamicEarthNet](https://arxiv.org/abs/2203.12560) | Planet PlanetScope (RGB+NIR) | 3 m | LULC change, 7 classes | Satellite VHR | CC-BY-4.0 |
| [FLAIR #2](https://arxiv.org/abs/2310.13336) | Aerial RGBN (0.2 m) + S2 TS (aux) | 0.2 m (aerial) | Land cover, 13 classes | Aerial VHR | Open License 2.0 |
| [SpaceNet 2](https://arxiv.org/abs/1807.01232) | WorldView-2/3 (8-band pansharpened) | 0.3 m (WV-3) / 0.5 m (WV-2) | Urban, building footprints, 2 classes | Satellite VHR | CC-BY-SA-4.0 |
| [SpaceNet 7](https://arxiv.org/abs/2102.04420) | Planet Dove (RGB+NIR) | 4 m | Urban, multi-temporal buildings, 2 classes | Satellite VHR | CC-BY-SA-4.0 |

- **DynamicEarthNet**: 75 AOIs, 6 continents, daily imagery (2018-2019), monthly labels. [Data](https://mediatum.ub.tum.de/1650201)
- **FLAIR #2**: 20B+ annotated pixels, metropolitan France, 55 spatial domains. [HuggingFace](https://huggingface.co/datasets/IGNF/FLAIR-1-2)
- **SpaceNet 2**: 5 cities (Rio, Las Vegas, Shanghai, Khartoum, Paris). [AWS](https://registry.opendata.aws/spacenet/)
- **SpaceNet 7**: 100+ geographies, 24 monthly time steps, 11M+ building annotations. [AWS](https://registry.opendata.aws/spacenet/)

**Object Detection**

| Dataset | Sensor | Resolution | Domain | Category | License |
|---------|--------|------------|--------|----------|---------|
| [EverWatch](https://zenodo.org/records/10811969) | UAS/drone aerial RGB | 0.1 m | Ecology, wading birds, 9 classes | Aerial VHR (drone) | CC0 |
| [NZ Cattle](https://zenodo.org/records/5908869) | Aerial RGB (LINZ) | 0.1 m | Agriculture, cattle detection, 2 classes | Aerial VHR | CC-BY-4.0 |

- **EverWatch**: 5,125 images, Everglades, Florida. [GitHub](https://github.com/weecology/everwatch-workflow)
- **NZ Cattle**: 655 images, 29,803 annotated cows, New Zealand.

### PanGea (PANGAEA) VHR Datasets

Not used in the experiments yet but listing them out : 

Source: [PANGAEA](https://arxiv.org/abs/2412.04204) · [GitHub](https://github.com/VMarsocci/pangaea-bench)

**Core VHR Datasets (< 10 m)**

| Dataset | Sensor | Resolution | Domain | Category | License |
|---------|--------|------------|--------|----------|---------|
| [xView2 (xBD)](https://arxiv.org/abs/1911.09296) | Maxar WorldView-3 (RGB) | 0.3-0.5 m | HADR, building damage, 5 classes | Satellite VHR | CC-BY-NC-SA-4.0 |
| [Five Billion Pixels](https://arxiv.org/abs/2209.00727) | Gaofen-2 (B/G/R/NIR) | 4 m | Land cover / urban, 24 classes | Satellite VHR | Open Source |

Also includes DynamicEarthNet and SpaceNet 7 (see GEO-Bench-2 above).

- **xView2 (xBD)**: 22,068 images across 15 countries, 6 disaster types, 850,736 building polygons. 1024x1024 px pairs.
- **Five Billion Pixels**: 150 Gaofen-2 images (6800x7200 px) covering 60+ districts in China, >50,000 km2. Cropped to 520x520 tiles in PanGea.

**Community-Contributed VHR Datasets**

| Dataset | Sensor | Resolution | Domain | Category | License |
|---------|--------|------------|--------|----------|---------|
| [ISPRS Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | Aerial camera (NIR/R/G/B + DSM) | 0.05 m | Urban land cover, 6 classes | Aerial VHR | Free (research) |
| [Open-Canopy](https://arxiv.org/abs/2407.09392) | SPOT 6-7 + aerial LiDAR | 1.5 m | Forestry, canopy height | Satellite VHR | Open License 2.0 |

- **ISPRS Potsdam**: 38 tiles (6000x6000 px) covering 3.42 km2 in Potsdam, Germany. 24 annotated, 14 held-out.
- **Open-Canopy**: ~360 GB, >87,000 km2 of France. Continuous canopy height regression target. [HuggingFace](https://huggingface.co/datasets/AI4Forest/Open-Canopy)

### Additional VHR Datasets (Not in GEO-Bench-2 or PanGea core)

( Not used here but listing them ) 

From Pierre's GeoFM survey, filtered to GSD <= 10m, publicly licensed. 6 of 15 total surveyed VHR datasets are not included in either PanGea or GEO-Bench-2. ( However their license is not truely opensource )

| Dataset | Sensor | Resolution | Domain | Category | License |
|---------|--------|------------|--------|----------|---------|
| [Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | Aerial (IR, R, G, DSM) | 0.09 m | Urban, semantic seg., 6 classes | Aerial VHR | Free (research) |
| [DOTA V1.0](https://captain-whu.github.io/DOTA/) | Google Earth, GF-2, JL-1 | 0.1-4.5 m | Aerial objects, object detection, 15 classes | Satellite VHR + Aerial VHR | Not specified |
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | Google Earth | 0.5 m | Building change, change detection | Satellite VHR | Academic only |
| [UCMerced](https://vision.ucmerced.edu/datasets/) | USGS aerial RGB | 0.3 m | Land use, classification, 21 classes | Aerial VHR | Public domain |
| [fMoW](https://github.com/fMoW/dataset) | QuickBird, GeoEye, WorldView | 0.5 m | Land use (temporal), classification, 62 classes | Satellite VHR | Custom (fMoW) |
| [NWPU-RESISC45](https://arxiv.org/abs/1703.00121) | Google Earth | 0.2-30 m | Scene, classification, 45 classes | Satellite VHR | Not specified |

## Models

Models evaluated in GEO-Bench-2, categorized by pretraining data type. Source: [GEO-Bench-2 Table 4](https://arxiv.org/pdf/2511.15658)

### Category 1: Native VHR GeoFM

Models pretrained on VHR imagery (GSD < 10m), including satellite and/or aerial sources.

| Model | Backbone | Params | Technique | Pretraining Data | Resolution | Pretrain Samples | VHR Pretraining | License |
|-------|----------|--------|-----------|------------------|------------|------------------|-----------------|---------|
| [DINOv3-ViT-L-SAT](https://arxiv.org/abs/2508.10104) | ViT | 300M | Distillation | Maxar RGB | 0.6 m | 493M | Satellite VHR (Maxar) | DINO V3 |
| [DOFA-ViT 300M](https://arxiv.org/abs/2403.15356) | ViT | 300M | MAE | S1, S2, Gaofen, NAIP, EnMAP | 1-30 m | 8M | Satellite VHR (Gaofen) + Aerial VHR (NAIP) | CC-BY-4.0 |
| [Clay-V1 ViT-B](https://clay-foundation.github.io/model/) | ViT | 86M | MAE | Landsat 8/9, S1, S2, NAIP, LINZ, MODIS | 1-30 m | 70M | Aerial VHR (NAIP ~1m, LINZ) | Apache 2.0 |
| [Satlas-SwinB-NAIP](https://arxiv.org/abs/2211.15660) | Swin | 88M | Supervised | NAIP (USGS) | 1 m | NA | Aerial VHR (NAIP ~1m) | ODC-BY |

### Category 2: General CV FM

Models pretrained on natural/web images only (ImageNet, LVD). No aerial or satellite imagery in pretraining.

| Model | Backbone | Params | Technique | Pretraining Data | Resolution | Pretrain Samples | VHR Pretraining | License |
|-------|----------|--------|-----------|------------------|------------|------------------|-----------------|---------|
| [ConvNext-XLarge-ImageNet](https://arxiv.org/abs/2201.03545) | ConvNext | 390M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |
| [DINOv3-ConvNext-Large-WEB](https://arxiv.org/abs/2508.10104) | ConvNext | 230M | Distillation | LVD-1689M | NA | 1689M | None | DINO V3 |
| [ConvNext-Large-ImageNet](https://arxiv.org/abs/2201.03545) | ConvNext | 230M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |
| [ResNet50-ImageNet](https://arxiv.org/abs/1512.03385) | ResNet-50 | 25M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |

### Category 3: GeoFM (Low-Res)

Models pretrained on publicly available low-resolution satellite data (>= 10m), primarily Sentinel-1/2.

| Model | Backbone | Params | Technique | Pretraining Data | Resolution | Pretrain Samples | VHR Pretraining | License |
|-------|----------|--------|-----------|------------------|------------|------------------|-----------------|---------|
| [TerraMind-V1-Large](https://arxiv.org/abs/2504.11171) | ViT | 300M | Correlation | S1, S2, LULC, DEM, NDVI | 10 m | 9M | None | Apache 2.0 |
| [TerraMind-V1-Base](https://arxiv.org/abs/2504.11171) | ViT | 86M | Correlation | S1, S2, LULC, DEM, NDVI | 10 m | 9M | None | Apache 2.0 |
| [Prithvi-EO-V2-600M-TL](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) | ViT | 600M | MAE | HLS (Harmonized) | 30 m | 4.2M | None | Apache 2.0 |
| [Prithvi-EO-V2-300M-TL](https://github.com/NASA-IMPACT/Prithvi-EO-2.0) | ViT | 300M | MAE | HLS | 30 m | 4.2M | None | Apache 2.0 |
| [Satlas-SwinB-Sentinel2](https://arxiv.org/abs/2211.15660) | Swin | 88M | Supervised | Sentinel-2 | 10 m | NA | None | ODC-BY |
| [Satlas-Swin-100M](https://arxiv.org/abs/2211.15660) | Swin | 100M | Supervised | Sentinel-2 | 10 m | NA | None | ODC-BY |

### Scoring Methodology

Performance scores are computed using a patched version of the [GEO-Bench-2 Leaderboard](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard) tooling. The script ([scripts/patch_geo_bench.py](scripts/patch_geo_bench.py)) extends the official leaderboard pipeline to include object detection datasets (EverWatch, NZCattle) in the "Under 10m Resolution" capability score.

**How scores are computed (following GEO-Bench-2 paper Section 3.4):**

1. **Raw metrics**: Each model submission contains per-seed results across multiple training runs (stored as CSVs in the leaderboard repo)
2. **Normalization**: Raw metrics are normalized to a 0–100 scale using the leaderboard's [normalizer.json](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard/blob/main/utils/normalizer/leaderboard_combined/normalizer.json), which maps dataset-specific metrics (mIoU, accuracy, mAP, etc.) to comparable scales
3. **Per-dataset scores**: Computed as the interquartile mean (IQM) of the normalized metric across bootstrap samples, with error from trimmed SEM
4. **Aggregate score**: The "Under 10M Resolution" capability score is a bootstrapped IQM aggregated across all constituent datasets, with error bars from standard error of the mean (SEM)
5. **Frozen vs fine-tuned**: The script filters submissions by training mode (`--frozen` for frozen backbone, default for full fine-tuning)
6. **Temporal data handling**: Several VHR datasets are temporal (DynamicEarthNet, SpaceNet7, TreeSatAI). For non-temporal models, each timestamp is encoded separately and outputs are averaged before the decoder. Models with native temporal support (e.g., Prithvi, T=4) process multiple timestamps directly. The paper's ablation found that dropping temporal info caused the largest ranking perturbations among all ablations (GEO-Bench-2 Section 4.3).

**Flags:**

- `--extra`: Includes object detection datasets (EverWatch, NZCattle) in the capability score (used for fine-tuned results below)
- `--frozen`: Evaluates frozen backbone submissions instead of fully fine-tuned (object detection results are not available in frozen mode)
- `--out <path>`: Exports results as CSV

Results were accessed & generated on 2026-03-15 (fine-tuned) and 2026-03-20 (frozen). Raw data: [data/](data/)

### Model Performance on VHR Datasets: Fine-tuned

Normalized IQM scores from GEO-Bench-2 after full fine-tuning. Includes object detection datasets (EverWatch, NZCattle) via patched evaluation (`--extra` flag).

| #  | Model                | Category  | Params | Score        | SN2  | TreeSat | FLAIR2 | DynEN | SN7  | EverW | NZCat |
|----|----------------------|-----------|--------|--------------|------|---------|--------|-------|------|-------|-------|
| 1  | ConvNext-XL-ImgNet   | CV-FM     | 390M   | 86.9 +/- 0.1 | 85.8 | 84.2    | 65.7   | 73.4  | 91.5 | 96.6  | 96.5  |
| 2  | DINOv3-ConvNext-WEB  | CV-FM     | 230M   | 80.1 +/- 0.1 | 95.3 | 92.5    | 80.6   | 28.3  | 92.5 | 52.4  | 75.0  |
| 3  | DINOv3-ViT-L-SAT     | VHR-FM    | 300M   | 77.8 +/- 0.2 | 73.4 | 93.7    | 98.5   | 74.5  | 89.2 | 36.1  | 57.3  |
| 4  | ConvNext-L-ImgNet    | CV-FM     | 230M   | 77.0 +/- 0.1 | 81.1 | 86.0    | 71.7   | 44.2  | 87.1 | 59.7  | 82.0  |
| 5  | DOFA-ViT 300M        | VHR-FM    | 300M   | 62.7 +/- 0.2 | 64.6 | 78.4    | 83.2   | 40.1  | 76.0 | 20.0  | 44.0  |
| 6  | Clay-V1 ViT-B        | VHR-FM    | 100M   | 60.7 +/- 0.2 | 65.8 | 69.4    | 65.6   | 49.8  | 96.9 | 43.5  | 49.5  |
| 7  | TerraMind-V1-L       | LowRes-FM | 300M   | 55.7 +/- 0.1 | 64.3 | 50.6    | 42.6   | 96.2  | 76.9 | 30.7  | 48.9  |
| 8  | Satlas-SwinB-NAIP    | VHR-FM    | 100M   | 53.4 +/- 0.1 | 49.2 | 42.4    | 59.0   | 5.6   | 84.6 | 51.0  | 69.3  |
| 9  | Satlas-Swin 100M     | LowRes-FM | 100M   | 50.8 +/- 0.2 | 41.5 | 47.8    | 40.7   | 50.8  | 84.8 | 43.1  | 66.9  |
| 10 | Prithvi-V2-600M      | LowRes-FM | 600M   | 47.8 +/- 0.1 | 76.3 | 52.3    | 40.7   | 30.8  | 90.4 | 44.1  | 4.6   |
| 11 | Prithvi-V2-300M      | LowRes-FM | 300M   | 43.8 +/- 0.2 | 61.8 | 35.0    | 32.9   | 41.7  | 81.4 | 29.5  | 52.6  |
| 12 | TerraMind-V1-B       | LowRes-FM | 100M   | 42.1 +/- 0.1 | 41.0 | 39.8    | 40.5   | 34.1  | 75.4 | 28.2  | 45.7  |
| 13 | ResNet50-ImgNet      | CV-FM     | 25M    | 36.1 +/- 0.1 | 52.8 | 28.7    | 42.6   | 23.2  | 81.5 | 21.6  | 28.9  |
| 14 | ResNet50-DeCUR       | LowRes-FM | 25M    | 24.3 +/- 0.1 | 3.3  | 20.8    | 18.7   | 54.7  | 72.8 | 3.9   | 30.7  |

Categories: **VHR-FM** = pretrained on VHR imagery, **CV-FM** = pretrained on natural/web images, **LowRes-FM** = pretrained on Sentinel (>=10m).
SN2 = SpaceNet2, SN7 = SpaceNet7, DynEN = DynamicEarthNet, EverW = EverWatch, NZCat = NZCattle.

**Key observations:**

- CV-FM models (ConvNext, DINOv3-WEB) top the leaderboard after fine-tuning, outperforming native VHR GeoFMs
- DINOv3-ViT-L-SAT is the strongest VHR-aware model
- LowRes-FM models (TerraMind, Prithvi) fall 30+ points behind, confirming Sentinel-only pretraining is insufficient for VHR
- Natural-image pretraining transfers well to VHR RGB but fails on multispectral tasks

### Model Performance on VHR Datasets: Frozen Encoder

Normalized IQM scores from GEO-Bench-2 with frozen encoder. Object detection datasets not available in frozen mode, so scores cover only 5 classification/segmentation datasets. The **FT Score** column shows the fine-tuned aggregate on these same 5 datasets (from `geo-bench2-results-vhr-base.csv`) for direct comparison.

Source: [GEO-Bench-2 Leaderboard](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard) with `--frozen` flag.

| #  | Model                | Category  | Params | Frozen        | FT Score | SN2   | TreeSat | FLAIR2 | DynEN | SN7  |
|----|----------------------|-----------|--------|---------------|----------|-------|---------|--------|-------|------|
| 1  | DINOv3-ViT-L-SAT     | VHR-FM    | 300M   | 55.4 +/- 0.3  | 88.4     | 44.9  | 1.9     | 58.0   | 71.9  | 81.3 |
| 2  | DINOv3-ConvNext-WEB  | CV-FM     | 230M   | 46.0 +/- 0.3  | 88.3     | 36.4  | -18.7   | 65.4   | 33.2  | 82.6 |
| 3  | Clay-V1 ViT-B        | VHR-FM    | 100M   | 38.0 +/- 0.4  | 67.4     | 18.1  | -18.0   | 51.8   | 36.5  | 89.2 |
| 4  | DOFA-ViT 300M        | VHR-FM    | 300M   | 36.4 +/- 0.3  | 73.5     | 13.9  | -3.0    | 57.7   | 35.2  | 73.3 |
| 5  | ConvNext-XL-ImgNet   | CV-FM     | 390M   | 35.6 +/- 0.2  | 81.1     | 37.4  | -0.2    | 38.9   | 32.2  | 82.2 |
| 6  | Prithvi-V2-600M      | LowRes-FM | 600M   | 15.4 +/- 0.3  | 56.0     | 7.5   | -28.9   | 3.0    | 37.7  | 79.6 |
| 7  | TerraMind-V1-L       | LowRes-FM | 300M   | 7.8 +/- 0.1   | 63.2     | -35.0 | -42.4   | -2.6   | 81.5  | 65.0 |
| 8  | ResNet50-DeCUR       | LowRes-FM | 25M    | -0.5 +/- 0.2  | 31.5     | -37.8 | -39.8   | -4.1   | 36.8  | 68.1 |
| 9  | Satlas-ResNet50      | LowRes-FM | 25M    | -12.5 +/- 0.3 | 44.5     | -67.1 | -65.2   | -21.0  | 55.6  | 65.5 |
| 10 | Satlas-Swin 100M     | LowRes-FM | 100M   | -22.9 +/- 0.3 | 48.1     | -86.2 | -72.5   | -27.7  | 35.7  | 73.1 |

**Key observations:**

- DINOv3-ViT-L-SAT dominates frozen evaluation (55.4), confirming its pretrained features are the most transferable to VHR without adaptation
- VHR-FM models occupy 3 of top 4 ranks
- ConvNext-XL drops from FT 81.1 to frozen 35.6 (-45.5 pts), the largest absolute drop ; its strength relies on adaptation capacity, not pretrained feature relevance
- LowRes-FM models collapse in frozen mode ; Satlas-Swin goes from FT 48.1 to frozen -22.9 (-71 pts)
- TerraMind-V1-L is strong on DynEN (81.5, highest) but catastrophic on SN2 (-35.0) and TreeSat (-42.4), suggesting Sentinel features only transfer to lower-res VHR (3m)
- Models pretrained exclusively on Sentinel data substantially underperform in both modes; VHR pretraining produces genuinely more relevant feature representations, not just better initialization
