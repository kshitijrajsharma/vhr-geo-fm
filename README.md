## Efficient Geo FM Models for VHR

### Research Objective

Investigate whether existing geospatial foundation models (GeoFMs) are sufficient for Very High Resolution (VHR) imagery tasks, or if VHR-specific pretraining is required. The primary focus is on commercial VHR satellite and aerial imagery (GSD < 10m), since many current models rely heavily on public Sentinel images which are considered insufficient for this resolution scope.

**Key research questions:**

- Do existing models fail on VHR tasks if they were not pretrained on VHR data?
- Do VHR tasks actually require VHR pretraining, or can models trained on natural/public low res sentinel images adapt well enough?
- If existing models are insufficient, should we adapt an existing FM or train from scratch efficiently?

**Scope decisions:**

- NOT focusing on VHR SAR
- NOT focusing on temporal VHR
- Primary interest: RGB and RGB+NIR combinations at sub-10m resolution
- Downstream tasks: semantic segmentation, classification, object detection, change detection
- Benchmarking through GEO-Bench-2 and PanGea (PANGAEA) as standardized evaluation frameworks

**Key early findings:** Sentinel-only models lag significantly on VHR; natural-image models adapt surprisingly well to VHR RGB but fail on multispectral; VHR pretraining matters most when encoders are frozen. See [fine-tuned](#model-performance-on-vhr-datasets-under-10m-resolution--fine-tuned) and [frozen encoder](#model-performance-on-vhr-datasets--frozen-encoder) results for details.

**References:**

- [GEO-Bench-2 paper](https://arxiv.org/pdf/2511.15658)
- [PANGAEA paper](https://arxiv.org/abs/2412.04204) · [GitHub](https://github.com/VMarsocci/pangaea-bench)
- [GEO-Bench-2 Leaderboard](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard)
- [Awesome Remote Sensing Foundation Models](https://github.com/Jack-bo1220/Awesome-Remote-Sensing-Foundation-Models)

### Definitions

**Satellite VHR**: Imagery acquired from orbital satellite platforms at sub-10m ground sampling distance (GSD). Modern commercial VHR satellites reach 30cm panchromatic natively, with pansharpened products down to 15cm. Key commercial VHR satellite sensors:

| Sensor | Operator | Panchromatic | Multispectral | Bands | Notes |
|--------|----------|-------------|---------------|-------|-------|
| WorldView-3 | Maxar | 0.3 m | 1.24 m | 8 VNIR + 8 SWIR | Highest spectral diversity |
| WorldView Legion | Maxar | 0.3 m | ~1.2 m | VNIR | Constellation for high revisit |
| Pleiades Neo 3/4 | Airbus | 0.3 m | 1.2 m | 6 (incl. Deep Blue, Red-Edge) | 15cm pansharpened product |
| WorldView-2 | Maxar | 0.5 m | 1.84 m | 8 VNIR | First 8-band VHR satellite |
| GeoEye-1 | Maxar | 0.5 m | 1.65 m | 4 VNIR | High geolocation accuracy |
| Pleiades-1A/1B | Airbus | 0.5 m | 2.0 m | 4 (RGB+NIR) | Twin constellation |
| SkySat | Planet | 0.5 m | 0.8 m native | 4 (RGB+NIR) | High-cadence tasking |
| SPOT 6/7 | Airbus | 1.5 m | 6.0 m | 4 (RGB+NIR) | Wide swath (60km) |
| Gaofen-2 | CNSA | 0.8 m | 3.2 m | 4 (RGB+NIR) | Chinese civilian VHR |
| PlanetScope (Dove) | Planet | N/A | 3-5 m | 4-8 bands | Daily global coverage |

Satellite VHR images cover large areas but may have atmospheric distortion, off-nadir angles, and varying revisit depending on constellation. Upcoming: Pleiades Neo Next (~20cm class), Planet Pelican.

**Aerial VHR**: Imagery acquired from airborne platforms (manned aircraft, drones/UAS) at very fine resolution, typically sub-1m. Examples: NAIP (~1m, US national coverage), LINZ (New Zealand aerial surveys), IGN aerial campaigns (0.2m, France), drone/UAS imagery (0.1m or finer). These images are captured under controlled conditions (ideal weather, nadir view) and typically offer only RGB or RGB+NIR bands with limited spectral diversity compared to satellite sensors.

## Datasets

### GEO-Bench-2 VHR Datasets

Source: [GEO-Bench-2](https://arxiv.org/pdf/2511.15658)

**Classification**

| Dataset | Sensor | Resolution | Domain | Category | License | Resource |
|---------|--------|------------|--------|----------|---------|----------|
| [TreeSatAI](https://essd.copernicus.org/articles/15/681/2023/) | S2 TS (benchmark); full archive also has S1 SAR + 0.2m aerial CIR | 10 m (S2) / 0.2 m (aerial) | Forestry, 13 tree species | Satellite + Aerial | CC-BY-4.0 | [Paper](https://essd.copernicus.org/articles/15/681/2023/) · [Zenodo](https://doi.org/10.5281/zenodo.6598390) |

- 50,381 image triplets, Lower Saxony, Germany. Multi-label, 20 species / 15 genera. GEO-Bench-2 uses S2 TS only.

**Semantic Segmentation**

| Dataset | Sensor | Resolution | Domain | Category | License | Resource |
|---------|--------|------------|--------|----------|---------|----------|
| [DynamicEarthNet](https://github.com/aysim/dynnet) | Planet PlanetScope (RGB+NIR) | 3 m | LULC change, 7 classes | Satellite VHR | CC-BY-4.0 | [Paper](https://arxiv.org/abs/2203.12560) · [Data](https://mediatum.ub.tum.de/1650201) · [GitHub](https://github.com/aysim/dynnet) |
| [FLAIR #2](https://ignf.github.io/FLAIR/) | Aerial RGBN (0.2 m) + S2 TS (aux) | 0.2 m (aerial) | Land cover, 13 classes | Aerial VHR | Open License 2.0 | [Project](https://ignf.github.io/FLAIR/) · [Paper](https://arxiv.org/abs/2310.13336) · [GitHub](https://github.com/IGNF/FLAIR-2) · [HuggingFace](https://huggingface.co/datasets/IGNF/FLAIR-1-2) |
| [SpaceNet 2](https://registry.opendata.aws/spacenet/) | WorldView-2/3 (8-band pansharpened) | 0.3 m (WV-3) / 0.5 m (WV-2) | Urban, building footprints, 2 classes | Satellite VHR | CC-BY-SA-4.0 | [AWS](https://registry.opendata.aws/spacenet/) · [Paper](https://arxiv.org/abs/1807.01232) |
| [SpaceNet 7](https://registry.opendata.aws/spacenet/) | Planet Dove (RGB+NIR) | 4 m | Urban, multi-temporal buildings, 2 classes | Satellite VHR | CC-BY-SA-4.0 | [Paper](https://arxiv.org/abs/2102.04420) · [AWS](https://registry.opendata.aws/spacenet/) |

- **DynamicEarthNet**: 75 AOIs, 6 continents, daily imagery (2018-2019), monthly labels.
- **FLAIR #2**: 20B+ annotated pixels, metropolitan France, 55 spatial domains.
- **SpaceNet 2**: 5 cities (Rio, Las Vegas, Shanghai, Khartoum, Paris).
- **SpaceNet 7**: 100+ geographies, 24 monthly time steps, 11M+ building annotations.

**Object Detection**

| Dataset | Sensor | Resolution | Domain | Category | License | Resource |
|---------|--------|------------|--------|----------|---------|----------|
| [EverWatch](https://zenodo.org/records/10811969) | UAS/drone aerial RGB | 0.1 m | Ecology, wading birds, 9 classes | Aerial VHR (drone) | CC0 | [Zenodo](https://zenodo.org/records/10811969) · [Project](https://everglades.weecology.org/data/uav/) · [GitHub](https://github.com/weecology/everwatch-workflow) |
| [NZ Cattle](https://zenodo.org/records/5908869) | Aerial RGB (LINZ) | 0.1 m | Agriculture, cattle detection, 2 classes | Aerial VHR | CC-BY-4.0 | [Zenodo](https://zenodo.org/records/5908869) |

- **EverWatch**: 5,125 images, Everglades, Florida.
- **NZ Cattle**: 655 images, 29,803 annotated cows, New Zealand.

### PanGea (PANGAEA) VHR Datasets

Source: [PANGAEA](https://arxiv.org/abs/2412.04204) · [GitHub](https://github.com/VMarsocci/pangaea-bench)

**Core VHR Datasets (< 10 m)**

| Dataset | Sensor | Resolution | Domain | Category | License | Resource |
|---------|--------|------------|--------|----------|---------|----------|
| [xView2 (xBD)](https://xview2.org/dataset) | Maxar WorldView-3 (RGB) | 0.3-0.5 m | HADR, building damage, 5 classes | Satellite VHR | CC-BY-NC-SA-4.0 | [Dataset](https://xview2.org/dataset) · [Paper](https://arxiv.org/abs/1911.09296) |
| [Five Billion Pixels](https://x-ytong.github.io/project/Five-Billion-Pixels.html) | Gaofen-2 (B/G/R/NIR) | 4 m | Land cover / urban, 24 classes | Satellite VHR | Open Source | [Project](https://x-ytong.github.io/project/Five-Billion-Pixels.html) · [Paper](https://arxiv.org/abs/2209.00727) |

Also includes DynamicEarthNet and SpaceNet 7 (see GEO-Bench-2 above).

- **xView2 (xBD)**: 22,068 images across 15 countries, 6 disaster types, 850,736 building polygons. 1024x1024 px pairs.
- **Five Billion Pixels**: 150 Gaofen-2 images (6800x7200 px) covering 60+ districts in China, >50,000 km2. Cropped to 520x520 tiles in PanGea.

**Community-Contributed VHR Datasets**

| Dataset | Sensor | Resolution | Domain | Category | License | Resource |
|---------|--------|------------|--------|----------|---------|----------|
| [ISPRS Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | Aerial camera (NIR/R/G/B + DSM) | 0.05 m | Urban land cover, 6 classes | Aerial VHR | Free (research) | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) |
| [Open-Canopy](https://huggingface.co/datasets/AI4Forest/Open-Canopy) | SPOT 6-7 + aerial LiDAR | 1.5 m | Forestry, canopy height | Satellite VHR | Open License 2.0 | [HuggingFace](https://huggingface.co/datasets/AI4Forest/Open-Canopy) · [Paper](https://arxiv.org/abs/2407.09392) · [GitHub](https://github.com/fajwel/Open-Canopy) |

- **ISPRS Potsdam**: 38 tiles (6000x6000 px) covering 3.42 km2 in Potsdam, Germany. 24 annotated, 14 held-out.
- **Open-Canopy**: ~360 GB, >87,000 km2 of France. Continuous canopy height regression target.

### Additional VHR Datasets (Not in GEO-Bench-2 or PanGea core)

From Pierre's GeoFM survey, filtered to GSD <= 10m, publicly licensed. 6 of 15 total surveyed VHR datasets are not included in either PanGea or GEO-Bench-2. ( However their license is not truely opensource )

| Dataset | Sensor | Resolution | Domain | Category | License | Resource |
|---------|--------|------------|--------|----------|---------|----------|
| [Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | Aerial (IR, R, G, DSM) | 0.09 m | Urban, semantic seg., 6 classes | Aerial VHR | Free (research) | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) |
| [DOTA V1.0](https://captain-whu.github.io/DOTA/) | Google Earth, GF-2, JL-1 | 0.1-4.5 m | Aerial objects, object detection, 15 classes | Satellite VHR + Aerial VHR | Not specified | [Project](https://captain-whu.github.io/DOTA/) |
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | Google Earth | 0.5 m | Building change, change detection | Satellite VHR | Academic only | [Project](https://justchenhao.github.io/LEVIR/) |
| [UCMerced](https://vision.ucmerced.edu/datasets/) | USGS aerial RGB | 0.3 m | Land use, classification, 21 classes | Aerial VHR | Public domain | [Dataset](https://vision.ucmerced.edu/datasets/) |
| [fMoW](https://github.com/fMoW/dataset) | QuickBird, GeoEye, WorldView | 0.5 m | Land use (temporal), classification, 62 classes | Satellite VHR | Custom (fMoW) | [GitHub](https://github.com/fMoW/dataset) |
| [NWPU-RESISC45](https://arxiv.org/abs/1703.00121) | Google Earth | 0.2-30 m | Scene, classification, 45 classes | Satellite VHR | Not specified | [Paper](https://arxiv.org/abs/1703.00121) |

- **Vaihingen**: Bavarian town, Germany. 33 patches, ISPRS benchmark. Same format as Potsdam.
- **DOTA V1.0**: 2,806 images, 188,282 annotated instances. Mixed sources (satellite + aerial).
- **LEVIR-CD**: 637 bi-temporal image pairs from Texas cities, 31,333 building change instances. Academic use only.
- **UCMerced**: 2,100 images (256x256), 21 land use classes from USGS National Map. Widely used baseline.
- **fMoW**: ~1M images, 62 functional categories (airports, hospitals, farms, etc.). ~3.5TB multispectral, ~200GB RGB.
- **NWPU-RESISC45**: 31,500 images, 45 scene classes, 700 per class. Created by Northwestern Polytechnical University.

### Dataset Summary by Category

**Satellite VHR** (commercial satellite sensors, GSD < 10m)

| Dataset | Sensor | Resolution | License | Benchmark |
|---------|--------|------------|---------|-----------|
| SpaceNet 2 | WorldView-2/3 | 0.3-0.5 m | CC-BY-SA-4.0 | GEO-Bench-2 |
| xView2 (xBD) | Maxar WorldView-3 | 0.3-0.5 m | CC-BY-NC-SA-4.0 | PanGea |
| fMoW | QuickBird, GeoEye, WV | 0.5 m | Custom (fMoW) | Additional |
| LEVIR-CD | Google Earth | 0.5 m | Academic only | Additional |
| Open-Canopy | SPOT 6-7 | 1.5 m | Open License 2.0 | PanGea (community) |
| DynamicEarthNet | Planet PlanetScope | 3 m | CC-BY-4.0 | GEO-Bench-2, PanGea |
| SpaceNet 7 | Planet Dove | 4 m | CC-BY-SA-4.0 | GEO-Bench-2, PanGea |
| Five Billion Pixels | Gaofen-2 | 4 m | Open Source | PanGea |
| NWPU-RESISC45 | Google Earth | 0.2-30 m | Not specified | Additional |

**Aerial VHR** (airborne / drone sensors)

| Dataset | Sensor | Resolution | License | Benchmark |
|---------|--------|------------|---------|-----------|
| ISPRS Potsdam | Aerial camera (NIRGB+DSM) | 0.05 m | Free (research) | PanGea (community) |
| Vaihingen | Aerial (IR, R, G, DSM) | 0.09 m | Free (research) | Additional |
| EverWatch | UAS/drone RGB | 0.1 m | CC0 | GEO-Bench-2 |
| NZ Cattle | Aerial RGB (LINZ) | 0.1 m | CC-BY-4.0 | GEO-Bench-2 |
| FLAIR #2 | Aerial RGBN + S2 (aux) | 0.2 m | Open License 2.0 | GEO-Bench-2 |
| UCMerced | USGS aerial RGB | 0.3 m | Public domain | Additional |

**Mixed (Satellite + Aerial)**

| Dataset | Sensor | Resolution | License | Benchmark |
|---------|--------|------------|---------|-----------|
| TreeSatAI | Sentinel-1/2 + aerial CIR | 10 m (S2) / 0.2 m (aerial) | CC-BY-4.0 | GEO-Bench-2 |
| DOTA V1.0 | Google Earth + GF-2 + JL-1 | 0.1-4.5 m | Not specified | Additional |

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
| ConvNext-XLarge-ImageNet | ConvNext | 390M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |
| [DINOv3-ConvNext-Large-WEB](https://arxiv.org/abs/2508.10104) | ConvNext | 230M | Distillation | LVD-1689M | NA | 1689M | None | DINO V3 |
| ConvNext-Large-ImageNet | ConvNext | 230M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |
| ResNet50-ImageNet | ResNet-50 | 25M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |

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
| [ResNet50-DeCUR](https://arxiv.org/abs/2209.11124) | ResNet-50 | 25M | Contrastive | Sentinel-2 | 10 m | 1M | None | Apache 2.0 |

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

### Model Performance on VHR Datasets (Under 10m Resolution) : Fine-tuned

Rankings from GEO-Bench-2 after full fine-tuning (normalized IQM scores). Includes object detection datasets (EverWatch, NZCattle) via patched evaluation (`--extra` flag).

| Rank | Model | Category | Params | VHR Pretraining | Score |
|------|-------|----------|--------|-----------------|-------|
| 1 | ConvNext-XLarge-ImageNet | CV-General-FM | 390M | None | 86.9 +/- 0.1 |
| 2 | DINOv3-ConvNext-Large-WEB | CV-General-FM | 230M | None | 80.1 +/- 0.1 |
| 3 | DINOv3-ViT-L-SAT | Native-VHR-GeoFM | 300M | Satellite VHR (Maxar) | 77.8 +/- 0.2 |
| 4 | ConvNext-Large-ImageNet | CV-General-FM | 230M | None | 77.0 +/- 0.1 |
| 5 | DOFA-ViT 300M | Native-VHR-GeoFM | 300M | Satellite + Aerial VHR | 62.7 +/- 0.2 |
| 6 | Clay-V1 ViT-B | Native-VHR-GeoFM | 100M | Aerial VHR (NAIP, LINZ) | 60.7 +/- 0.2 |
| 7 | TerraMind-V1-Large | GeoFM-LowRes | 300M | None | 55.7 +/- 0.1 |
| 8 | Satlas-SwinB-NAIP | Native-VHR-GeoFM | 100M | Aerial VHR (NAIP) | 53.4 +/- 0.1 |
| 9 | Satlas-Swin-100M | GeoFM-LowRes | 100M | None | 50.8 +/- 0.2 |
| 10 | Prithvi-EO-V2-600M-TL | GeoFM-LowRes | 600M | None | 47.8 +/- 0.1 |
| 11 | Prithvi-EO-V2-300M-TL | GeoFM-LowRes | 300M | None | 43.8 +/- 0.2 |
| 12 | TerraMind-V1-Base | GeoFM-LowRes | 100M | None | 42.1 +/- 0.1 |
| 13 | ResNet50-ImageNet | CV-General-FM | 25M | None | 36.1 +/- 0.1 |
| 14 | ResNet50-DeCUR | GeoFM-LowRes | 25M | None | 24.3 +/- 0.1 |

**Key observations (fine-tuned):**

- CV-General-FM models (ConvNext, DINOv3-WEB) top the VHR leaderboard after fine-tuning, outperforming native VHR GeoFMs
- Native-VHR-GeoFM models (DOFA, Clay) perform moderately; DINOv3-ViT-L-SAT (satellite VHR + natural image distillation) is the strongest VHR-aware model
- Models with aerial VHR only (Clay, Satlas-NAIP) rank lower than models with satellite VHR exposure (DINOv3-SAT, DOFA)
- GeoFM-LowRes models (TerraMind, Prithvi) fall 30+ points behind, confirming Sentinel-only pretraining is insufficient for VHR
- Natural-image pretraining transfers well to VHR RGB tasks but fails on multispectral tasks

### Model Performance on VHR Datasets : Frozen Encoder

Rankings from GEO-Bench-2 with frozen encoder (normalized IQM scores). Object detection datasets (EverWatch, NZCattle) are not available in frozen mode, so scores cover only classification and semantic segmentation tasks (TreeSatAI, SpaceNet2, FLAIR2, DynamicEarthNet, SpaceNet7). Source: [GEO-Bench-2 Leaderboard](https://github.com/The-AI-Alliance/GEO-Bench-2-Leaderboard) with `--frozen` flag.

| Rank | Model | Category | Params | VHR Pretraining | Score |
|------|-------|----------|--------|-----------------|-------|
| 1 | DINOv3-ViT-L-SAT | Native-VHR-GeoFM | 300M | Satellite VHR (Maxar) | 55.4 +/- 0.3 |
| 2 | DINOv3-ConvNext-Large-WEB | CV-General-FM | 230M | None | 46.0 +/- 0.3 |
| 3 | Clay-V1 ViT-B | Native-VHR-GeoFM | 100M | Aerial VHR (NAIP, LINZ) | 38.0 +/- 0.4 |
| 4 | DOFA-ViT 300M | Native-VHR-GeoFM | 300M | Satellite + Aerial VHR | 36.4 +/- 0.3 |
| 5 | ConvNext-XLarge-ImageNet | CV-General-FM | 390M | None | 35.6 +/- 0.2 |
| 6 | Prithvi-EO-V2-600M-TL | GeoFM-LowRes | 600M | None | 15.4 +/- 0.3 |
| 7 | TerraMind-V1-Large | GeoFM-LowRes | 300M | None | 7.8 +/- 0.1 |
| 8 | ResNet50-DeCUR | GeoFM-LowRes | 25M | None | -0.5 +/- 0.2 |
| 9 | Satlas-ResNet50 | GeoFM-LowRes | 25M | None | -12.5 +/- 0.3 |
| 10 | Satlas-Swin-100M | GeoFM-LowRes | 100M | None | -22.9 +/- 0.3 |

Per-dataset frozen encoder scores (normalized):

| Rank | Model | SpaceNet2 | TreeSatAI | FLAIR2 | DynamicEarthNet | SpaceNet7 |
|------|-------|-----------|-----------|--------|-----------------|-----------|
| 1 | DINOv3-ViT-L-SAT | 44.9 +/- 1.1 | 1.9 +/- 0.6 | 58.0 +/- 2.7 | 71.9 +/- 6.3 | 81.3 +/- 0.8 |
| 2 | DINOv3-ConvNext-Large-WEB | 36.4 +/- 4.9 | -18.7 +/- 2.1 | 65.4 +/- 2.7 | 33.2 +/- 3.3 | 82.6 +/- 0.7 |
| 3 | Clay-V1 ViT-B | 18.1 +/- 3.7 | -18.0 +/- 1.1 | 51.8 +/- 1.5 | 36.5 +/- 1.8 | 89.2 +/- 0.6 |
| 4 | DOFA-ViT 300M | 13.9 +/- 1.8 | -3.0 +/- 0.6 | 57.7 +/- 1.5 | 35.2 +/- 1.9 | 73.3 +/- 0.9 |
| 5 | ConvNext-XLarge-ImageNet | 37.4 +/- 1.8 | -0.2 +/- 0.5 | 38.9 +/- 0.4 | 32.2 +/- 1.2 | 82.2 +/- 0.4 |
| 6 | Prithvi-EO-V2-600M-TL | 7.5 +/- 1.6 | -28.9 +/- 1.3 | 3.0 +/- 1.7 | 37.7 +/- 5.1 | 79.6 +/- 1.2 |
| 7 | TerraMind-V1-Large | -35.0 +/- 1.1 | -42.4 +/- 4.4 | -2.6 +/- 2.0 | 81.5 +/- 5.8 | 65.0 +/- 1.4 |
| 8 | ResNet50-DeCUR | -37.8 +/- 3.4 | -39.8 +/- 1.3 | -4.1 +/- 0.8 | 36.8 +/- 3.7 | 68.1 +/- 0.1 |
| 9 | Satlas-ResNet50 | -67.1 +/- 4.2 | -65.2 +/- 1.0 | -21.0 +/- 2.2 | 55.6 +/- 14.7 | 65.5 +/- 0.4 |
| 10 | Satlas-Swin-100M | -86.2 +/- 2.2 | -72.5 +/- 0.6 | -27.7 +/- 3.1 | 35.7 +/- 6.5 | 73.1 +/- 1.2 |

**Key observations (frozen encoder):**

- DINOv3-ViT-L-SAT dominates frozen evaluation (55.4), confirming its pretrained features are the most transferable to VHR tasks without adaptation
- Native VHR GeoFMs are clear winners in frozen mode: DINOv3-SAT, Clay, and DOFA occupy 3 of the top 4 ranks
- ConvNext-XLarge-ImageNet drops from rank #1 (fine-tuned) to #5 (frozen), indicating its strong fine-tuned performance relies on adaptation capacity rather than pretrained feature relevance to VHR
- GeoFM-LowRes models collapse in frozen mode ; Satlas-Swin-100M and Satlas-ResNet50 score deeply negative, meaning their Sentinel-pretrained features are essentially unusable for VHR tasks without fine-tuning
- TerraMind-V1-Large shows an interesting pattern: strong on DynamicEarthNet (81.5, highest among all models) but catastrophic on SpaceNet2 (-35.0) and TreeSatAI (-42.4), suggesting its Sentinel features transfer only to lower-resolution VHR tasks (3m) but fail at true high-res (0.3m)
- Clay-V1 achieves the highest SpaceNet7 score (89.2) in frozen mode despite ranking #3 overall, suggesting strong temporal/urban building features from NAIP pretraining

### Frozen vs Fine-tuned: Key Ranking Changes

| Model | Fine-tuned Rank | Frozen Rank | Shift | Interpretation |
|-------|-----------------|-------------|-------|----------------|
| DINOv3-ViT-L-SAT | 3 | 1 | +2 | Best pretrained VHR features; benefits less from fine-tuning relative to others |
| ConvNext-XLarge-ImageNet | 1 | 5 | -4 | Strong learner but weak VHR features out-of-the-box |
| Clay-V1 ViT-B | 6 | 3 | +3 | VHR pretraining (NAIP/LINZ) provides useful frozen features |
| DOFA-ViT 300M | 5 | 4 | +1 | Multi-sensor VHR pretraining holds up in both modes |
| TerraMind-V1-Large | 7 | 7 | 0 | Consistently mid-low regardless of mode |

Both frozen and fine-tuned evaluations confirm that models pretrained exclusively on Sentinel data (TerraMind, Prithvi, Satlas-S2, DeCUR) substantially underperform on VHR tasks. The frozen results further reveal that VHR pretraining produces genuinely more relevant feature representations, not just better initialization for fine-tuning. However, large natural-image models (ConvNext-XLarge) can partially compensate for lack of VHR pretraining through their superior fine-tuning capacity ; a gap that disappears when encoders are frozen.
