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

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [TreeSatAI](https://essd.copernicus.org/articles/15/681/2023/) | S2 TS (benchmark); full archive also has S1 SAR + 0.2m aerial CIR | 10 m (S2) / 0.2 m (aerial) | Forestry, 13 tree species | Satellite + Aerial | [Paper](https://essd.copernicus.org/articles/15/681/2023/) · [Zenodo](https://doi.org/10.5281/zenodo.6598390) |

- 50,381 image triplets, Lower Saxony, Germany. Multi-label, 20 species / 15 genera. GEO-Bench-2 uses S2 TS only. CC-BY-4.0.

**Semantic Segmentation**

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [DynamicEarthNet](https://github.com/aysim/dynnet) | Planet PlanetScope (RGB+NIR) | 3 m | LULC change, 7 classes | Satellite VHR | [Paper](https://arxiv.org/abs/2203.12560) · [Data](https://mediatum.ub.tum.de/1650201) · [GitHub](https://github.com/aysim/dynnet) |
| [FLAIR #2](https://ignf.github.io/FLAIR/) | Aerial RGBN (0.2 m) + S2 TS (aux) | 0.2 m (aerial) | Land cover, 13 classes | Aerial VHR | [Project](https://ignf.github.io/FLAIR/) · [Paper](https://arxiv.org/abs/2310.13336) · [GitHub](https://github.com/IGNF/FLAIR-2) · [HuggingFace](https://huggingface.co/datasets/IGNF/FLAIR-1-2) |
| [SpaceNet 2](https://registry.opendata.aws/spacenet/) | WorldView-2/3 (8-band pansharpened) | 0.3 m (WV-3) / 0.5 m (WV-2) | Urban, building footprints, 2 classes | Satellite VHR | [AWS](https://registry.opendata.aws/spacenet/) · [Paper](https://arxiv.org/abs/1807.01232) |
| [SpaceNet 7](https://registry.opendata.aws/spacenet/) | Planet Dove (RGB+NIR) | 4 m | Urban, multi-temporal buildings, 2 classes | Satellite VHR | [Paper](https://arxiv.org/abs/2102.04420) · [AWS](https://registry.opendata.aws/spacenet/) |

- **DynamicEarthNet**: 75 AOIs, 6 continents, daily imagery (2018-2019), monthly labels. CC-BY-4.0.
- **FLAIR #2**: 20B+ annotated pixels, metropolitan France, 55 spatial domains. Open Licence 2.0.
- **SpaceNet 2**: 5 cities (Rio, Las Vegas, Shanghai, Khartoum, Paris). CC-BY-SA-4.0.
- **SpaceNet 7**: 100+ geographies, 24 monthly time steps, 11M+ building annotations. CC-BY-SA-4.0.

**Object Detection**

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [EverWatch](https://zenodo.org/records/10811969) | UAS/drone aerial RGB | 0.1 m | Ecology, wading birds, 9 classes | Aerial VHR (drone) | [Zenodo](https://zenodo.org/records/10811969) · [Project](https://everglades.weecology.org/data/uav/) · [GitHub](https://github.com/weecology/everwatch-workflow) |
| [NZ Cattle](https://zenodo.org/records/5908869) | Aerial RGB (LINZ) | 0.1 m | Agriculture, cattle detection, 2 classes | Aerial VHR | [Zenodo](https://zenodo.org/records/5908869) |

- **EverWatch**: 5,125 images, Everglades, Florida. CC0.
- **NZ Cattle**: 655 images, 29,803 annotated cows, New Zealand. CC-BY-4.0.

Note: GEO-Bench-2 capability section (Under 10m Resolution) includes SpaceNet2, TreeSatAI, FLAIR2, DynamicEarthNet, SpaceNet7 but excludes EverWatch and NZ Cattle (object detection excluded from capability score).

### PanGea (PANGAEA) VHR Datasets

Source: [PANGAEA](https://arxiv.org/abs/2412.04204) · [GitHub](https://github.com/VMarsocci/pangaea-bench)

**Core VHR Datasets (< 10 m)**

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [xView2 (xBD)](https://xview2.org/dataset) | Maxar WorldView-3 (RGB) | 0.3-0.5 m | HADR, building damage, 5 classes | Satellite VHR | [Dataset](https://xview2.org/dataset) · [Paper](https://arxiv.org/abs/1911.09296) |
| [Five Billion Pixels](https://x-ytong.github.io/project/Five-Billion-Pixels.html) | Gaofen-2 (B/G/R/NIR) | 4 m | Land cover / urban, 24 classes | Satellite VHR | [Project](https://x-ytong.github.io/project/Five-Billion-Pixels.html) · [Paper](https://arxiv.org/abs/2209.00727) |
| [DynamicEarthNet](https://github.com/aysim/dynnet) | Planet PlanetScope (RGB+NIR) | 3 m | LULC change, 7 classes | Satellite VHR | [Paper](https://arxiv.org/abs/2203.12560) |
| [SpaceNet 7](https://registry.opendata.aws/spacenet/) | Planet Dove (RGB+NIR) | 4 m | Urban, building tracking, 2 classes | Satellite VHR | [Paper](https://arxiv.org/abs/2102.04420) |

- **xView2 (xBD)**: 22,068 images across 15 countries, 6 disaster types, 850,736 building polygons. 1024x1024 px pairs.
- **Five Billion Pixels**: 150 Gaofen-2 images (6800x7200 px) covering 60+ districts in China, >50,000 km2. Cropped to 520x520 tiles in PanGea.

**Community-Contributed VHR Datasets**

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [ISPRS Potsdam](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | Aerial camera (NIR/R/G/B + DSM) | 0.05 m | Urban land cover, 6 classes | Aerial VHR | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) |
| [Open-Canopy](https://huggingface.co/datasets/AI4Forest/Open-Canopy) | SPOT 6-7 + aerial LiDAR | 1.5 m | Forestry, canopy height | Satellite VHR | [HuggingFace](https://huggingface.co/datasets/AI4Forest/Open-Canopy) · [Paper](https://arxiv.org/abs/2407.09392) · [GitHub](https://github.com/fajwel/Open-Canopy) |

- **ISPRS Potsdam**: 38 tiles (6000x6000 px) covering 3.42 km2 in Potsdam, Germany. 24 annotated, 14 held-out.
- **Open-Canopy**: ~360 GB, >87,000 km2 of France. Continuous canopy height regression target.

### Additional VHR Datasets (Not in GEO-Bench-2 or PanGea core)

From Pierre's GeoFM survey, filtered to GSD <= 10m, publicly licensed. 6 of 15 total surveyed VHR datasets are not included in either PanGea or GEO-Bench-2.

| Dataset | Sensor | Resolution | Domain | Category | Resource |
|---------|--------|------------|--------|----------|----------|
| [Vaihingen](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) | Aerial (IR, R, G, DSM) | 0.09 m | Urban, semantic seg., 6 classes | Aerial VHR | [ISPRS](https://www.isprs.org/resources/datasets/benchmarks/UrbanSemLab/semantic-labeling.aspx) |
| [DOTA V1.0](https://captain-whu.github.io/DOTA/) | Google Earth, GF-2, JL-1 | 0.1-4.5 m | Aerial objects, object detection, 15 classes | Satellite VHR + Aerial VHR | [Project](https://captain-whu.github.io/DOTA/) |
| [LEVIR-CD](https://justchenhao.github.io/LEVIR/) | Google Earth | 0.5 m | Building change, change detection | Satellite VHR | [Project](https://justchenhao.github.io/LEVIR/) |
| [UCMerced](https://vision.ucmerced.edu/datasets/) | USGS aerial RGB | 0.3 m | Land use, classification, 21 classes | Aerial VHR | [Dataset](https://vision.ucmerced.edu/datasets/) |
| [fMoW](https://github.com/fMoW/dataset) | QuickBird, GeoEye, WorldView | 0.5 m | Land use (temporal), classification, 62 classes | Satellite VHR | [GitHub](https://github.com/fMoW/dataset) |
| [NWPU-RESISC45](https://arxiv.org/abs/1703.00121) | Google Earth | 0.2-30 m | Scene, classification, 45 classes | Satellite VHR | [Paper](https://arxiv.org/abs/1703.00121) |

- **Vaihingen**: Bavarian town, Germany. 33 patches, ISPRS benchmark. Same format as Potsdam.
- **DOTA V1.0**: 2,806 images, 188,282 annotated instances. Mixed sources (satellite + aerial).
- **LEVIR-CD**: 637 bi-temporal image pairs from Texas cities, 31,333 building change instances. Academic use only.
- **UCMerced**: 2,100 images (256x256), 21 land use classes from USGS National Map. Widely used baseline.
- **fMoW**: ~1M images, 62 functional categories (airports, hospitals, farms, etc.). ~3.5TB multispectral, ~200GB RGB.
- **NWPU-RESISC45**: 31,500 images, 45 scene classes, 700 per class. Created by Northwestern Polytechnical University.

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
| fMoW | QuickBird, GeoEye, WV | 0.5 m | Additional |
| LEVIR-CD | Google Earth | 0.5 m | Additional |
| Open-Canopy | SPOT 6-7 | 1.5 m | PanGea (community) |
| DynamicEarthNet | Planet PlanetScope | 3 m | GEO-Bench-2, PanGea |
| SpaceNet 7 | Planet Dove | 4 m | GEO-Bench-2, PanGea |
| Five Billion Pixels | Gaofen-2 | 4 m | PanGea |
| NWPU-RESISC45 | Google Earth | 0.2-30 m | Additional |

**Aerial VHR** (airborne / drone sensors)

| Dataset | Sensor | Resolution | Benchmark |
|---------|--------|------------|-----------|
| ISPRS Potsdam | Aerial camera (NIRGB+DSM) | 0.05 m | PanGea (community) |
| Vaihingen | Aerial (IR, R, G, DSM) | 0.09 m | Additional |
| EverWatch | UAS/drone RGB | 0.1 m | GEO-Bench-2 |
| NZ Cattle | Aerial RGB (LINZ) | 0.1 m | GEO-Bench-2 |
| FLAIR #2 | Aerial RGBN + S2 (aux) | 0.2 m | GEO-Bench-2 |
| UCMerced | USGS aerial RGB | 0.3 m | Additional |

**Mixed (Satellite + Aerial)**

| Dataset | Sensor | Resolution | Benchmark |
|---------|--------|------------|-----------|
| TreeSatAI | Sentinel-1/2 + aerial CIR | 10 m (S2) / 0.2 m (aerial) | GEO-Bench-2 |
| DOTA V1.0 | Google Earth + GF-2 + JL-1 | 0.1-4.5 m | Additional |


## Models

Models evaluated in GEO-Bench-2, categorized by pretraining data type. Source: [GEO-Bench-2 Table 4](https://arxiv.org/pdf/2511.15658)

### Category 1: Native VHR GeoFM

Models pretrained on VHR imagery (GSD < 10m). These have seen high-resolution imagery during pretraining.

| Model | Backbone | Params | Technique | Pretraining Data | Resolution | Pretrain Samples | VHR Pretraining | License |
|-------|----------|--------|-----------|------------------|------------|------------------|-----------------|---------|
| DINOv3-ViT-L-SAT | ViT | 300M | Distillation | Maxar RGB | 0.6 m | 493M | Satellite VHR (Maxar) | DINO V3 |
| DOFA-ViT 300M | ViT | 300M | MAE | S1, S2, Gaofen, NAIP, EnMAP | 1-30 m | 8M | Satellite VHR (Gaofen) + Aerial VHR (NAIP) | CC-BY-4.0 |
| Clay-V1 ViT-B | ViT | 86M | MAE | Landsat 8/9, S1, S2, NAIP, LINZ, MODIS | 1-30 m | 70M | Aerial VHR (NAIP ~1m, LINZ) | Apache 2.0 |
| Satlas-SwinB-NAIP | Swin | 88M | Supervised | NAIP (USGS) | 1 m | NA | Aerial VHR (NAIP ~1m) | ODC-BY |

- **DINOv3-ViT-L-SAT**: Pretrained on 493M Maxar ortho-rectified satellite RGB tiles at 0.6m. Satellite VHR only. [Paper](https://arxiv.org/abs/2508.10104) · [HuggingFace](https://huggingface.co/facebook/dinov3-vit7b16-pretrain-sat493m)
- **DOFA**: Dynamic One-For-All. Trained on 5 sensor modalities including Gaofen (4m satellite VHR) and NAIP (~1m aerial VHR). [Paper](https://arxiv.org/abs/2403.15356) · [GitHub](https://github.com/zhu-xlab/DOFA)
- **Clay-V1**: Open foundation model. Includes NAIP (~1m, US aerial) and LINZ (NZ aerial) in pretraining. No satellite VHR sources. [Docs](https://clay-foundation.github.io/model/) · [GitHub](https://github.com/Clay-foundation/model)
- **Satlas-NAIP**: SatlasPretrain model trained specifically on NAIP aerial imagery. [Paper](https://arxiv.org/abs/2211.15660) · [GitHub](https://github.com/allenai/satlaspretrain_models)

### Category 2: General CV FM

Models pretrained on natural / web images only (ImageNet, LVD). No aerial or satellite images in pretraining.

| Model | Backbone | Params | Technique | Pretraining Data | Resolution | Pretrain Samples | VHR Pretraining | License |
|-------|----------|--------|-----------|------------------|------------|------------------|-----------------|---------|
| ConvNext-XLarge-ImageNet | ConvNext | 390M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |
| DINOv3-ConvNext-Large-WEB | ConvNext | 230M | Distillation | LVD-1689M | NA | 1689M | None | DINO V3 |
| ConvNext-Large-ImageNet | ConvNext | 230M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |
| ResNet50-ImageNet | ResNet-50 | 25M | Supervised | ImageNet-22k | NA | 14M | None | Apache 2.0 |

- **ConvNext variants**: Standard ImageNet-pretrained ConvNeXt backbones from timm. No EO data.
- **DINOv3-ConvNext-Large-WEB**: DINOv3 distillation on LVD-1689M (curated web images). No satellite or aerial imagery. [Paper](https://arxiv.org/abs/2508.10104)
- **ResNet50-ImageNet**: Standard ResNet-50 supervised on ImageNet-22k.

### Category 3: GeoFM (Low-Res)

Models pretrained on publicly available low-resolution satellite data (>= 10m), primarily Sentinel-1/2.

| Model | Backbone | Params | Technique | Pretraining Data | Resolution | Pretrain Samples | VHR Pretraining | License |
|-------|----------|--------|-----------|------------------|------------|------------------|-----------------|---------|
| TerraMind-V1-Large | ViT | 300M | Correlation | S1, S2, LULC, DEM, NDVI | 10 m | 9M | None | Apache 2.0 |
| TerraMind-V1-Base | ViT | 86M | Correlation | S1, S2, LULC, DEM, NDVI | 10 m | 9M | None | Apache 2.0 |
| Prithvi-EO-V2-600M-TL | ViT | 600M | MAE | HLS (Harmonized) | 30 m | 4.2M | None | Apache 2.0 |
| Prithvi-EO-V2-300M-TL | ViT | 300M | MAE | HLS | 30 m | 4.2M | None | Apache 2.0 |
| Satlas-SwinB-Sentinel2 | Swin | 88M | Supervised | Sentinel-2 | 10 m | NA | None | ODC-BY |
| Satlas-Swin-100M | Swin | 100M | Supervised | Sentinel-2 | 10 m | NA | None | ODC-BY |
| ResNet50-DeCUR | ResNet-50 | 25M | Contrastive | Sentinel-2 | 10 m | 1M | None | Apache 2.0 |

- **TerraMind**: IBM-ESA model. Pretrained on TerraMesh (9M samples from S1/S2/LULC/DEM/NDVI at 10m). [Paper](https://arxiv.org/abs/2504.11171) · [GitHub](https://github.com/IBM/terramind)
- **Prithvi-EO-V2**: NASA-IBM model. Pretrained on Harmonized Landsat-Sentinel (HLS) at 30m. [GitHub](https://github.com/NASA-IMPACT/Prithvi-EO-2.0)
- **Satlas-Sentinel2**: SatlasPretrain backbone trained on Sentinel-2 imagery at 10m. [Paper](https://arxiv.org/abs/2211.15660)
- **ResNet50-DeCUR**: Contrastive self-supervised on Sentinel-2 at 10m. [Paper](https://arxiv.org/abs/2209.11124)

### Model Performance on VHR Datasets (Under 10m Resolution)

Rankings from GEO-Bench-2 after full fine-tuning (normalized IQM scores). Includes object detection datasets (EverWatch, NZCattle) via patched evaluation.

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

**Key observations:**

- CV-General-FM models (ConvNext, DINOv3-WEB) top the VHR leaderboard, outperforming native VHR GeoFMs
- Native-VHR-GeoFM models (DOFA, Clay) perform moderately; DINOv3-ViT-L-SAT (satellite VHR + natural image distillation) is the strongest VHR-aware model
- Models with aerial VHR only (Clay, Satlas-NAIP) rank lower than models with satellite VHR exposure (DINOv3-SAT, DOFA)
- GeoFM-LowRes models (TerraMind, Prithvi) fall 30+ points behind, confirming Sentinel-only pretraining is insufficient for VHR
- Natural-image pretraining transfers well to VHR RGB tasks but fails on multispectral tasks
