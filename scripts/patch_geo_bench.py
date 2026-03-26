# uv pip install tabulate scipy pandas

import argparse
import json
import os

import pandas as pd
from scipy.stats import sem
from utils import compute_tools
from utils.constants import DIMENSIONS, MODEL_INFO_FILE, RESULTS_DIR


def format_metric(val: float, err: float) -> str:
    return f"{val * 100:.1f} ± {err * 100:.1f}"


def map_model_category(model_name: str) -> str:
    """Classifies models into their training data provenance categories."""
    vhr_aerial = {
        "DinoV3-ViT-L-SAT",
        "DOFA-ViT 300M",
        "DOFA-ViT-B",
        "Clay-V1 ViT-B",
        "Satlas-SwinB-Naip",
        "Satlas-SwinB-Naip",
    }
    natural_images = {
        "Resnet50-ImageNet",
        "ConvNext-Large-ImageNet",
        "ConvNext-XLarge-ImageNet",
        "DinoV3-ConvNext-Large-WEB",
    }
    satellite_geo_fm = {
        "Resnet50-DeCUR",
        "Satlas-Swin 100M",
        "Satlas-Resnet50",
        "Prithvi-EO-V2-100",
        "Prithvi-EO-V2 300M",
        "Prithvi-EO-V2 300M TL",
        "Prithvi-EO-V2 600M",
        "Prithvi-EO-V2 600M TL",
        "Prithvi-EO-V1-100",
        "TerraMind-V1-Base",
        "TerraMind-V1-Large",
        "THOR-V1-Base",
    }

    if model_name in vhr_aerial:
        return "Native-VHR-GeoFM"
    if model_name in natural_images:
        return "CV-General-FM"
    if model_name in satellite_geo_fm:
        return "GeoFM-LowRes"
    return "Unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Add extra object detection datasets (everwatch, nzcattle)",
    )
    parser.add_argument(
        "--frozen",
        action="store_true",
        help="Use frozen backbone results instead of fully finetuned",
    )
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="Optional file path to export the results as a CSV",
    )
    args = parser.parse_args()

    target_key = next(k for k in DIMENSIONS if k.lower() == "under 10m resolution")
    required_datasets = list(DIMENSIONS[target_key])

    if args.extra:
        required_datasets.extend(["everwatch", "nzcattle"])

    with open(MODEL_INFO_FILE, "r") as f:
        model_info = json.load(f)

    all_results = []
    for sub in os.listdir(RESULTS_DIR):
        csv_path = os.path.join(RESULTS_DIR, sub, "results_and_parameters.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            ft_filter = "frozen" if args.frozen else "full_ft"
            if not df.empty and df["frozen_or_full_ft"].iloc[0] == ft_filter:
                df["submission"] = sub
                all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)

    normalizer = compute_tools.load_normalizer(benchmark_name="leaderboard_combined")
    new_metric = normalizer.normalize_data_frame(df=combined_df, metric="test metric")

    req_set = set(required_datasets)
    dimension_data = combined_df[combined_df["dataset"].isin(req_set)].copy()
    compiled_metrics = []

    for (sub, backbone), bb_df in dimension_data.groupby(["submission", "backbone"]):
        if set(bb_df["dataset"].unique()) != req_set:
            continue

        bb_dim_iqms = compute_tools.bootstrap_iqm_aggregate(bb_df, metric=new_metric)[
            new_metric
        ]
        dim_val, dim_err = bb_dim_iqms.mean(), sem(bb_dim_iqms)

        display_name = model_info["BACKBONE_NAMES"].get(backbone, backbone)
        row = {
            "Model": display_name,
            "Category": map_model_category(display_name),
            "# Params": model_info["MODEL_SIZE"].get(backbone, "-"),
            "Submission": str(sub).split("-")[0],
            "Under 10M Resolution": format_metric(dim_val, dim_err),
        }

        for ds in required_datasets:
            ds_series = bb_df[bb_df["dataset"] == ds][new_metric]
            row[ds.title()] = format_metric(
                compute_tools.iqm(ds_series), compute_tools.trimmed_sem(ds_series)
            )

        compiled_metrics.append((dim_val, row))

    compiled_metrics.sort(key=lambda x: x[0], reverse=True)
    final_df = pd.DataFrame([r[1] for r in compiled_metrics])
    final_df.insert(0, "Rank", range(1, len(final_df) + 1))

    if args.out:
        final_df.to_csv(args.out, index=False)
    else:
        print(final_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
