import argparse
import json
import os

import pandas as pd
from scipy.stats import sem
from utils import compute_tools
from utils.constants import DIMENSIONS, MODEL_INFO_FILE, RESULTS_DIR


def format_metric(val: float, err: float) -> str:
    return f"{val * 100:.1f} ± {err * 100:.1f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Add extra object detection datasets (everwatch, nzcattle)",
    )
    args = parser.parse_args()

    # Identify base datasets mapped to capability
    target_key = next(k for k in DIMENSIONS if k.lower() == "under 10m resolution")
    required_datasets = list(DIMENSIONS[target_key])

    # Inject additional datasets into the aggregation pool when requested
    if args.extra:
        required_datasets.extend(["everwatch", "nzcattle"])

    with open(MODEL_INFO_FILE, "r") as f:
        model_info = json.load(f)

    all_results = []
    for sub in os.listdir(RESULTS_DIR):
        csv_path = os.path.join(RESULTS_DIR, sub, "results_and_parameters.csv")
        if os.path.isfile(csv_path):
            df = pd.read_csv(csv_path)
            # Restrict to full fine-tuning configurations to guarantee equivalent evaluation conditions
            if not df.empty and df["frozen_or_full_ft"].iloc[0] == "full_ft":
                df["submission"] = sub
                all_results.append(df)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Base normalization is strictly required prior to calculating IQM distributions to match reference methodology
    normalizer = compute_tools.load_normalizer(benchmark_name="leaderboard_combined")
    new_metric = normalizer.normalize_data_frame(df=combined_df, metric="test metric")

    req_set = set(required_datasets)
    dimension_data = combined_df[combined_df["dataset"].isin(req_set)].copy()
    compiled_metrics = []

    for (sub, backbone), bb_df in dimension_data.groupby(["submission", "backbone"]):
        # Discard model variations lacking evaluations for ALL required datasets to avoid skewed aggregate means
        if set(bb_df["dataset"].unique()) != req_set:
            continue

        # Stratified bootstrap aligns IQM distributions to calculate the overall capability score
        bb_dim_iqms = compute_tools.bootstrap_iqm_aggregate(bb_df, metric=new_metric)[
            new_metric
        ]
        dim_val, dim_err = bb_dim_iqms.mean(), sem(bb_dim_iqms)

        row = {
            "Model": model_info["BACKBONE_NAMES"].get(backbone, backbone),
            "# Params": model_info["MODEL_SIZE"].get(backbone, "-"),
            "Submission": str(sub).split("-")[0],
            "Under 10M Resolution": format_metric(dim_val, dim_err),
        }

        # Calculate isolated per-dataset IQM and Trimmed SEM
        for ds in required_datasets:
            ds_series = bb_df[bb_df["dataset"] == ds][new_metric]
            row[ds.title()] = format_metric(
                compute_tools.iqm(ds_series), compute_tools.trimmed_sem(ds_series)
            )

        compiled_metrics.append((dim_val, row))

    compiled_metrics.sort(key=lambda x: x[0], reverse=True)
    final_df = pd.DataFrame([r[1] for r in compiled_metrics])
    final_df.insert(0, "Rank", range(1, len(final_df) + 1))

    print(final_df.to_markdown(index=False))


if __name__ == "__main__":
    main()
