"""Score normalization, IQM aggregation, and result export following GEO-Bench-2."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem, trim_mean

# Normalizer ranges from GEO-Bench-2 leaderboard normalizer.json
# Each entry: [worst_model_score, best_model_score]
NORMALIZER: dict[str, tuple[float, float]] = {
    "spacenet2": (0.8431392908096313, 0.8844738006591797),
    "spacenet7": (0.3912872672080993, 0.6402554512023926),
    "flair2": (0.4062314331531524, 0.5503402948379517),
    "dynamic_earthnet": (0.1660756915807724, 0.3561779260635376),
    "treesatai": (0.5388144254684448, 0.6719674468040466),
    "everwatch": (0.2177008986473083, 0.3155497610569),
    "nzcattle": (0.2819505035877228, 0.401744931936264),
}


def normalize_score(dataset: str, raw_value: float) -> float:
    """Normalize a raw metric to [0, 1] using leaderboard min-max."""
    mn, mx = NORMALIZER[dataset]
    return (raw_value - mn) / (mx - mn)


def iqm(scores) -> float:
    """Interquartile mean: trim 25% from each side."""
    return float(trim_mean(np.asarray(scores), proportiontocut=0.25))


def bootstrap_aggregate(
    df: pd.DataFrame,
    metric_col: str = "normalized",
    n_bootstrap: int = 100,
    seed: int = 100,
) -> tuple[float, float]:
    """Stratified bootstrap IQM aggregation across datasets.

    Returns (mean_score, sem) both scaled to 0-100.
    """
    rng = np.random.RandomState(seed)
    bootstrap_iqms = []
    for _ in range(n_bootstrap):
        sampled = df.groupby("dataset")[metric_col].apply(
            lambda x: rng.choice(x.values, size=len(x), replace=True)
        )
        all_sampled = np.concatenate(sampled.values)
        bootstrap_iqms.append(iqm(all_sampled))

    return float(np.mean(bootstrap_iqms) * 100), float(sem(bootstrap_iqms) * 100)


def compute_scores(results: list[dict]) -> pd.DataFrame:
    """Compute normalized scores from raw results."""
    df = pd.DataFrame(results)
    df["normalized"] = df.apply(
        lambda row: normalize_score(row["dataset"], row["test_metric"]), axis=1
    )
    return df


def print_table(df: pd.DataFrame, datasets: list[str]) -> None:
    """Print per-dataset scores and aggregate."""
    print(f"\n{'=' * 60}")
    print("  Results")
    print(f"{'=' * 60}")

    for ds in datasets:
        ds_data = df[df["dataset"] == ds]["normalized"]
        if len(ds_data) == 0:
            continue
        score = iqm(ds_data) * 100
        ds_sem = float(sem(ds_data)) * 100
        raw_data = df[df["dataset"] == ds]["test_metric"]
        raw_score = iqm(raw_data)
        print(f"  {ds:20s}: {score:6.1f} +/- {ds_sem:4.1f}  (raw: {raw_score:.4f})")

    agg_score, agg_sem = bootstrap_aggregate(df)
    print(f"\n  {'Under 10M Resolution':20s}: {agg_score:6.1f} +/- {agg_sem:4.1f}")


def export_csv(
    df: pd.DataFrame,
    output_path: Path,
    backbone: str,
    frozen: bool,
) -> None:
    """Export results in GEO-Bench-2 leaderboard CSV format."""
    rows = []
    for _, row in df.iterrows():
        rows.append(
            {
                "dataset": row["dataset"],
                "Metric": row["metric_leaderboard"],
                "test metric": row["test_metric"],
                "Seed": row["seed"],
                "batch_size": row["batch_size"],
                "lr": row["lr"],
                "decoder": row["decoder"],
                "backbone": backbone,
                "early_stop_patience": row["early_stop_patience"],
                "n_trials": row["n_trials"],
                "partition_name": "default",
                "data_percentages": row["data_pct"],
                "batch_size_selection": "HPO",
                "weight_decay": row["weight_decay"],
                "partition name": f"{row['data_pct']}%",
                "frozen_or_full_ft": "frozen" if frozen else "full_ft",
                "experiment_name": f"{backbone}_{row['dataset']}",
                "mlflow_run_name": f"{row['dataset']}_{row['seed']}",
                "mlflow_run_id": "N/A",
                "mlflow_run_status": "FINISHED",
                "experiment_id": "N/A",
                "index": 0,
            }
        )

    out_df = pd.DataFrame(rows)
    output_path.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path / "results_and_parameters.csv", index=False)


def export_model_info(
    output_path: Path,
    backbone: str,
    model_registry: dict,
) -> None:
    """Export additional_info.json for leaderboard submission."""
    display_name = backbone
    model_size = "N/A"
    for cat in model_registry.get("categories", {}).values():
        if backbone in cat.get("models", {}):
            info = cat["models"][backbone]
            display_name = info.get("display_name", backbone)
            model_size = info.get("params", "N/A")
            break

    data = {
        "Paper Link": "N/A",
        "Code Repository Link ": "N/A",
        "License": "N/A",
        "Number of HPO trials": "16",
        "Additional information about submission": (
            "GEO-Bench-2 VHR evaluation following standard protocol."
        ),
        "Comments on new models in submission": "",
        "New model info": [
            {
                "model_display_name": display_name,
                "model_size": model_size,
                "unique_backbone_key": backbone,
            }
        ],
    }

    output_path.mkdir(parents=True, exist_ok=True)
    with open(output_path / "additional_info.json", "w") as f:
        json.dump(data, f, indent=2)
