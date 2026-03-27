"""Score normalization, IQM aggregation, and result export following GEO-Bench-2."""

from __future__ import annotations

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
    run_metadata: dict | None = None,
) -> None:
    """Export concise, comparable experiment results."""
    mode = "frozen" if frozen else "full_ft"
    rows = []
    for _, row in df.iterrows():
        row_data = {
            "dataset": row["dataset"],
            "metric": row["metric_leaderboard"],
            "test_metric": row["test_metric"],
            "normalized": row["normalized"],
            "seed": row["seed"],
            "backbone": backbone,
            "mode": mode,
            "decoder": row["decoder"],
            "batch_size": row["batch_size"],
            "lr": row["lr"],
            "weight_decay": row["weight_decay"],
            "hpo_trials": row["n_trials"],
            "early_stop_patience": row["early_stop_patience"],
            "data_pct": row["data_pct"],
            "run_seconds": row.get("run_seconds", None),
        }

        if run_metadata is not None:
            row_data.update(
                {
                    "run_started_at_utc": run_metadata.get("started_at_utc"),
                    "run_ended_at_utc": run_metadata.get("ended_at_utc"),
                    "total_run_seconds": run_metadata.get("total_run_seconds"),
                    "hostname": run_metadata.get("hostname"),
                    "platform": run_metadata.get("platform"),
                    "python_version": run_metadata.get("python_version"),
                    "torch_version": run_metadata.get("torch_version"),
                    "lightning_version": run_metadata.get("lightning_version"),
                    "cpu_cores": run_metadata.get("cpu_cores"),
                    "gpu_name": run_metadata.get("gpu_name"),
                    "gpu_count": run_metadata.get("gpu_count"),
                    "gpu_total_memory_gb": run_metadata.get("gpu_total_memory_gb"),
                    "gpu_free_memory_gb_at_start": run_metadata.get(
                        "gpu_free_memory_gb_at_start"
                    ),
                    "gpu_compute_capability": run_metadata.get("gpu_compute_capability"),
                }
            )

        rows.append(row_data)

    out_df = pd.DataFrame(rows)
    out_df = out_df.sort_values(by=["dataset", "seed"], ignore_index=True)
    output_path.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(output_path / "results_and_parameters.csv", index=False)
