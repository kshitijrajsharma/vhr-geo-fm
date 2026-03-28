"""Score normalization, IQM aggregation, and CSV export per GEO-Bench-2."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import sem, trim_mean

# Min-max ranges from GEO-Bench-2 leaderboard normalizer.json
NORMALIZER: dict[str, tuple[float, float]] = {
    "spacenet2": (0.8431392908096313, 0.8844738006591797),
    "spacenet7": (0.3912872672080993, 0.6402554512023926),
    "flair2": (0.4062314331531524, 0.5503402948379517),
    "dynamic_earthnet": (0.1660756915807724, 0.3561779260635376),
    "treesatai": (0.5388144254684448, 0.6719674468040466),
    "everwatch": (0.2177008986473083, 0.3155497610569),
    "nzcattle": (0.2819505035877228, 0.401744931936264),
}


def normalize_score(dataset: str, raw: float) -> float:
    mn, mx = NORMALIZER[dataset]
    return (raw - mn) / (mx - mn)


def iqm(scores) -> float:
    """Interquartile mean: trim 25% from each side."""
    return float(trim_mean(np.asarray(scores), proportiontocut=0.25))


def bootstrap_aggregate(
    df: pd.DataFrame,
    metric_col: str = "normalized",
    n_bootstrap: int = 100,
    seed: int = 100,
) -> tuple[float, float]:
    """Stratified bootstrap IQM. Returns (mean, sem) scaled to 0-100."""
    rng = np.random.RandomState(seed)
    iqms = []
    for _ in range(n_bootstrap):
        sampled = df.groupby("dataset")[metric_col].apply(
            lambda x: rng.choice(x.values, size=len(x), replace=True)
        )
        iqms.append(iqm(np.concatenate(sampled.values)))
    return float(np.mean(iqms) * 100), float(sem(iqms) * 100)


def compute_scores(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    df["normalized"] = df.apply(lambda r: normalize_score(r["dataset"], r["test_metric"]), axis=1)
    return df


def print_table(df: pd.DataFrame, datasets: list[str]) -> None:
    print(f"\n{'=' * 60}\n  Results\n{'=' * 60}")
    for ds in datasets:
        d = df[df["dataset"] == ds]
        if d.empty:
            continue
        score = iqm(d["normalized"]) * 100
        raw = iqm(d["test_metric"])
        print(f"  {ds:20s}: {score:6.1f} +/- {sem(d['normalized']) * 100:4.1f}  (raw: {raw:.4f})")

    agg, agg_sem = bootstrap_aggregate(df)
    print(f"\n  {'Under 10M Resolution':20s}: {agg:6.1f} +/- {agg_sem:4.1f}")


def export_csv(
    df: pd.DataFrame,
    output_path: Path,
    backbone: str,
    frozen: bool,
    run_metadata: dict | None = None,
) -> None:
    mode = "frozen" if frozen else "full_ft"
    rows = []
    for _, r in df.iterrows():
        row = {
            "dataset": r["dataset"],
            "metric": r["metric_leaderboard"],
            "test_metric": r["test_metric"],
            "normalized": r["normalized"],
            "seed": r["seed"],
            "backbone": backbone,
            "mode": mode,
            "decoder": r["decoder"],
            "batch_size": r["batch_size"],
            "lr": r["lr"],
            "weight_decay": r["weight_decay"],
            "hpo_trials": r["n_trials"],
            "early_stop_patience": r["early_stop_patience"],
            "data_pct": r["data_pct"],
            "run_seconds": r.get("run_seconds"),
        }
        if run_metadata:
            row |= run_metadata
        rows.append(row)

    out = pd.DataFrame(rows).sort_values(["dataset", "seed"], ignore_index=True)
    output_path.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_path / "results.csv", index=False)
