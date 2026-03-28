"""Tests for score normalization, IQM, and bootstrap aggregation."""

import numpy as np
import pandas as pd
import pytest

from eval.scoring import NORMALIZER, bootstrap_aggregate, compute_scores, iqm, normalize_score


class TestNormalize:
    def test_at_min_returns_zero(self):
        for ds, (mn, _) in NORMALIZER.items():
            assert normalize_score(ds, mn) == pytest.approx(0.0)

    def test_at_max_returns_one(self):
        for ds, (_, mx) in NORMALIZER.items():
            assert normalize_score(ds, mx) == pytest.approx(1.0)

    def test_midpoint(self):
        mn, mx = NORMALIZER["spacenet2"]
        assert normalize_score("spacenet2", (mn + mx) / 2) == pytest.approx(0.5)

    def test_above_max_exceeds_one(self):
        _, mx = NORMALIZER["spacenet2"]
        assert normalize_score("spacenet2", mx + 0.01) > 1.0

    def test_unknown_dataset_raises(self):
        with pytest.raises(KeyError):
            normalize_score("nonexistent", 0.5)


class TestIQM:
    def test_uniform_scores(self):
        assert iqm([0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)

    def test_outlier_robustness(self):
        # IQM trims 25% each side, so extreme outliers are dropped
        scores = [0.0, 0.5, 0.5, 1.0]
        result = iqm(scores)
        assert result == pytest.approx(0.5, abs=0.01)

    def test_single_value(self):
        assert iqm([0.7]) == pytest.approx(0.7)


class TestComputeScores:
    def test_adds_normalized_column(self):
        results = [
            {"dataset": "spacenet2", "test_metric": NORMALIZER["spacenet2"][0], "seed": 42},
            {"dataset": "spacenet2", "test_metric": NORMALIZER["spacenet2"][1], "seed": 43},
        ]
        df = compute_scores(results)
        assert "normalized" in df.columns
        assert df["normalized"].iloc[0] == pytest.approx(0.0)
        assert df["normalized"].iloc[1] == pytest.approx(1.0)


class TestBootstrap:
    def test_perfect_scores_return_100(self):
        # All normalized scores = 1.0 -> aggregate should be ~100
        rows = []
        for ds in ["spacenet2", "spacenet7"]:
            for seed in range(5):
                rows.append({"dataset": ds, "normalized": 1.0, "seed": seed})
        df = pd.DataFrame(rows)
        mean, se = bootstrap_aggregate(df, n_bootstrap=50)
        assert mean == pytest.approx(100.0, abs=0.1)
        assert se < 1.0

    def test_zero_scores_return_zero(self):
        rows = [{"dataset": "spacenet2", "normalized": 0.0, "seed": i} for i in range(5)]
        df = pd.DataFrame(rows)
        mean, _ = bootstrap_aggregate(df, n_bootstrap=50)
        assert mean == pytest.approx(0.0, abs=0.1)

    def test_deterministic(self):
        rows = []
        for ds in ["spacenet2", "flair2"]:
            for seed in range(5):
                rows.append({"dataset": ds, "normalized": np.random.rand(), "seed": seed})
        df = pd.DataFrame(rows)
        r1 = bootstrap_aggregate(df, seed=99)
        r2 = bootstrap_aggregate(df, seed=99)
        assert r1[0] == r2[0]
