"""
test_training.py
----------------
Unit tests for model training and evaluation utilities.
"""

import sys
import pickle
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.model_training import split_features_target, load_model
from src.model_evaluation import (
    compute_regression_metrics,
    residual_analysis,
    cross_validate_model,
)
from src.config import TARGET_COLUMN


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def simple_df():
    """Minimal engineered DataFrame with two feature columns and a target."""
    np.random.seed(42)
    n = 200
    sqft = np.random.randint(500, 3000, n).astype(float)
    bhk = np.random.randint(1, 5, n).astype(float)
    price = 0.05 * sqft + 10 * bhk + np.random.normal(0, 5, n)
    return pd.DataFrame({"total_sqft": sqft, "bhk": bhk, TARGET_COLUMN: price})


@pytest.fixture
def trained_lr(simple_df):
    """A fitted LinearRegression on simple_df."""
    X, y = split_features_target(simple_df)
    model = LinearRegression()
    model.fit(X, y)
    return model, X, y


# ── split_features_target ─────────────────────────────────────────────────────

class TestSplitFeaturesTarget:
    def test_target_not_in_features(self, simple_df):
        X, y = split_features_target(simple_df)
        assert TARGET_COLUMN not in X.columns

    def test_target_series_name(self, simple_df):
        _, y = split_features_target(simple_df)
        assert y.name == TARGET_COLUMN

    def test_shapes(self, simple_df):
        X, y = split_features_target(simple_df)
        assert X.shape[0] == y.shape[0] == len(simple_df)
        assert X.shape[1] == len(simple_df.columns) - 1

    def test_raises_if_target_missing(self, simple_df):
        df_no_target = simple_df.drop(columns=[TARGET_COLUMN])
        with pytest.raises(KeyError):
            split_features_target(df_no_target)


# ── compute_regression_metrics ────────────────────────────────────────────────

class TestComputeRegressionMetrics:
    def test_perfect_predictions(self):
        y = np.array([1.0, 2.0, 3.0])
        metrics = compute_regression_metrics(y, y)
        assert metrics["r2"] == pytest.approx(1.0)
        assert metrics["mae"] == pytest.approx(0.0)
        assert metrics["rmse"] == pytest.approx(0.0)

    def test_all_keys_present(self):
        y = np.random.rand(50)
        pred = y + np.random.normal(0, 0.1, 50)
        metrics = compute_regression_metrics(y, pred)
        assert set(metrics.keys()) == {"r2", "mae", "rmse", "mape"}

    def test_r2_range(self):
        # For a good model (low noise), R² should be positive and below 1
        np.random.seed(0)
        y = np.random.rand(100) + 10  # shift to avoid near-zero denominator
        pred = y + np.random.normal(0, 0.05, 100)  # small noise → good model
        metrics = compute_regression_metrics(y, pred)
        assert 0 < metrics["r2"] <= 1


# ── residual_analysis ─────────────────────────────────────────────────────────

class TestResidualAnalysis:
    def test_keys_present(self):
        y = np.random.rand(100)
        p = y + np.random.normal(0, 0.1, 100)
        result = residual_analysis(y, p)
        expected_keys = {"residual_mean", "residual_std", "residual_skew", "percentiles"}
        assert expected_keys.issubset(result.keys())

    def test_percentile_keys(self):
        y = np.random.rand(100)
        p = y + np.random.normal(0, 0.1, 100)
        result = residual_analysis(y, p)
        assert set(result["percentiles"].keys()) == {"p5", "p25", "p50", "p75", "p95"}

    def test_zero_residuals(self):
        y = np.array([1.0, 2.0, 3.0])
        result = residual_analysis(y, y)
        assert result["residual_mean"] == pytest.approx(0.0)
        assert result["residual_std"] == pytest.approx(0.0)


# ── cross_validate_model ──────────────────────────────────────────────────────

class TestCrossValidateModel:
    def test_cv_keys(self, simple_df):
        X, y = split_features_target(simple_df)
        result = cross_validate_model(LinearRegression(), X, y)
        assert "cv_scores" in result
        assert "mean_cv_score" in result
        assert "std_cv_score" in result

    def test_cv_score_count(self, simple_df):
        X, y = split_features_target(simple_df)
        result = cross_validate_model(LinearRegression(), X, y)
        assert len(result["cv_scores"]) == 5  # CV_SPLITS default


# ── load_model ────────────────────────────────────────────────────────────────

class TestLoadModel:
    def test_round_trip(self, trained_lr):
        model, _, _ = trained_lr
        with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
            tmp_path = Path(f.name)
        with open(tmp_path, "wb") as fh:
            pickle.dump(model, fh)
        loaded = load_model(tmp_path)
        assert isinstance(loaded, LinearRegression)
        tmp_path.unlink()

    def test_raises_if_missing(self):
        with pytest.raises(FileNotFoundError):
            load_model(Path("/nonexistent/model.pkl"))
