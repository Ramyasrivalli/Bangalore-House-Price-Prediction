"""
test_prediction.py
------------------
Unit tests for the inference pipeline (HousePricePredictor).
"""

import json
import pickle
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predictor import HousePricePredictor


# ── Fixtures ──────────────────────────────────────────────────────────────────

MOCK_LOCATIONS = ["Indiranagar", "Whitefield", "Koramangala", "HSR Layout"]
# Feature order: total_sqft, bath, bhk, <locations…>
MOCK_COLUMNS = ["total_sqft", "bath", "bhk"] + MOCK_LOCATIONS


def make_predictor(tmp_path: Path) -> HousePricePredictor:
    """Build a HousePricePredictor backed by a trivial fitted model."""
    # Fit a real LinearRegression on random data so predict() works
    n_features = len(MOCK_COLUMNS)
    X = np.random.rand(100, n_features)
    y = X[:, 0] * 50 + X[:, 2] * 10 + np.random.normal(0, 5, 100)

    model = LinearRegression()
    model.fit(X, y)

    model_path = tmp_path / "model.pkl"
    columns_path = tmp_path / "columns.json"

    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    with open(columns_path, "w") as f:
        json.dump({"data_columns": MOCK_COLUMNS}, f)

    return HousePricePredictor(model_path=model_path, columns_path=columns_path)


@pytest.fixture
def predictor(tmp_path):
    return make_predictor(tmp_path)


# ── locations property ────────────────────────────────────────────────────────

class TestLocationsProperty:
    def test_returns_list(self, predictor):
        assert isinstance(predictor.locations, list)

    def test_excludes_numeric_features(self, predictor):
        assert "total_sqft" not in predictor.locations
        assert "bath" not in predictor.locations
        assert "bhk" not in predictor.locations

    def test_contains_known_locations(self, predictor):
        for loc in MOCK_LOCATIONS:
            assert loc in predictor.locations

    def test_is_sorted(self, predictor):
        locs = predictor.locations
        assert locs == sorted(locs)


# ── predict ───────────────────────────────────────────────────────────────────

class TestPredict:
    def test_returns_float(self, predictor):
        result = predictor.predict("Indiranagar", sqft=1200, bath=2, bhk=2)
        assert isinstance(result, float)

    def test_result_is_non_negative(self, predictor):
        result = predictor.predict("Whitefield", sqft=1000, bath=1, bhk=1)
        assert result >= 0.0

    def test_unknown_location_does_not_raise(self, predictor):
        """Graceful fallback: unknown location → 'other' (all-zero one-hot)."""
        result = predictor.predict("UnknownPlace", sqft=1200, bath=2, bhk=2)
        assert isinstance(result, float)

    def test_larger_sqft_gives_higher_price(self, predictor):
        """Sanity check: more area → higher predicted price."""
        small = predictor.predict("Indiranagar", sqft=500, bath=1, bhk=1)
        large = predictor.predict("Indiranagar", sqft=3000, bath=1, bhk=1)
        assert large > small

    def test_invalid_sqft_raises(self, predictor):
        with pytest.raises(ValueError, match="sqft"):
            predictor.predict("Indiranagar", sqft=0, bath=2, bhk=2)

    def test_invalid_bath_raises(self, predictor):
        with pytest.raises(ValueError, match="bath"):
            predictor.predict("Indiranagar", sqft=1200, bath=0, bhk=2)

    def test_invalid_bhk_raises(self, predictor):
        with pytest.raises(ValueError, match="bhk"):
            predictor.predict("Indiranagar", sqft=1200, bath=2, bhk=0)

    def test_suspicious_bath_over_bhk_raises(self, predictor):
        with pytest.raises(ValueError, match="Suspicious"):
            predictor.predict("Indiranagar", sqft=1200, bath=10, bhk=2)


# ── artifacts missing ─────────────────────────────────────────────────────────

class TestMissingArtifacts:
    def test_missing_model_raises(self, tmp_path):
        columns_path = tmp_path / "columns.json"
        columns_path.write_text(json.dumps({"data_columns": MOCK_COLUMNS}))
        with pytest.raises(FileNotFoundError):
            HousePricePredictor(
                model_path=tmp_path / "no_model.pkl",
                columns_path=columns_path,
            )

    def test_missing_columns_raises(self, tmp_path):
        model_path = tmp_path / "model.pkl"
        model = LinearRegression()
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        with pytest.raises(FileNotFoundError):
            HousePricePredictor(
                model_path=model_path,
                columns_path=tmp_path / "no_columns.json",
            )
