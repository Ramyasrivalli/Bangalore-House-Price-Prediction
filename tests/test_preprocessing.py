"""
test_preprocessing.py
---------------------
Unit tests for the data cleaning pipeline.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocessing import (
    drop_nulls,
    extract_bhk,
    convert_sqft,
    clean_data,
    _parse_bhk,
    _parse_sqft,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_df(**kwargs) -> pd.DataFrame:
    """Create a minimal DataFrame for testing."""
    defaults = {
        "location": ["Indiranagar", "Whitefield", "Koramangala"],
        "size": ["2 BHK", "3 BHK", "4 Bedroom"],
        "total_sqft": ["1200", "1500-1800", "2100"],
        "bath": [2, 3, 4],
        "price": [80.0, 120.0, 200.0],
    }
    defaults.update(kwargs)
    return pd.DataFrame(defaults)


# ── drop_nulls ────────────────────────────────────────────────────────────────

class TestDropNulls:
    def test_removes_null_rows(self):
        df = make_df()
        df.loc[1, "location"] = np.nan
        result = drop_nulls(df)
        assert len(result) == 2
        assert result.isnull().sum().sum() == 0

    def test_no_nulls_unchanged(self):
        df = make_df()
        result = drop_nulls(df)
        assert len(result) == 3

    def test_all_nulls_returns_empty(self):
        # Build a fully-float DataFrame so NaN assignment works on all columns
        df = pd.DataFrame({
            "location": [np.nan, np.nan],
            "size": [np.nan, np.nan],
            "total_sqft": [np.nan, np.nan],
            "bath": [np.nan, np.nan],
            "price": [np.nan, np.nan],
        })
        result = drop_nulls(df)
        assert len(result) == 0


# ── _parse_bhk / extract_bhk ─────────────────────────────────────────────────

class TestParseBhk:
    @pytest.mark.parametrize("value,expected", [
        ("2 BHK", 2),
        ("3 BHK", 3),
        ("4 Bedroom", 4),
        ("1 RK", 1),
    ])
    def test_valid_sizes(self, value, expected):
        assert _parse_bhk(value) == expected

    @pytest.mark.parametrize("value", ["", "BHK", None, "NaN"])
    def test_invalid_returns_none(self, value):
        assert _parse_bhk(value) is None


class TestExtractBhk:
    def test_creates_bhk_column(self):
        df = make_df()
        result = extract_bhk(df)
        assert "bhk" in result.columns

    def test_correct_values(self):
        df = make_df()
        result = extract_bhk(df)
        assert list(result["bhk"]) == [2, 3, 4]

    def test_drops_unparseable_rows(self):
        df = make_df(size=["2 BHK", "Bad Value", "3 BHK"])
        result = extract_bhk(df)
        assert len(result) == 2

    def test_raises_without_size_column(self):
        df = make_df().drop(columns=["size"])
        with pytest.raises(KeyError):
            extract_bhk(df)


# ── _parse_sqft / convert_sqft ────────────────────────────────────────────────

class TestParseSqft:
    @pytest.mark.parametrize("value,expected", [
        ("1200", 1200.0),
        ("1500.5", 1500.5),
        ("1200-1500", 1350.0),
        ("2100-2850", 2475.0),
    ])
    def test_valid_inputs(self, value, expected):
        assert _parse_sqft(value) == pytest.approx(expected)

    @pytest.mark.parametrize("value", ["34.46Sq. Meter", "N/A", ""])
    def test_invalid_returns_none(self, value):
        assert _parse_sqft(value) is None


class TestConvertSqft:
    def test_plain_numeric(self):
        df = make_df(total_sqft=["1200", "1500", "2100"])
        result = convert_sqft(df)
        assert list(result["total_sqft"]) == pytest.approx([1200.0, 1500.0, 2100.0])

    def test_range_converted_to_mean(self):
        df = make_df(total_sqft=["1200-1800", "1500", "2100"])
        result = convert_sqft(df)
        assert result["total_sqft"].iloc[0] == pytest.approx(1500.0)

    def test_unparseable_rows_dropped(self):
        df = make_df(total_sqft=["1200", "34.46Sq. Meter", "2100"])
        result = convert_sqft(df)
        assert len(result) == 2

    def test_raises_without_column(self):
        df = make_df().drop(columns=["total_sqft"])
        with pytest.raises(KeyError):
            convert_sqft(df)


# ── clean_data (integration) ──────────────────────────────────────────────────

class TestCleanData:
    def test_output_has_bhk_column(self):
        df = make_df()
        result = clean_data(df)
        assert "bhk" in result.columns

    def test_output_has_numeric_sqft(self):
        df = make_df()
        result = clean_data(df)
        assert pd.api.types.is_float_dtype(result["total_sqft"])

    def test_no_nulls_in_output(self):
        df = make_df()
        df.loc[0, "bath"] = np.nan
        result = clean_data(df)
        assert result.isnull().sum().sum() == 0
