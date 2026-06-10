"""
preprocessing.py
----------------
Data cleaning pipeline: null removal, type coercion, and sqft normalisation.
"""

from typing import Optional

import numpy as np
import pandas as pd

from src.utils import get_logger, timer

logger = get_logger(__name__)


def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows containing any null value and log the change in size.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with null rows removed.
    """
    before = len(df)
    df = df.dropna()
    dropped = before - len(df)
    logger.info("drop_nulls: removed %d null rows (%d → %d)", dropped, before, len(df))
    return df.reset_index(drop=True)


def extract_bhk(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the free-text ``size`` column (e.g. "2 BHK", "4 Bedroom") into
    an integer ``bhk`` column.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``size`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with a new integer ``bhk`` column; original ``size``
        column is retained for traceability.

    Raises
    ------
    KeyError
        If the ``size`` column is absent.
    """
    if "size" not in df.columns:
        raise KeyError("Column 'size' not found in DataFrame.")

    df = df.copy()
    df["bhk"] = df["size"].apply(_parse_bhk)
    invalid_mask = df["bhk"].isnull()
    if invalid_mask.any():
        logger.warning("extract_bhk: %d rows have unparseable 'size' values.", invalid_mask.sum())
        df = df[~invalid_mask].copy()

    df["bhk"] = df["bhk"].astype(int)
    logger.info("extract_bhk: bhk unique values = %s", sorted(df["bhk"].unique()))
    return df


def _parse_bhk(value: str) -> Optional[int]:
    """Parse a single size string into BHK count."""
    try:
        return int(str(value).split()[0])
    except (ValueError, IndexError):
        return None


def convert_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalise the ``total_sqft`` column.

    Handles three formats:
    * Plain numeric strings → direct float cast.
    * Range strings (``"1200-1500"``) → arithmetic mean.
    * Anything else (e.g. ``"34.46Sq. Meter"``) → ``NaN`` (dropped).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``total_sqft`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with ``total_sqft`` as a float64 column; unparseable
        rows removed.
    """
    if "total_sqft" not in df.columns:
        raise KeyError("Column 'total_sqft' not found in DataFrame.")

    df = df.copy()
    df["total_sqft"] = df["total_sqft"].apply(_parse_sqft)

    before = len(df)
    df = df[df["total_sqft"].notnull()].copy()
    logger.info("convert_sqft: dropped %d unparseable sqft rows.", before - len(df))
    return df.reset_index(drop=True)


def _parse_sqft(value) -> Optional[float]:
    """Parse a single sqft value (plain, range, or unparseable)."""
    s = str(value).strip()
    parts = s.split("-")
    if len(parts) == 2:
        try:
            return (float(parts[0]) + float(parts[1])) / 2
        except ValueError:
            return None
    try:
        return float(s)
    except ValueError:
        return None


@timer
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    End-to-end cleaning pipeline.

    Steps
    -----
    1. Drop null rows.
    2. Extract integer BHK from the ``size`` column.
    3. Convert ``total_sqft`` to a numeric float.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame (after column selection).

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame, ready for feature engineering.
    """
    logger.info("Starting data cleaning pipeline. Input shape: %s", df.shape)
    df = drop_nulls(df)
    df = extract_bhk(df)
    df = convert_sqft(df)
    logger.info("Cleaning complete. Output shape: %s", df.shape)
    return df
