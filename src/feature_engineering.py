"""
feature_engineering.py
-----------------------
Feature engineering pipeline: price-per-sqft, location grouping,
outlier removal, and one-hot encoding.
"""

import numpy as np
import pandas as pd

from src.config import (
    MIN_SQFT_PER_BHK,
    MAX_BATH_OVER_BHK,
    RARE_LOCATION_THRESHOLD,
    PRICE_SCALE_FACTOR,
)
from src.utils import get_logger, timer

logger = get_logger(__name__)


def add_price_per_sqft(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute ``price_per_sqft`` (₹ per sq ft) as a diagnostic feature.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``price`` (lakhs) and ``total_sqft`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with an additional ``price_per_sqft`` float column.
    """
    df = df.copy()
    df["price_per_sqft"] = df["price"] * PRICE_SCALE_FACTOR / df["total_sqft"]
    logger.info("price_per_sqft: mean=%.1f  std=%.1f", df["price_per_sqft"].mean(), df["price_per_sqft"].std())
    return df


def group_rare_locations(df: pd.DataFrame, threshold: int = RARE_LOCATION_THRESHOLD) -> pd.DataFrame:
    """
    Collapse infrequent locations (count ≤ *threshold*) into ``"other"``.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``location`` column.
    threshold : int, optional
        Minimum listing count to keep a location name.

    Returns
    -------
    pd.DataFrame
        DataFrame with collapsed ``location`` values.
    """
    df = df.copy()
    df["location"] = df["location"].str.strip()
    counts = df["location"].value_counts()
    rare = counts[counts <= threshold].index
    df["location"] = df["location"].apply(lambda x: "other" if x in rare else x)
    logger.info(
        "group_rare_locations: %d rare locations → 'other'; %d distinct locations remain.",
        len(rare),
        df["location"].nunique(),
    )
    return df


def remove_sqft_per_bhk_outliers(df: pd.DataFrame, min_ratio: float = MIN_SQFT_PER_BHK) -> pd.DataFrame:
    """
    Remove listings where ``total_sqft / bhk < min_ratio``.

    These almost certainly represent data-entry errors (e.g. a 3 BHK flat
    at 200 sq ft total).

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``total_sqft`` and ``bhk`` columns.
    min_ratio : float, optional
        Minimum acceptable sqft per bedroom.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    before = len(df)
    mask = df["total_sqft"] / df["bhk"] >= min_ratio
    df = df[mask].reset_index(drop=True)
    logger.info("remove_sqft_per_bhk_outliers: removed %d rows.", before - len(df))
    return df


def remove_price_per_sqft_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove price-per-sqft outliers per location using a 1-sigma rule.

    For each location we retain only listings within one standard
    deviation of that location's mean price per sqft. This is a
    location-aware approach that avoids penalising genuinely
    expensive areas.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``location`` and ``price_per_sqft`` columns.

    Returns
    -------
    pd.DataFrame
        DataFrame with per-location outliers removed.
    """
    before = len(df)
    frames = []
    for _, sub in df.groupby("location"):
        mu = sub["price_per_sqft"].mean()
        sigma = sub["price_per_sqft"].std()
        frames.append(sub[(sub["price_per_sqft"] > mu - sigma) & (sub["price_per_sqft"] <= mu + sigma)])

    df_out = pd.concat(frames, ignore_index=True)
    logger.info("remove_price_per_sqft_outliers: removed %d rows.", before - len(df_out))
    return df_out


def remove_bhk_price_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove listings where a higher-BHK property is cheaper per sqft
    than a lower-BHK property in the same location.

    Rationale: a 3 BHK at a lower price-per-sqft than the average
    2 BHK in the same location is almost certainly a bad listing.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``location``, ``bhk``, and ``price_per_sqft``
        columns.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    before = len(df)
    exclude = np.array([], dtype=int)

    for _, loc_df in df.groupby("location"):
        bhk_stats: dict = {}
        for bhk, bhk_df in loc_df.groupby("bhk"):
            bhk_stats[bhk] = {
                "mean": bhk_df["price_per_sqft"].mean(),
                "count": len(bhk_df),
            }
        for bhk, bhk_df in loc_df.groupby("bhk"):
            prev = bhk_stats.get(bhk - 1)
            if prev and prev["count"] > 5:
                bad = bhk_df[bhk_df["price_per_sqft"] < prev["mean"]].index.values
                exclude = np.append(exclude, bad)

    df_out = df.drop(index=exclude).reset_index(drop=True)
    logger.info("remove_bhk_price_outliers: removed %d rows.", before - len(df_out))
    return df_out


def remove_bath_outliers(df: pd.DataFrame, max_over: int = MAX_BATH_OVER_BHK) -> pd.DataFrame:
    """
    Drop listings where bathrooms ≥ BHK + *max_over*.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with ``bath`` and ``bhk`` columns.
    max_over : int, optional
        Maximum allowed excess of bathrooms over BHK.

    Returns
    -------
    pd.DataFrame
        Filtered DataFrame.
    """
    before = len(df)
    df = df[df["bath"] < df["bhk"] + max_over].reset_index(drop=True)
    logger.info("remove_bath_outliers: removed %d rows.", before - len(df))
    return df


def encode_locations(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encode the ``location`` column, dropping the ``"other"``
    dummy to avoid multicollinearity.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with a ``location`` column.

    Returns
    -------
    pd.DataFrame
        DataFrame with location dummies appended and the original
        ``location`` column removed.
    """
    dummies = pd.get_dummies(df["location"])
    if "other" in dummies.columns:
        dummies = dummies.drop(columns=["other"])
    df = pd.concat([df.drop(columns=["location"]), dummies], axis=1)
    logger.info("encode_locations: added %d location dummy columns.", dummies.shape[1])
    return df


@timer
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline.

    Steps
    -----
    1. Add ``price_per_sqft``.
    2. Group rare locations → ``"other"``.
    3. Remove sqft-per-BHK outliers.
    4. Remove price-per-sqft outliers per location.
    5. Remove BHK price ordering violations.
    6. Remove bathroom outliers.
    7. Drop helper / redundant columns.
    8. One-hot encode locations.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned DataFrame from ``preprocessing.clean_data``.

    Returns
    -------
    pd.DataFrame
        Fully engineered DataFrame ready for model training.
    """
    logger.info("Starting feature engineering. Input shape: %s", df.shape)

    df = add_price_per_sqft(df)
    df = group_rare_locations(df)
    df = remove_sqft_per_bhk_outliers(df)
    df = remove_price_per_sqft_outliers(df)
    df = remove_bhk_price_outliers(df)
    df = remove_bath_outliers(df)

    # Drop columns no longer needed after engineering
    drop_cols = [c for c in ["size", "price_per_sqft"] if c in df.columns]
    df = df.drop(columns=drop_cols)

    df = encode_locations(df)
    logger.info("Feature engineering complete. Output shape: %s", df.shape)
    return df
