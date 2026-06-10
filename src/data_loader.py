"""
data_loader.py
--------------
Responsible for ingesting raw CSV data with schema validation.
"""

from pathlib import Path

import pandas as pd

from src.config import RAW_DATA_PATH, COLUMNS_TO_DROP
from src.utils import get_logger, timer

logger = get_logger(__name__)

EXPECTED_COLUMNS = {
    "area_type",
    "availability",
    "location",
    "size",
    "society",
    "total_sqft",
    "bath",
    "balcony",
    "price",
}


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load the raw Bengaluru house price CSV.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file. Defaults to the configured ``RAW_DATA_PATH``.

    Returns
    -------
    pd.DataFrame
        Raw, unmodified DataFrame.

    Raises
    ------
    FileNotFoundError
        If the CSV file does not exist at *path*.
    ValueError
        If the CSV is missing expected columns.
    """
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {path}")

    logger.info("Loading raw data from '%s'", path)
    df = pd.read_csv(path)

    missing_cols = EXPECTED_COLUMNS - set(df.columns)
    if missing_cols:
        raise ValueError(f"CSV is missing expected columns: {missing_cols}")

    logger.info("Loaded %d rows × %d columns", *df.shape)
    return df


@timer
def load_and_select(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """
    Load raw data and drop irrelevant columns.

    Parameters
    ----------
    path : Path, optional
        Path to the CSV file.

    Returns
    -------
    pd.DataFrame
        DataFrame with only modelling-relevant columns retained.
    """
    df = load_raw_data(path)
    df = df.drop(columns=COLUMNS_TO_DROP, errors="ignore")
    logger.info("After column selection: %d rows × %d columns", *df.shape)
    return df


def get_data_summary(df: pd.DataFrame) -> dict:
    """
    Return a concise summary dictionary for a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.

    Returns
    -------
    dict
        Keys: ``shape``, ``dtypes``, ``null_counts``, ``numeric_stats``.
    """
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "null_counts": df.isnull().sum().to_dict(),
        "numeric_stats": df.describe().to_dict(),
    }
