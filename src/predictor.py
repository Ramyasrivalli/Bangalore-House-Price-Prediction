"""
predictor.py
------------
Production inference pipeline.

Loads the serialised model and feature columns once at startup, then
exposes a clean ``predict`` interface used by both the Flask API and
any programmatic caller.
"""

from pathlib import Path
from typing import Optional, Any

import numpy as np

from src.config import MODEL_PATH, COLUMNS_PATH
from src.model_training import load_model
from src.utils import get_logger, load_json

logger = get_logger(__name__)


class HousePricePredictor:
    """
    Stateful inference wrapper around the serialised regression model.

    Attributes
    ----------
    model : sklearn estimator
        Loaded, fitted estimator.
    data_columns : list[str]
        Ordered feature column names expected by the model.

    Examples
    --------
    >>> predictor = HousePricePredictor()
    >>> price = predictor.predict("Indira Nagar", sqft=1200, bath=2, bhk=2)
    >>> print(f"Estimated price: ₹{price:.2f} Lakhs")
    """

    def __init__(
        self,
        model_path: Path = MODEL_PATH,
        columns_path: Path = COLUMNS_PATH,
    ) -> None:
        """
        Load model and feature metadata from disk.

        Parameters
        ----------
        model_path : Path, optional
            Path to the pickled model file.
        columns_path : Path, optional
            Path to the columns JSON file.

        Raises
        ------
        FileNotFoundError
            If either artifact is missing (run training first).
        """
        self.model: Any = load_model(model_path)
        columns_data: dict = load_json(columns_path)
        self.data_columns: list[str] = columns_data["data_columns"]
        logger.info(
            "HousePricePredictor initialised. Feature dimensionality: %d",
            len(self.data_columns),
        )

    @property
    def locations(self) -> list[str]:
        """Return sorted list of recognised location names."""
        known = [c for c in self.data_columns if c not in ("total_sqft", "bath", "bhk")]
        return sorted(known)

    def predict(
        self,
        location: str,
        sqft: float,
        bath: int,
        bhk: int,
    ) -> float:
        """
        Predict house price in Indian Rupees (Lakhs).

        Parameters
        ----------
        location : str
            Recognised location string (case-sensitive).
        sqft : float
            Total area in square feet.
        bath : int
            Number of bathrooms.
        bhk : int
            Number of bedrooms (BHK).

        Returns
        -------
        float
            Predicted price in Lakhs (₹).

        Raises
        ------
        ValueError
            If *sqft* ≤ 0 or *bath* ≤ 0 or *bhk* ≤ 0.
        """
        self._validate_inputs(sqft, bath, bhk)

        x = np.zeros(len(self.data_columns))
        x[0] = sqft
        x[1] = bath
        x[2] = bhk

        # One-hot encode location
        if location in self.data_columns:
            loc_idx = self.data_columns.index(location)
            x[loc_idx] = 1
        else:
            logger.warning("Location '%s' not in training data; defaulting to 'other'.", location)

        price: float = self.model.predict([x])[0]
        logger.debug(
            "Prediction: location=%s sqft=%.0f bath=%d bhk=%d → ₹%.2f L",
            location, sqft, bath, bhk, price,
        )
        return round(max(price, 0.0), 2)

    @staticmethod
    def _validate_inputs(sqft: float, bath: int, bhk: int) -> None:
        """Raise ValueError for nonsensical input values."""
        if sqft <= 0:
            raise ValueError(f"sqft must be positive, got {sqft}")
        if bath <= 0:
            raise ValueError(f"bath must be positive, got {bath}")
        if bhk <= 0:
            raise ValueError(f"bhk must be positive, got {bhk}")
        if bath > bhk + 2:
            raise ValueError(
                f"Suspicious input: {bath} bathrooms for {bhk} BHK. "
                "Bathroom count should be ≤ BHK + 2."
            )


# Module-level singleton – load once, reuse across Flask requests
_predictor: Optional[HousePricePredictor] = None


def get_predictor() -> HousePricePredictor:
    """
    Return the module-level singleton predictor, initialising it on
    first call.

    Returns
    -------
    HousePricePredictor
        Ready-to-use predictor instance.
    """
    global _predictor
    if _predictor is None:
        _predictor = HousePricePredictor()
    return _predictor
