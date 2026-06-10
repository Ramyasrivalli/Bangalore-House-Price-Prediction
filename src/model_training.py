"""
model_training.py
-----------------
Model selection, hyperparameter tuning, and serialisation.

Trains multiple regression algorithms via GridSearchCV, picks the best,
and persists the trained model + feature column list to disk.
"""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from src.config import (
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    CV_SPLITS,
    PARAM_GRIDS,
    MODEL_PATH,
    COLUMNS_PATH,
)
from src.utils import ensure_dir, get_logger, save_json, timer

logger = get_logger(__name__)

MODEL_REGISTRY: dict[str, Any] = {
    "linear_regression": LinearRegression(),
    "lasso": Lasso(max_iter=5000),
    "ridge": Ridge(),
    "decision_tree": DecisionTreeRegressor(random_state=RANDOM_STATE),
    "random_forest": RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1),
}


def split_features_target(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate feature matrix from the target vector.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered DataFrame.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        ``(X, y)`` pair where ``X`` contains all columns except the
        target and ``y`` contains the target column.

    Raises
    ------
    KeyError
        If ``TARGET_COLUMN`` is not present in *df*.
    """
    if TARGET_COLUMN not in df.columns:
        raise KeyError(f"Target column '{TARGET_COLUMN}' not found in DataFrame.")

    X = df.drop(columns=[TARGET_COLUMN])
    y = df[TARGET_COLUMN]
    logger.info("Feature matrix shape: %s  |  Target shape: %s", X.shape, y.shape)
    return X, y


def find_best_model(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    Grid search across all configured models and return a comparison table.

    Uses ``ShuffleSplit`` cross-validation so results are stable across
    different dataset sizes.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    pd.DataFrame
        Sorted table with columns ``model``, ``best_score``,
        ``best_params`` for every algorithm.
    """
    cv = ShuffleSplit(n_splits=CV_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    results = []

    for name, model in MODEL_REGISTRY.items():
        params = PARAM_GRIDS.get(name, {})
        gs = GridSearchCV(model, params, cv=cv, scoring="r2", return_train_score=False, n_jobs=-1)
        gs.fit(X, y)
        results.append(
            {
                "model": name,
                "best_score": round(gs.best_score_, 4),
                "best_params": gs.best_params_,
            }
        )
        logger.info("  %-20s  R²=%.4f  params=%s", name, gs.best_score_, gs.best_params_)

    df_results = pd.DataFrame(results).sort_values("best_score", ascending=False).reset_index(drop=True)
    return df_results


@timer
def train(df: pd.DataFrame) -> tuple[Any, list[str], dict]:
    """
    Full training pipeline.

    1. Split into features / target.
    2. Hold out a test set for final evaluation.
    3. Grid-search across all models.
    4. Re-train the winner on the full training split.
    5. Serialise model + feature columns.

    Parameters
    ----------
    df : pd.DataFrame
        Fully engineered DataFrame.

    Returns
    -------
    tuple[Any, list[str], dict]
        ``(trained_model, feature_columns, comparison_table_dict)``
    """
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    logger.info("Train/test split: %d train | %d test", len(X_train), len(X_test))

    logger.info("Running grid search across %d models …", len(MODEL_REGISTRY))
    comparison = find_best_model(X_train, y_train)
    best_name = comparison.iloc[0]["model"]
    best_params = comparison.iloc[0]["best_params"]
    logger.info("Best model: %s (R²=%.4f)", best_name, comparison.iloc[0]["best_score"])

    # Re-train best model on full training data with optimal hyperparams
    best_model = MODEL_REGISTRY[best_name].set_params(**best_params)
    best_model.fit(X_train, y_train)

    # Persist artifacts
    feature_columns = list(X.columns)
    _save_model(best_model, MODEL_PATH)
    _save_columns(feature_columns, COLUMNS_PATH)

    return best_model, feature_columns, comparison.to_dict(orient="records")


def _save_model(model: Any, path: Path) -> None:
    """Pickle a trained model to *path*."""
    ensure_dir(path.parent)
    with open(path, "wb") as fh:
        pickle.dump(model, fh)
    logger.info("Model saved → %s", path)


def _save_columns(columns: list[str], path: Path) -> None:
    """Persist feature column list as JSON."""
    save_json({"data_columns": columns}, path)
    logger.info("Columns saved → %s", path)


def load_model(path: Path = MODEL_PATH) -> Any:
    """
    Load a previously serialised model.

    Parameters
    ----------
    path : Path, optional
        Path to the pickle file.

    Returns
    -------
    Any
        Deserialised scikit-learn estimator.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    with open(path, "rb") as fh:
        model = pickle.load(fh)
    logger.info("Loaded model from '%s'", path)
    return model
