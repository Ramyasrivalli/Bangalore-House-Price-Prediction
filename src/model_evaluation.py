"""
model_evaluation.py
-------------------
Comprehensive evaluation utilities: regression metrics, cross-validation,
residual analysis, and human-readable reporting.
"""

from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import ShuffleSplit, cross_val_score, train_test_split

from src.config import CV_SPLITS, RANDOM_STATE, TEST_SIZE
from src.utils import get_logger

logger = get_logger(__name__)


def compute_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute a standard suite of regression metrics.

    Parameters
    ----------
    y_true : array-like
        Ground-truth target values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    dict
        Keys: ``r2``, ``mae``, ``rmse``, ``mape`` (percent).
    """
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = mean_absolute_percentage_error(y_true, y_pred) * 100

    metrics = {"r2": round(r2, 4), "mae": round(mae, 2), "rmse": round(rmse, 2), "mape": round(mape, 2)}
    logger.info("Metrics → R²=%.4f  MAE=%.2f  RMSE=%.2f  MAPE=%.2f%%", r2, mae, rmse, mape)
    return metrics


def evaluate_on_holdout(model: Any, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Evaluate *model* on an 80/20 hold-out split.

    Parameters
    ----------
    model : sklearn estimator
        Fitted estimator with a ``predict`` method.
    X : pd.DataFrame
        Full feature matrix.
    y : pd.Series
        Full target vector.

    Returns
    -------
    dict
        Regression metrics computed on the test split.
    """
    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_pred = model.predict(X_test)
    logger.info("Holdout evaluation on %d samples:", len(y_test))
    return compute_regression_metrics(y_test.values, y_pred)


def cross_validate_model(model: Any, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Run k-fold cross-validation and report mean ± std.

    Parameters
    ----------
    model : sklearn estimator
        Unfitted (or fitted) estimator.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    dict
        Keys: ``cv_scores`` (list), ``mean_cv_score``, ``std_cv_score``.
    """
    cv = ShuffleSplit(n_splits=CV_SPLITS, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    result = {
        "cv_scores": [round(s, 4) for s in scores.tolist()],
        "mean_cv_score": round(scores.mean(), 4),
        "std_cv_score": round(scores.std(), 4),
    }
    logger.info(
        "Cross-validation R² = %.4f ± %.4f  (splits: %s)",
        scores.mean(),
        scores.std(),
        [f"{s:.4f}" for s in scores],
    )
    return result


def residual_analysis(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    Compute residual statistics useful for detecting systematic bias.

    Parameters
    ----------
    y_true : array-like
        Ground-truth values.
    y_pred : array-like
        Model predictions.

    Returns
    -------
    dict
        Residual mean, std, skewness, and percentile table.
    """
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    percentiles = np.percentile(residuals, [5, 25, 50, 75, 95])

    return {
        "residual_mean": round(float(residuals.mean()), 4),
        "residual_std": round(float(residuals.std()), 4),
        "residual_skew": round(float(pd.Series(residuals).skew()), 4),
        "percentiles": {
            "p5": round(percentiles[0], 2),
            "p25": round(percentiles[1], 2),
            "p50": round(percentiles[2], 2),
            "p75": round(percentiles[3], 2),
            "p95": round(percentiles[4], 2),
        },
    }


def full_evaluation_report(model: Any, X: pd.DataFrame, y: pd.Series) -> dict:
    """
    Generate a complete evaluation report combining hold-out metrics,
    cross-validation, and residual analysis.

    Parameters
    ----------
    model : sklearn estimator
        Fitted estimator.
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target vector.

    Returns
    -------
    dict
        Nested dict with ``holdout_metrics``, ``cross_validation``,
        and ``residual_analysis`` sections.
    """
    logger.info("Generating full evaluation report …")

    holdout = evaluate_on_holdout(model, X, y)

    _, X_test, _, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_pred = model.predict(X_test)

    cv = cross_validate_model(model, X, y)
    residuals = residual_analysis(y_test.values, y_pred)

    report = {
        "holdout_metrics": holdout,
        "cross_validation": cv,
        "residual_analysis": residuals,
    }
    logger.info("Evaluation report generated.")
    return report
