#!/usr/bin/env python3
"""
train.py
--------
Entry-point script for the end-to-end training pipeline.

Usage
-----
    python train.py

Runs data loading → cleaning → feature engineering → model selection →
evaluation, then writes trained_model.pkl and columns.json to models/.
"""

import json
import sys
from pathlib import Path

# Make the repo root importable
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.data_loader import load_and_select
from src.preprocessing import clean_data
from src.feature_engineering import engineer_features
from src.model_training import train, split_features_target
from src.model_evaluation import full_evaluation_report
from src.utils import get_logger, ensure_dir, save_json
from src.config import MODELS_DIR

logger = get_logger("train")


def main() -> None:
    logger.info("=" * 60)
    logger.info("Bangalore House Price Prediction — Training Pipeline")
    logger.info("=" * 60)

    # 1. Load
    df_raw = load_and_select()

    # 2. Clean
    df_clean = clean_data(df_raw)

    # 3. Feature engineering
    df_engineered = engineer_features(df_clean)

    # 4. Train (grid search + serialise)
    model, feature_columns, comparison = train(df_engineered)

    # 5. Evaluate
    X, y = split_features_target(df_engineered)
    report = full_evaluation_report(model, X, y)

    # 6. Log summary
    logger.info("-" * 60)
    logger.info("Model comparison:")
    for row in comparison:
        logger.info("  %-20s  R²=%.4f", row["model"], row["best_score"])

    logger.info("-" * 60)
    logger.info("Final evaluation:")
    hm = report["holdout_metrics"]
    logger.info("  R²   = %.4f", hm["r2"])
    logger.info("  MAE  = %.2f Lakhs", hm["mae"])
    logger.info("  RMSE = %.2f Lakhs", hm["rmse"])
    logger.info("  MAPE = %.2f%%", hm["mape"])

    cv = report["cross_validation"]
    logger.info("  CV R² = %.4f ± %.4f", cv["mean_cv_score"], cv["std_cv_score"])

    # 7. Persist evaluation report
    ensure_dir(MODELS_DIR)
    save_json(report, MODELS_DIR / "evaluation_report.json")
    save_json({"comparison": comparison}, MODELS_DIR / "model_comparison.json")

    logger.info("=" * 60)
    logger.info("Training complete. Artifacts saved to models/")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
