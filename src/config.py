"""
config.py
---------
Centralized configuration management for the Bangalore House Price Prediction project.
All tunable parameters, file paths, and model hyperparameters live here.
"""

from pathlib import Path

# ── Project root ──────────────────────────────────────────────────────────────
ROOT_DIR = Path(__file__).resolve().parent.parent

# ── Data paths ────────────────────────────────────────────────────────────────
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "Bengaluru_House_Data.csv"

# ── Model artifact paths ──────────────────────────────────────────────────────
MODELS_DIR = ROOT_DIR / "models"
MODEL_PATH = MODELS_DIR / "trained_model.pkl"
COLUMNS_PATH = MODELS_DIR / "columns.json"

# ── Preprocessing constants ───────────────────────────────────────────────────
COLUMNS_TO_DROP = ["area_type", "society", "balcony", "availability"]
MIN_SQFT_PER_BHK = 300          # Minimum acceptable sqft per bedroom
MAX_BATH_OVER_BHK = 2           # Bathrooms must be < BHK + this threshold
RARE_LOCATION_THRESHOLD = 10    # Locations with ≤ this count → "other"
PRICE_SCALE_FACTOR = 100_000    # price column is in lakhs

# ── Feature engineering ───────────────────────────────────────────────────────
TARGET_COLUMN = "price"
NUMERIC_FEATURES = ["total_sqft", "bath", "bhk"]

# ── Model training ────────────────────────────────────────────────────────────
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_SPLITS = 5

# Hyperparameter search spaces
PARAM_GRIDS = {
    "linear_regression": {
        "fit_intercept": [True, False],
    },
    "lasso": {
        "alpha": [0.1, 0.5, 1.0, 2.0, 5.0],
        "selection": ["random", "cyclic"],
        "max_iter": [5000],
    },
    "ridge": {
        "alpha": [0.1, 1.0, 5.0, 10.0, 50.0],
        "fit_intercept": [True, False],
    },
    "decision_tree": {
        "criterion": ["squared_error", "absolute_error"],
        "splitter": ["best", "random"],
        "max_depth": [None, 5, 10, 20],
    },
    "random_forest": {
        "n_estimators": [50, 100],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────
LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_LEVEL = "INFO"

# ── Flask app ─────────────────────────────────────────────────────────────────
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 5000
FLASK_DEBUG = False
