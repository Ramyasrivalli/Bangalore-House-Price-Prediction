# System Architecture

## Overview

The Bangalore House Price Prediction system is structured as a clean, layered ML pipeline. Each layer has a single responsibility and is independently testable.

```
┌─────────────────────────────────────────────────────────┐
│                     DATA LAYER                          │
│  data_loader.py  ── raw CSV ingestion + schema checks   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  PREPROCESSING LAYER                    │
│  preprocessing.py  ── null removal, BHK parsing, sqft  │
│                        normalisation                    │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              FEATURE ENGINEERING LAYER                  │
│  feature_engineering.py  ── outlier removal, location  │
│                              grouping, one-hot encoding │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│                  TRAINING LAYER                         │
│  model_training.py  ── GridSearchCV across 5 models,   │
│                         ShuffleSplit CV, serialisation  │
└────────────────────────┬────────────────────────────────┘
                         │
                ┌────────┴────────┐
                ▼                 ▼
┌───────────────────┐   ┌─────────────────────────────────┐
│  EVALUATION LAYER │   │        INFERENCE LAYER          │
│  model_eval.py    │   │  predictor.py  ── singleton     │
│  ── metrics, CV,  │   │  HousePricePredictor class      │
│     residuals     │   └──────────────┬──────────────────┘
└───────────────────┘                  │
                                       ▼
                          ┌────────────────────────┐
                          │     PRESENTATION LAYER │
                          │  app/app.py  ── Flask  │
                          │  REST API + HTML UI    │
                          └────────────────────────┘
```

## Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `config.py` | Single source of truth for all constants and paths |
| `utils.py` | Logging factory, timing decorator, JSON I/O helpers |
| `data_loader.py` | CSV ingestion, schema validation, column selection |
| `preprocessing.py` | Null removal, BHK extraction, sqft normalisation |
| `feature_engineering.py` | Outlier removal, location grouping, one-hot encoding |
| `model_training.py` | GridSearchCV, model selection, artifact serialisation |
| `model_evaluation.py` | Holdout metrics, cross-validation, residual analysis |
| `predictor.py` | Stateful inference wrapper (singleton pattern) |
| `app/app.py` | Flask REST API and HTML serving |

## Data Flow

```
Bengaluru_House_Data.csv (13,320 rows)
  └─▶ Column selection       → 13,320 × 5
  └─▶ Null removal           → ~13,246 × 5
  └─▶ BHK extraction         → ~13,246 × 6
  └─▶ Sqft normalisation     → ~13,200 × 6
  └─▶ price_per_sqft added   → ~13,200 × 7
  └─▶ Rare location grouping → ~13,200 × 7
  └─▶ Sqft/BHK outliers out  → ~12,400 × 7
  └─▶ PPS outlier removal    → ~11,200 × 7
  └─▶ BHK price anomalies    → ~10,800 × 7
  └─▶ Bath anomalies removed → ~10,750 × 7
  └─▶ One-hot encode locs    → ~10,750 × (7 + N_locs)
  └─▶ Train / Test split     →  80% / 20%
  └─▶ GridSearchCV           → Best model selected
  └─▶ trained_model.pkl + columns.json
```

## Artifact Contracts

### `models/trained_model.pkl`
Serialised scikit-learn estimator. Deserialisable with `pickle.load`.

### `models/columns.json`
```json
{
  "data_columns": ["total_sqft", "bath", "bhk", "1st Block Jayanagar", ...]
}
```
Column order must match training-time feature matrix.

## Inference Contract

```
Input:  location (str), sqft (float), bath (int), bhk (int)
Output: predicted_price (float, Lakhs ₹)
```

The predictor builds a zero-vector of length `len(data_columns)`, fills
positions 0-2 with sqft/bath/bhk, sets the one-hot index for the
requested location to 1, and calls `model.predict`.
