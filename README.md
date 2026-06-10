<div align="center">

# 🏠 Bangalore House Price Prediction

**A production-grade end-to-end Machine Learning system for predicting residential property prices in Bangalore, India.**

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-F7931E?style=flat-square&logo=scikitlearn&logoColor=white)](https://scikit-learn.org)
[![Flask](https://img.shields.io/badge/Flask-3.0%2B-000000?style=flat-square&logo=flask)](https://flask.palletsprojects.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Tests](https://img.shields.io/badge/Tests-40%20cases-brightgreen?style=flat-square)](#testing)

[Overview](#overview) · [Dataset](#dataset) · [Architecture](#architecture) · [Results](#results) · [Quick Start](#quick-start) · [Web App](#web-application) · [Docs](#documentation)

</div>

---

## Overview

This project transforms a standard Kaggle dataset into a **production-ready ML system** that would be at home in an industry codebase. It demonstrates the full lifecycle:

```
Raw CSV  →  Cleaning  →  Feature Engineering  →  Model Selection  →  REST API  →  Web UI
```

Rather than a single monolithic notebook, the system is architected as a proper Python package with modular responsibilities, unit tests, configuration management, logging, and a Flask web application — the same patterns used in production ML at scale.

### Highlights

- **R² ≈ 0.848** on held-out test data (MAE ≈ ₹18.5 Lakhs)
- **5-fold GridSearchCV** across 5 algorithms with domain-tuned hyperparameter grids
- **Multi-stage outlier removal** using domain knowledge (per-location σ-banding, BHK price ordering)
- **7 modular source files** following PEP-8 with full type hints and NumPy docstrings
- **40 unit tests** covering preprocessing, training utilities, and the inference pipeline
- **Flask REST API** + modern dark-themed web UI

---

## Problem Statement

Bangalore's real estate market is deeply fragmented. A 2 BHK in Indiranagar can list for 5× the price of an identical flat in Whitefield. This information asymmetry disadvantages buyers and creates pricing inefficiencies. This project builds a **data-driven price estimator** that gives any stakeholder a transparent, objective benchmark — accessible through a browser.

---

## Dataset

| Property | Value |
|----------|-------|
| **Source** | [Kaggle – Bengaluru House Price Data](https://www.kaggle.com/datasets/amitabhajoy/bengaluru-house-price-data) |
| **Raw size** | 13,320 rows × 9 columns |
| **Final training size** | ~10,750 rows |
| **Target** | `price` (₹ Lakhs, float) |
| **Price range** | ₹8 L – ₹3,600 L |
| **Median price** | ₹72 L |
| **Distinct locations** | ~1,300 raw → ~242 after grouping |

### Dataset Statistics

```
          bath        balcony       price
count  13247.00      12711.00    13320.00
mean       2.69          1.58      112.57
std        1.34          0.82      148.97
min        1.00          0.00        8.00
25%        2.00          1.00       50.00
50%        2.00          2.00       72.00
75%        3.00          2.00      120.00
max       40.00          3.00     3600.00
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      TRAINING PIPELINE                      │
│                                                             │
│  data_loader.py ──▶ preprocessing.py ──▶ feature_eng.py    │
│       │                                        │            │
│  Schema check      Null removal           Outlier removal   │
│  Column select     BHK extraction         Location grouping │
│                    sqft normalise          One-hot encode    │
│                                                │            │
│                               model_training.py             │
│                          GridSearchCV (5 models × k-fold)   │
│                               │              │              │
│                          models/        model_eval.py       │
│                         *.pkl  *.json   R² / MAE / RMSE     │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    INFERENCE PIPELINE                       │
│                                                             │
│   predictor.py  (HousePricePredictor singleton)             │
│        │                                                    │
│        ▼                                                    │
│   app/app.py  (Flask REST API)                              │
│        │                                                    │
│        ▼                                                    │
│   app/templates/index.html  (Browser UI)                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Technology Stack

| Category | Technology |
|----------|-----------|
| Language | Python 3.10+ |
| Data wrangling | pandas 2.x · NumPy |
| ML framework | scikit-learn 1.3+ |
| Web framework | Flask 3.x |
| Serialisation | pickle (stdlib) |
| Testing | pytest · pytest-cov |
| Code quality | black · flake8 · isort |

---

## Results

### Model Comparison

| Rank | Model | CV R² | Notes |
|------|-------|-------|-------|
| 🥇 1 | **Linear Regression** | **~0.847** | Best CV score, lowest variance |
| 🥈 2 | Ridge Regression | ~0.846 | Comparable; slightly regularised |
| 🥉 3 | Lasso Regression | ~0.843 | Induces sparsity |
| 4 | Random Forest | ~0.841 | Higher variance across folds |
| 5 | Decision Tree | ~0.735 | Overfits without max_depth |

### Final Metrics (Hold-out Test Set, 20%)

| Metric | Value |
|--------|-------|
| **R²** | **~0.848** |
| **MAE** | **~₹18.5 Lakhs** |
| **RMSE** | **~₹31.2 Lakhs** |
| **MAPE** | **~24.1%** |
| **CV R² (mean ± std)** | **0.847 ± 0.006** |

> Linear Regression outperforms tree-based models because price-area relationships are approximately linear within location groups after domain-specific outlier removal.

---

## Quick Start

### Prerequisites

- Python 3.10+
- Git

### 1. Clone

```bash
git clone https://github.com/Ramyasrivalli/Bangalore-House-Price-Prediction.git
cd Bangalore-House-Price-Prediction
```

### 2. Install dependencies

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows (Git Bash):
source .venv/Scripts/activate

pip install -r requirements.txt
pip install -e .
```

### 3. Train the model

```bash
python train.py
```

This runs the full pipeline and saves `models/trained_model.pkl` + `models/columns.json`.

Expected output:

```
2025-01-01 12:00:00 | INFO     | train | ============================================================
2025-01-01 12:00:00 | INFO     | train | Bangalore House Price Prediction — Training Pipeline
...
2025-01-01 12:00:08 | INFO     | train | Best model: linear_regression (R²=0.8475)
2025-01-01 12:00:09 | INFO     | train | Final evaluation:
2025-01-01 12:00:09 | INFO     | train |   R²   = 0.8482
2025-01-01 12:00:09 | INFO     | train |   MAE  = 18.51 Lakhs
2025-01-01 12:00:09 | INFO     | train |   RMSE = 31.24 Lakhs
```

### 4. Run tests

```bash
pytest tests/ -v --cov=src --cov-report=term-missing
```

### 5. Use the predictor programmatically

```python
from src.predictor import HousePricePredictor

predictor = HousePricePredictor()

price = predictor.predict(
    location="Indiranagar",
    sqft=1200,
    bath=2,
    bhk=2,
)
print(f"Estimated price: ₹{price:.2f} Lakhs")
# → Estimated price: ₹95.43 Lakhs
```

---

## Web Application

### Launch

```bash
python app/app.py
```

Then open [http://localhost:5000](http://localhost:5000) in your browser.

### Features

- **Location dropdown** – all ~242 recognised Bangalore localities
- **Numeric inputs** – area (sq ft), BHK, bathrooms
- **Instant prediction** – async fetch, no page reload
- **Dual display** – price in Lakhs and Crores
- **Client + server validation** – graceful error messages
- **Health check** – `GET /health`
- **JSON API** – `POST /predict` (see below)

### REST API

**Predict**

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"location": "Whitefield", "sqft": 1500, "bath": 2, "bhk": 3}'
```

```json
{ "price": 87.24 }
```

**List locations**

```bash
curl http://localhost:5000/locations
```

---

## Repository Structure

```
Bangalore-House-Price-Prediction/
├── README.md                    ← You are here
├── LICENSE                      ← MIT
├── requirements.txt
├── setup.py
├── .gitignore
├── CONTRIBUTING.md
├── project_report.md            ← Full report + resume content
├── train.py                     ← CLI entry point
│
├── data/
│   └── Bengaluru_House_Data.csv
│
├── notebooks/
│   └── bangalore_home_prices_final.ipynb
│
├── src/
│   ├── __init__.py
│   ├── config.py                ← All constants & paths
│   ├── utils.py                 ← Logger, timer, JSON helpers
│   ├── data_loader.py           ← CSV ingestion + validation
│   ├── preprocessing.py         ← Cleaning pipeline
│   ├── feature_engineering.py   ← Outliers, encoding
│   ├── model_training.py        ← GridSearchCV + serialisation
│   ├── model_evaluation.py      ← Metrics, CV, residuals
│   └── predictor.py             ← Inference singleton
│
├── models/
│   ├── trained_model.pkl        ← Generated by train.py
│   └── columns.json             ← Feature column list
│
├── tests/
│   ├── __init__.py
│   ├── test_preprocessing.py    ← 18 test cases
│   ├── test_training.py         ← 12 test cases
│   └── test_prediction.py       ← 12 test cases
│
├── app/
│   ├── app.py                   ← Flask application
│   └── templates/
│       └── index.html           ← Dark-themed web UI
│
└── docs/
    ├── architecture.md
    ├── methodology.md
    └── results.md
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [docs/architecture.md](docs/architecture.md) | System design, module responsibilities, data flow |
| [docs/methodology.md](docs/methodology.md) | Full ML methodology: preprocessing, feature engineering, model selection |
| [docs/results.md](docs/results.md) | Model comparison, evaluation metrics, key observations |
| [project_report.md](project_report.md) | Complete project report + resume/LinkedIn content |

---

## Testing

```bash
# Run all tests with coverage
pytest tests/ -v --cov=src --cov-report=term-missing

# Run a specific test file
pytest tests/test_preprocessing.py -v

# Run with HTML coverage report
pytest tests/ --cov=src --cov-report=html
open htmlcov/index.html
```

**Test coverage** spans:
- `test_preprocessing.py` – null removal, BHK parsing, sqft normalisation
- `test_training.py` – feature/target split, metrics computation, CV, model I/O
- `test_prediction.py` – predictor interface, input validation, artifact loading

---

## Future Enhancements

- [ ] Add XGBoost / LightGBM to the model registry
- [ ] Integrate SHAP for prediction explanations in the UI
- [ ] Add a data drift detector for production monitoring
- [ ] Containerise with Docker + `docker-compose.yml`
- [ ] Set up GitHub Actions CI (lint → test → build)
- [ ] Add MLflow experiment tracking
- [ ] Scrape updated Bangalore listings for periodic retraining

##-------------------------**********---------------------------------------------------##


🏠 How to Run – Bangalore House Price Predictor
> **Important:** Never open `index.html` directly in your browser.  
> This app needs a Flask server to work. Follow the steps below.
---
✅ Prerequisites
Make sure you have installed:
Python 3.10+ → Download here  
(During install on Windows: tick ✅ "Add Python to PATH")
Git (optional, only needed if cloning) → Download here
---
📁 Step 1 — Open the Project Folder in Terminal
Option A: Using VS Code
Open the folder `Bangalore-House-Price-Prediction` in VS Code
Press Ctrl + ` (backtick) to open the integrated terminal
Make sure the terminal shows this path:
```
   C:\...\Bangalore-House-Price-Prediction>
   ```
If it shows the parent folder, run:
```bash
   cd Bangalore-House-Price-Prediction
   ```
Option B: Using Windows PowerShell / CMD
Open PowerShell or Command Prompt
Navigate to the project:
```bash
   cd C:\Users\YourName\Downloads\Bangalore-House-Price-Prediction_1\Bangalore-House-Price-Prediction
   ```
---
📦 Step 2 — Install Dependencies
Run this once (you only need to do this the first time):
```bash
pip install -r requirements.txt
```
If that doesn't work, install manually:
```bash
pip install flask scikit-learn numpy pandas
```
---
🧠 Step 3 — Train the Model
Run this once to generate the ML model file:
```bash
python train.py
```
⏳ This takes 2–5 minutes. You'll see logs like:
```
INFO | Training Pipeline
INFO | Loaded 13320 rows
INFO | Best model: linear_regression (R²=0.847)
INFO | Model saved to models/trained_model.pkl ✅
```
> **Skip this step** if `models/trained_model.pkl` already exists in the folder.
---
🚀 Step 4 — Start the Web Server
```bash
python app/app.py
```
You should see:
```
 * Running on http://127.0.0.1:5000
 * Press CTRL+C to quit
```
---
🌐 Step 5 — Open the App in Browser
Open your browser and go to:
```
http://127.0.0.1:5000
```
> ❌ Do NOT open the `index.html` file directly  
> ✅ Always use `http://127.0.0.1:5000` after starting the server
---
🎯 How to Use the App
Field	Example Value
Location	Indiranagar
Total Area (sq ft)	1200
BHK	2
Bathrooms	2
Click Predict Price → You'll see the estimated price in ₹ Lakhs and Crores.
---
🛑 How to Stop the Server
Press Ctrl + C in the terminal.
---
❓ Common Issues
Problem	Fix
`python: command not found`	Install Python and tick "Add to PATH" during setup
`No module named flask`	Run `pip install flask scikit-learn numpy pandas`
`No such file train.py`	You're in the wrong folder — run `cd Bangalore-House-Price-Prediction`
`trained_model.pkl not found`	Run `python train.py` first
Location shows `{{ loc }}`	You opened the HTML file directly — use `http://127.0.0.1:5000` instead
Price shows ₹0.00	Input values are unrealistic (e.g. 88 BHK) — use normal values
---
📋 Quick Summary (All Commands)
```bash
# 1. Go to project folder
cd Bangalore-House-Price-Prediction

# 2. Install dependencies (first time only)
pip install -r requirements.txt

# 3. Train the model (first time only)
python train.py

# 4. Start the server
python app/app.py

# 5. Open browser → http://127.0.0.1:5000
```
---
