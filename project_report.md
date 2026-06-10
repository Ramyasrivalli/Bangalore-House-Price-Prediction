# Project Report: Bangalore House Price Prediction 
**GitHub:** [Bangalore-House-Price-Prediction](https://github.com/Ramyasrivalli/Bangalore-House-Price-Prediction)

---

## 1. Executive Summary

This project delivers a production-grade machine learning system that predicts residential property prices in Bangalore, India. Starting from a raw CSV dataset of 13,320 listings scraped from real-estate platforms, the system implements an end-to-end pipeline: data ingestion, multi-stage cleaning, feature engineering with domain-specific outlier removal, cross-validated model selection across five algorithms, and a Flask-based web application for interactive predictions.

The best model (Linear Regression with optimal hyperparameters) achieves **R² ≈ 0.847** on a held-out test set, with a mean absolute error of approximately **₹18.5 Lakhs**. The complete project follows industry engineering standards: modular Python packages, PEP-8 compliance, type hints, docstrings, logging, unit tests, and configuration management.

---

## 2. Problem Statement

Real estate pricing in Bangalore is highly heterogeneous. A 2 BHK in Indiranagar can cost five times more than an identical flat in an outer-ring locality. Buyers, sellers, and agents lack objective pricing benchmarks, leading to information asymmetry. This project addresses that gap by building a data-driven price estimator accessible through a web interface.

**Input features:**
- Location (one of ~242 recognised areas)
- Total area (sq ft)
- Number of bedrooms (BHK)
- Number of bathrooms

**Output:** Predicted price in Indian Rupees (Lakhs)

---

## 3. Dataset

| Attribute | Value |
|-----------|-------|
| Name | Bengaluru House Price Dataset |
| Source | Kaggle |
| Format | CSV |
| Raw rows | 13,320 |
| Raw columns | 9 |
| Target variable | `price` (₹ Lakhs, float64) |
| Price range | ₹8 L – ₹3,600 L (mean: ₹112 L, median: ₹72 L) |
| Distinct locations (raw) | ~1,300 |
| Missing values | ~550 rows (< 5%) |

**Columns:**

| Column | Type | Description |
|--------|------|-------------|
| area_type | object | Super built-up / Built-up / Plot / Carpet |
| availability | object | Possession date or "Ready To Move" |
| location | object | Neighbourhood name |
| size | object | "2 BHK", "4 Bedroom", etc. |
| society | object | Residential society name |
| total_sqft | object | Area – plain number or range string |
| bath | int/float | Number of bathrooms |
| balcony | int/float | Number of balconies |
| price | float | **Target** – price in ₹ Lakhs |

---

## 4. Methodology Summary

### 4.1 Data Cleaning

- Dropped four irrelevant columns: `area_type`, `society`, `balcony`, `availability`
- Removed rows with null values
- Parsed `size` → integer `bhk` column
- Normalised `total_sqft`: range strings converted to arithmetic mean; non-numeric rows dropped

### 4.2 Feature Engineering

- **Derived feature**: `price_per_sqft = price × 100,000 / total_sqft`
- **Location grouping**: locations with ≤ 10 listings → `"other"` (reduces from ~1,300 to ~242 categories)
- **Outlier removal (3 stages)**:
  1. sqft per BHK < 300 → remove
  2. Per-location price-per-sqft outside ±1σ → remove
  3. BHK price ordering violations within location → remove
  4. Bathrooms ≥ BHK + 2 → remove
- **One-hot encoding** of location (dropping `"other"` dummy)

### 4.3 Model Training

Five algorithms evaluated via 5-fold ShuffleSplit GridSearchCV:
Linear Regression, Lasso, Ridge, Decision Tree, Random Forest.

Best performer: **Linear Regression** (CV R² ≈ 0.847)

### 4.4 Evaluation

- Holdout test set (20%): R² ≈ 0.848, MAE ≈ ₹18.5 L, RMSE ≈ ₹31.2 L
- Cross-validation: R² = 0.847 ± 0.006 (stable generalisation)

---

## 5. Technology Stack

| Layer | Technology |
|-------|-----------|
| Language | Python 3.10+ |
| Data processing | Pandas 2.x, NumPy |
| ML framework | scikit-learn 1.3+ |
| Web framework | Flask 3.x |
| Serialisation | pickle (stdlib) |
| Testing | pytest + pytest-cov |
| Code quality | black, flake8, isort |
| Experiment tracking | Manual JSON reports |

---

## 6. Repository Structure

```
Bangalore-House-Price-Prediction/
├── src/           # Core ML modules (7 files)
├── app/           # Flask application + HTML UI
├── tests/         # Unit tests (3 test files, ~40 test cases)
├── docs/          # Architecture, methodology, results
├── data/          # Raw dataset
├── models/        # Serialised artifacts
├── notebooks/     # Original EDA notebook
└── train.py       # CLI training entry point
```

---

## 7. Key Learnings

1. **Outlier removal strategy matters more than algorithm choice**: Removing domain-specific outliers (sqft-per-BHK, per-location price anomalies, BHK ordering violations) improved R² more than switching from Linear Regression to an ensemble method.

2. **Location encoding is the dominant signal**: One-hot encoded location features account for the majority of explained variance. Simple linear models can exploit this efficiently.

3. **Cross-validation stability > peak CV score**: The Random Forest achieved comparable peak CV R² but with higher variance across folds, making Linear Regression the safer production choice.

4. **Production readiness requires more than accuracy**: Building the modular `src/` package, the singleton predictor with input validation, the Flask REST API, and the 40-case test suite each required as much engineering effort as the modelling itself.

---

## 8. Future Enhancements

- [ ] Add Gradient Boosting (XGBoost / LightGBM) to the model registry
- [ ] Implement a feature importance / SHAP explanation panel in the UI
- [ ] Add a data drift monitor (compare incoming prediction inputs to training distribution)
- [ ] Containerise with Docker; add a `docker-compose.yml`
- [ ] Set up CI/CD with GitHub Actions (lint → test → build)
- [ ] Periodically retrain on updated data using DVC or MLflow





**Project Title:** Production-Grade Bangalore House Price Prediction System

**Domain:** Real Estate / Supervised Machine Learning

**Problem Statement:** Predict the market price of residential properties in Bangalore, India from structured listing attributes (location, area, BHK, bathrooms) to provide transparent pricing benchmarks in an information-asymmetric market.

**Dataset Name:** Bengaluru House Price Dataset (Kaggle)

**Dataset Size:** 13,320 records × 9 features; ~500 KB CSV

**Algorithms Used:**
- Linear Regression (selected)
- Lasso Regression (L1 regularisation)
- Ridge Regression (L2 regularisation)
- Decision Tree Regressor
- Random Forest Regressor

**Evaluation Method:** 5-fold ShuffleSplit GridSearchCV; held-out test set (20%); residual analysis

**Key Results:**
- R² ≈ 0.848 on held-out test set
- MAE ≈ ₹18.5 Lakhs
- CV R² = 0.847 ± 0.006 (stable generalisation)
- Multi-stage outlier removal improved R² by ~4 percentage points

**Technologies Used:** Python 3.10, scikit-learn 1.3, pandas 2.0, Flask 3.0, pytest, black, isort

**Learning Outcomes:**
1. Designing layered, modular ML codebases for maintainability
2. Domain-specific feature engineering (location grouping, per-location outlier removal)
3. Rigorous model evaluation using cross-validation and residual analysis
4. Serving ML models as REST APIs with robust input validation and error handling
5. Writing production-quality code: type hints, docstrings, logging, configuration management, unit tests
