# Methodology

## Problem Formulation

This is a **supervised regression** task. Given property characteristics
(location, area, BHK, bathrooms), we predict the listing price in Indian Rupees (Lakhs).

**Formal definition**

```
f : (location, total_sqft, bath, bhk) → price (₹ Lakhs)
```

## Dataset

| Property | Value |
|----------|-------|
| Source | Kaggle – Bengaluru House Price Dataset |
| Raw shape | 13,320 rows × 9 columns |
| Target | `price` (₹ Lakhs, float) |
| Target range | ₹8 L – ₹3,600 L |
| Locations (raw) | ~1,300 distinct |
| Locations (after grouping) | ~242 distinct |

## Preprocessing Pipeline

### Step 1 – Column Selection

Four columns are dropped as uninformative for price modelling:

| Dropped Column | Reason |
|---------------|--------|
| `area_type` | Redundant with `total_sqft` |
| `society` | Too granular; over 4,000 unique values |
| `balcony` | Weak signal; large missingness |
| `availability` | Future possession date; not predictive of current price |

### Step 2 – Null Removal

Rows with any null value are dropped (< 1% of data).

### Step 3 – BHK Extraction

The `size` column is free-text (e.g. "2 BHK", "4 Bedroom"). An integer
`bhk` column is extracted by splitting on whitespace and taking the
first token.

### Step 4 – sqft Normalisation

`total_sqft` contains three formats:
- **Plain numeric** (`"1200"`) → direct cast
- **Range** (`"1200-1800"`) → arithmetic mean (1500.0)
- **Non-numeric** (`"34.46Sq. Meter"`) → NaN, row dropped

## Feature Engineering

### price_per_sqft

A diagnostic feature computed as `price × 100,000 / total_sqft`. Used
only for outlier removal; excluded from the final feature matrix.

### Location Grouping

Locations appearing ≤ 10 times in the dataset are collapsed to
`"other"`. This prevents the model from overfitting to sparse location
one-hot vectors and reduces feature dimensionality.

### Outlier Removal (3 stages)

**Stage 1 – sqft per BHK**

Any listing with `total_sqft / bhk < 300` is removed. A studio flat
cannot reasonably be smaller than 300 sq ft per bedroom.

**Stage 2 – Price per sqft (per-location)**

For each location, listings outside one standard deviation of the
location's mean price-per-sqft are removed. This is a location-aware
approach that preserves natural price variation between markets.

**Stage 3 – BHK price ordering**

If the average price-per-sqft of 2 BHK flats in a location is higher
than that of 3 BHK flats in the same location, the cheaper 3 BHK
listings are removed. This removes data-entry errors where a larger
unit was priced lower than a smaller one.

**Stage 4 – Bathroom anomalies**

Listings with `bath ≥ bhk + 2` are removed as likely data errors.

### One-Hot Encoding

The `location` column is one-hot encoded. The `"other"` dummy is
dropped (reference category) to avoid the dummy variable trap.

## Model Selection

Five algorithms are evaluated via 5-fold **ShuffleSplit** cross-
validation with an 80/20 split per fold:

| Model | Hyperparameter Space |
|-------|---------------------|
| Linear Regression | `fit_intercept`: {True, False} |
| Lasso Regression | `alpha`: {0.1, 0.5, 1, 2, 5}; `selection`: {random, cyclic} |
| Ridge Regression | `alpha`: {0.1, 1, 5, 10, 50}; `fit_intercept`: {True, False} |
| Decision Tree | `criterion`: {squared_error, absolute_error}; `max_depth`: {None, 5, 10, 20} |
| Random Forest | `n_estimators`: {50, 100}; `max_depth`: {None, 10, 20} |

Grid search optimises **R²** on the validation folds. The model with
the highest mean CV score is selected and retrained on the full
training set.

## Evaluation Strategy

Three complementary evaluation approaches are used:

1. **Hold-out set (20%)** – unbiased final metrics
2. **5-fold ShuffleSplit CV** – estimate of generalisation variance
3. **Residual analysis** – detect systematic bias or heteroscedasticity

**Metrics reported**:

| Metric | Description |
|--------|-------------|
| R² | Proportion of variance explained |
| MAE | Mean Absolute Error (Lakhs ₹) |
| RMSE | Root Mean Squared Error (Lakhs ₹) |
| MAPE | Mean Absolute Percentage Error |

## Reproducibility

- All random operations use `RANDOM_STATE = 42`
- Data splits are deterministic via `train_test_split(random_state=42)`
- CV splits use `ShuffleSplit(random_state=42)`
- Model artifacts are versioned by commit hash in production deployments
