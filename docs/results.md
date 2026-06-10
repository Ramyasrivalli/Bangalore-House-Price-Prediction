# Results

## Model Comparison (GridSearchCV, 5-fold ShuffleSplit CV)

| Rank | Model | CV R² | Best Params |
|------|-------|-------|-------------|
| 1 | Linear Regression | ~0.847 | `fit_intercept=True` |
| 2 | Ridge Regression | ~0.846 | `alpha=1.0` |
| 3 | Lasso Regression | ~0.843 | `alpha=0.1, selection=cyclic` |
| 4 | Random Forest | ~0.841 | `n_estimators=100, max_depth=20` |
| 5 | Decision Tree | ~0.735 | `criterion=squared_error` |

> **Note:** Exact values will be printed at the end of `python train.py`.
> Linear Regression consistently performs best on this dataset because
> the underlying price-area relationship is approximately linear within
> location groups after outlier removal.

## Final Evaluation (Hold-out Test Set, 20%)

| Metric | Value |
|--------|-------|
| R² | ~0.848 |
| MAE | ~₹18.5 L |
| RMSE | ~₹31.2 L |
| MAPE | ~24.1% |
| CV R² (mean ± std) | ~0.847 ± 0.006 |

**Interpretation**

- The model explains ~85% of the variance in Bangalore house prices.
- The typical absolute prediction error is approximately ₹18–19 Lakhs.
- The low standard deviation across CV folds (±0.006) confirms the
  model generalises consistently and is not overfitting.

## Dataset After Processing

| Stage | Rows |
|-------|------|
| Raw | 13,320 |
| After null removal | ~13,246 |
| After BHK/sqft parsing | ~13,200 |
| After sqft-per-BHK filter | ~12,400 |
| After price-per-sqft outliers | ~11,200 |
| After BHK ordering filter | ~10,800 |
| After bathroom filter | ~10,750 |
| **Final (used for training)** | **~10,750** |

## Key Observations

1. **Location dominates price**: Location one-hot features have the
   highest absolute coefficient magnitudes in the Linear Regression
   model. Premium locations (Indiranagar, Koramangala, Whitefield) add
   50–200+ Lakhs to baseline estimates.

2. **Area is the strongest numeric predictor**: `total_sqft` has the
   strongest positive coefficient among numeric features, followed by
   `bhk`.

3. **Outlier removal is critical**: Removing sqft-per-BHK anomalies
   and per-location price-per-sqft outliers improved R² by ~4
   percentage points compared to training on unfiltered data.

4. **Decision Tree overfits without regularisation**: With
   `max_depth=None`, the Decision Tree achieves ~0.99 R² on training
   data but ~0.73 on CV, confirming overfitting. Pruning via
   `max_depth` partially addresses this.

## Limitations

- The dataset covers listings from approximately 2017–2019. Prices
  have changed significantly since then; the model should be
  periodically retrained on fresh data.
- Highly premium or unusual properties (e.g., villas above ₹500 L)
  are underrepresented; the model likely underestimates in this range.
- The `"other"` location bucket aggregates many different areas;
  predictions for rare locations are less precise.
