# Real Estate Price Estimator

Regression-based property valuation with K-Means geographic sub-market segmentation and an interactive Streamlit price estimator.

## Problem & objective
Estimate property values from physical and location attributes, quantify prediction error, and show how prices differ across sub-markets.

## Dataset
**California Housing** (1990 US Census; Pace & Barry 1997; loaded with `sklearn.datasets.fetch_california_housing`, 20,640 block groups, 8 features + median house value). Public, no licence restrictions for learning use.
**Limitations:** rows are census *block groups*, not individual listings; prices are 1990 values, capped at $500,001 (capped rows are removed); no price-per-sq-ft or bathroom columns. `--demo` uses a synthetic stand-in for offline testing only.

## Methodology
1. **Cleaning:** duplicates, impossible values, extreme IQR outliers (3×IQR), censored prices removed; median imputation inside the pipeline.
2. **Features:** rooms per person, bedroom ratio, property-age and size bands, and **8 geographic metrics** (latitude, longitude, haversine distance to SF/LA/San Diego/San Jose/Sacramento, nearest-metro distance).
3. **Sub-markets:** K-Means (k=6, silhouette reported) on scaled geographic metrics, fitted inside the pipeline on training data only (no leakage); the cluster is then used as a model feature.
4. **Models:** Linear Regression, Decision Tree, Random Forest, Gradient Boosting; 80/20 split, 3-fold CV; metrics **R², MAE, RMSE**; each model is compared with and without the sub-market feature. Best model is chosen on CV R², not test score.
5. **App:** estimate + typical error + what-if sensitivity (+10% on each input) + feature importance.

## Structure
`src/{config,data,features,modelling,run}.py` · `dashboard/app.py` · `tests/test_housing.py`

## Run
```bash
pip install -r requirements.txt
python -m src.run          # add --demo offline
pytest -q
streamlit run dashboard/app.py
```

## Results (fill from `reports/model_comparison.csv` and `metrics.json`)
| Model | CV R² | Test R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | | | | |
| Decision Tree | | | | |
| Random Forest | | | | |
| Gradient Boosting | | | | |

MAE change from the K-Means sub-market feature: (see `metrics.json`; it may be small for tree models that already use latitude/longitude).

## Limitations
Block-group medians (not listings), 1990 data, census-based features, no interaction with macro conditions, spatial autocorrelation ignored in CV (random split).

## Future improvements
Spatial cross-validation, an Indian listing dataset with price/sq ft, hyper-parameter search, SHAP.

**Repo name:** `real-estate-price-estimator` · **Topics:** `real-estate` `regression` `random-forest` `gradient-boosting` `kmeans` `scikit-learn` `streamlit`
