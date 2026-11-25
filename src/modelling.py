import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from .config import *


class GeoCluster(BaseEstimator, TransformerMixin):
    """K-Means on the scaled geographic metrics -> one-hot sub-market label. Fitted on TRAIN data only (inside the pipeline)."""
    def __init__(self, k=N_CLUSTERS): self.k = k
    def fit(self, X, y=None):
        self.sc_ = StandardScaler().fit(X)
        self.km_ = KMeans(self.k, n_init=10, random_state=SEED).fit(self.sc_.transform(X)); return self
    def labels(self, X): return self.km_.predict(self.sc_.transform(X))
    def transform(self, X): return np.eye(self.k)[self.labels(X)]
    def get_feature_names_out(self, names=None): return np.array([f"submarket_{i}" for i in range(self.k)])


def make_models():
    return {"Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(max_depth=10, min_samples_leaf=10, random_state=SEED),
            "Random Forest": RandomForestRegressor(150, min_samples_leaf=3, n_jobs=-1, random_state=SEED),
            "Gradient Boosting": HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31, random_state=SEED)}


def build(model, clusters=True):
    """Preprocessing: median imputation + scaling (numeric), one-hot (bands), scaled raw geo metrics, optional K-Means sub-market."""
    parts = [("num", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), NUM),
             ("cat", OneHotEncoder(handle_unknown="ignore"), CAT),
             ("geo", make_pipeline(SimpleImputer(strategy="median"), StandardScaler()), GEO)]
    if clusters:
        parts.append(("cluster", GeoCluster(), GEO))
    return Pipeline([("prep", ColumnTransformer(parts)), ("model", model)])


def evaluate(pipe, Xte, yte):
    p = pipe.predict(Xte)
    return {"R2": r2_score(yte, p), "MAE": mean_absolute_error(yte, p), "RMSE": mean_squared_error(yte, p) ** 0.5}


def compare(Xtr, ytr, Xte, yte, cv=3):
    rows = []
    for use_cl in (False, True):
        for name, m in make_models().items():
            pipe = build(m, use_cl).fit(Xtr, ytr)
            cvr2 = cross_val_score(build(make_models()[name], use_cl), Xtr, ytr, cv=cv, scoring="r2", n_jobs=1).mean()
            rows.append({"model": name, "submarket_feature": use_cl, "CV_R2": cvr2, **evaluate(pipe, Xte, yte)})
    return pd.DataFrame(rows)


def feature_importance(pipe):
    names = pipe["prep"].get_feature_names_out()
    return pd.Series(pipe["model"].feature_importances_, index=[n.split("__")[-1] for n in names]).sort_values(ascending=False)
