import numpy as np, pandas as pd, pytest
from sklearn.model_selection import train_test_split
from src import data, features, modelling as M
from src.config import *


@pytest.fixture(scope="module")
def df():
    return features.add_features(data.clean(data.synthetic(3000)))


def test_haversine_sf_la():
    assert 540 < features.haversine_km(*CITIES["SF"], *CITIES["LA"]) < 580


def test_clean_removes_cap_and_dupes():
    d = data.synthetic(200); d.loc[0, TARGET] = 500001; d = pd.concat([d, d.iloc[[1]]])
    out = data.clean(d)
    assert out[TARGET].max() < 500000 and not out.duplicated().any()


def test_features_complete(df):
    assert len(GEO) == 8 and df[GEO + NUM + CAT].isna().sum().sum() == 0 and set(df.age_band) <= {"new", "mid", "old", "very_old"}


def test_pipeline_handles_missing_and_predicts(df):
    X, y = df.drop(columns=TARGET), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, random_state=0)
    Xte = Xte.copy(); Xte.iloc[0, Xte.columns.get_loc("MedInc")] = np.nan          # imputer must cope
    pipe = M.build(M.make_models()["Random Forest"]).fit(Xtr, ytr)
    assert np.isfinite(pipe.predict(Xte)).all() and M.evaluate(pipe, Xte.dropna(), yte[Xte.notna().all(axis=1)])["R2"] > 0.5


def test_cluster_transformer_one_hot(df):
    g = M.GeoCluster(4).fit(df[GEO]); out = g.transform(df[GEO])
    assert out.shape == (len(df), 4) and (out.sum(axis=1) == 1).all()
