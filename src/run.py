"""python -m src.run [--demo]"""
import argparse, json
import joblib, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from . import data, features, modelling as M
from .config import *


def main(demo=False):
    df = features.add_features(data.clean(data.synthetic() if demo else data.load()))
    for p in (PROCESSED, MODELS, REPORTS): p.mkdir(exist_ok=True, parents=True)
    X, y = df.drop(columns=TARGET), df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"rows={len(df)}")
    res = M.compare(Xtr, ytr, Xte, yte)
    print(res.round(3).to_string(index=False))
    res.to_csv(REPORTS / "model_comparison.csv", index=False)

    best_name = res[res.submarket_feature].sort_values("CV_R2").iloc[-1]["model"]     # chosen on CV (not test) score
    best = M.build(M.make_models()[best_name], True).fit(Xtr, ytr)
    wo = res[(res.model == best_name) & ~res.submarket_feature].iloc[0]; w = res[(res.model == best_name) & res.submarket_feature].iloc[0]
    print(f"\nbest={best_name} R2={w.R2:.3f} MAE=${w.MAE:,.0f}; MAE change from sub-market feature: {(1 - w.MAE / wo.MAE):.1%}")

    gc = best["prep"].named_transformers_["cluster"]
    df["submarket"] = gc.labels(df[GEO])
    sil = silhouette_score(gc.sc_.transform(df[GEO].sample(5000, random_state=SEED)), gc.labels(df[GEO].sample(5000, random_state=SEED)))
    seg = df.groupby("submarket").agg(blocks=(TARGET, "size"), avg_price=(TARGET, "mean"), med_income=("MedInc", "median"),
                                       avg_rooms=("AveRooms", "mean"), avg_age=("HouseAge", "mean"), lat=("Latitude", "mean"), lon=("Longitude", "mean"))
    seg["price_to_income"] = seg.avg_price / (seg.med_income * 10_000)
    print(f"\nK-Means silhouette={sil:.2f}\n{seg.round(1).to_string()}")
    seg.to_csv(REPORTS / "submarkets.csv"); df.to_csv(PROCESSED / "housing_clean.csv", index=False)
    imp = M.feature_importance(M.build(M.make_models()["Random Forest"], True).fit(Xtr, ytr)); imp.to_csv(REPORTS / "feature_importance.csv")
    joblib.dump({"model": best, "name": best_name, "defaults": X.median(numeric_only=True).to_dict()}, MODELS / "price_model.joblib")
    (REPORTS / "metrics.json").write_text(json.dumps({"best": best_name, "R2": w.R2, "MAE": w.MAE, "RMSE": w.RMSE,
        "mae_reduction_from_submarket": 1 - w.MAE / wo.MAE, "silhouette": sil}, indent=2, default=float))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--demo", action="store_true"); main(ap.parse_args().demo)
