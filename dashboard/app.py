"""streamlit run dashboard/app.py  (run `python -m src.run` first)"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib, pandas as pd, plotly.express as px, streamlit as st
from src import features
from src.config import *

st.set_page_config("Real Estate Price Estimator", layout="wide")
try:
    art = joblib.load(MODELS / "price_model.joblib"); df = pd.read_csv(PROCESSED / "housing_clean.csv")
    comp, seg, imp = (pd.read_csv(REPORTS / f) for f in ("model_comparison.csv", "submarkets.csv", "feature_importance.csv"))
    metrics = json.loads((REPORTS / "metrics.json").read_text())
except FileNotFoundError:
    st.error("Run `python -m src.run` (or `--demo`) first."); st.stop()

st.title("Real Estate Price Estimator")
st.caption("California Housing (1990 census block groups): prices are block-group MEDIANS, not individual listings.")
t1, t2, t3, t4 = st.tabs(["Estimate price", "Market segments", "Model comparison", "Explore data"])

with t1:
    c = st.columns(3)
    inc = c[0].slider("Median income (x$10k)", 0.5, 15.0, 4.0, 0.1); age = c[1].slider("Property age (years)", 1, 52, 25)
    rooms = c[2].slider("Avg rooms", 2.0, 10.0, 5.5, 0.1)
    c = st.columns(3)
    bed = c[0].slider("Avg bedrooms", 0.5, 3.0, 1.1, 0.1); occ = c[1].slider("Avg occupants", 1.0, 6.0, 3.0, 0.1)
    pop = c[2].slider("Block population", 300, 5000, 1400, 100)
    region = st.selectbox("Region", ["Custom"] + list(CITIES))
    lat, lon = (CITIES[region] if region != "Custom" else (st.slider("Latitude", 32.5, 42.0, 36.0), st.slider("Longitude", -124.4, -114.3, -119.5)))

    def predict(**kw):
        row = {"MedInc": inc, "HouseAge": age, "AveRooms": rooms, "AveBedrms": bed, "AveOccup": occ, "Population": pop, "Latitude": lat, "Longitude": lon, **kw}
        return float(art["model"].predict(features.add_features(pd.DataFrame([row])))[0])

    price = predict(); mae = metrics["MAE"]
    st.metric("Estimated median price", f"${price:,.0f}", help=f"Typical error about ±${mae:,.0f} (test MAE)")
    sens = {k: (predict(**{k: v * 1.1}) - price) for k, v in {"MedInc": inc, "HouseAge": age, "AveRooms": rooms, "AveOccup": occ, "Population": pop}.items()}
    st.plotly_chart(px.bar(pd.Series(sens), orientation="h", title="What-if: price change if each input rises 10%", labels={"value": "$ change", "index": ""}), use_container_width=True)
    st.plotly_chart(px.bar(imp.head(10).set_index(imp.columns[0])[imp.columns[1]], orientation="h", title="Global feature importance (Random Forest)"), use_container_width=True)

with t2:
    st.plotly_chart(px.scatter(df.sample(min(8000, len(df)), random_state=SEED), x="Longitude", y="Latitude", color=df.submarket.astype(str).sample(min(8000, len(df)), random_state=SEED),
                               title="K-Means sub-markets (8 geographic metrics)", height=600), use_container_width=True)
    st.dataframe(seg.round(1), use_container_width=True)

with t3:
    st.dataframe(comp.round(3), use_container_width=True)
    st.write(f"Selected model: **{metrics['best']}** — R² {metrics['R2']:.3f}, MAE ${metrics['MAE']:,.0f}, RMSE ${metrics['RMSE']:,.0f}. "
             f"MAE change from adding the K-Means sub-market feature: {metrics['mae_reduction_from_submarket']:.1%}.")

with t4:
    st.plotly_chart(px.histogram(df, x="price", nbins=60, title="Price distribution"), use_container_width=True)
    st.plotly_chart(px.scatter(df.sample(5000, random_state=SEED), x="MedInc", y="price", trendline="ols", title="Income vs price"), use_container_width=True)
