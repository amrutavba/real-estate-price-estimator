"""California Housing (1990 US Census, block-group level; StatLib / Pace & Barry 1997, via scikit-learn)."""
import numpy as np
import pandas as pd
from .config import *


def load():
    """Fetch via scikit-learn (cached after first download) and cache as CSV in data/raw."""
    path = RAW / "housing.csv"
    if not path.exists():
        try:
            from sklearn.datasets import fetch_california_housing
            df = fetch_california_housing(as_frame=True).frame.rename(columns={"MedHouseVal": TARGET})
            df[TARGET] *= 100_000                              # dataset unit is $100k
            RAW.mkdir(parents=True, exist_ok=True); df.to_csv(path, index=False)
        except Exception as e:
            raise RuntimeError(f"Could not download California Housing ({e}). Place a CSV with the same columns at {path}.") from e
    return pd.read_csv(path)


def clean(df):
    """Drop duplicates, impossible values, IQR outliers on occupancy/room ratios and the censored $500k cap."""
    df = df.drop_duplicates().copy()
    df = df[(df[TARGET] > 0) & (df.AveRooms > 0) & (df.AveBedrms > 0) & (df.AveOccup > 0)]
    df = df[df[TARGET] < 500_000]                              # prices are censored at $500,001 -> biased, removed
    for c in ["AveRooms", "AveBedrms", "AveOccup", "Population"]:
        q1, q3 = df[c].quantile([.25, .75]); iqr = q3 - q1
        df = df[df[c].between(q1 - 3 * iqr, q3 + 3 * iqr)]     # 3*IQR = remove only extreme outliers
    return df.reset_index(drop=True)


def synthetic(n=20000, seed=SEED):
    """Synthetic stand-in with identical columns (DEMO/TESTS ONLY)."""
    rng = np.random.default_rng(seed)
    lat, lon = rng.uniform(32.6, 38.8, n), rng.uniform(-123, -117, n)
    inc = rng.lognormal(1.2, .45, n).clip(.5, 15); age = rng.integers(2, 52, n).astype(float)
    rooms = rng.normal(5.4, 1.2, n).clip(2, 12); occ = rng.normal(3, .6, n).clip(1.2, 8)
    coast = np.exp(-((lon + 121.5) ** 2) / 3)
    price = 40000 + 38000 * inc + 60000 * coast - 600 * age + 8000 * rooms + rng.normal(0, 25000, n)
    return pd.DataFrame({"MedInc": inc, "HouseAge": age, "AveRooms": rooms, "AveBedrms": rooms * rng.uniform(.18, .3, n),
                         "Population": rng.integers(300, 3500, n), "AveOccup": occ, "Latitude": lat, "Longitude": lon,
                         TARGET: price.clip(30000, 490000)})
