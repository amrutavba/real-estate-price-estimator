import numpy as np
import pandas as pd
from .config import *


def haversine_km(lat1, lon1, lat2, lon2):
    """Great-circle distance in km."""
    p1, p2, dphi, dl = map(np.radians, (lat1, lat2, lat2 - lat1, lon2 - lon1))
    a = np.sin(dphi / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 6371 * 2 * np.arcsin(np.sqrt(a))


def add_features(df):
    """Ratios, categorical bands (fixed bins -> no leakage) and 8 geographic metrics (lat, lon, distance to 5 metros, nearest)."""
    d = df.copy()
    d["rooms_per_person"] = d.AveRooms / d.AveOccup
    d["bedroom_ratio"] = d.AveBedrms / d.AveRooms
    d["age_band"] = pd.cut(d.HouseAge, [0, 15, 30, 45, 200], labels=["new", "mid", "old", "very_old"]).astype(str)
    d["size_band"] = pd.cut(d.AveRooms, [0, 4, 6, 8, 100], labels=["small", "medium", "large", "xl"]).astype(str)
    for c, (la, lo) in CITIES.items():
        d[f"dist_{c}"] = haversine_km(d.Latitude, d.Longitude, la, lo)
    d["dist_nearest"] = d[[f"dist_{c}" for c in CITIES]].min(axis=1)
    return d
