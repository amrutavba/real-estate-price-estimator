from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW, PROCESSED, MODELS, REPORTS = ROOT / "data/raw", ROOT / "data/processed", ROOT / "models", ROOT / "reports"
SEED, N_CLUSTERS = 42, 6
TARGET = "price"
CITIES = {"SF": (37.77, -122.42), "LA": (34.05, -118.24), "SD": (32.72, -117.16), "SJ": (37.34, -121.89), "SAC": (38.58, -121.49)}
GEO = ["Latitude", "Longitude"] + [f"dist_{c}" for c in CITIES] + ["dist_nearest"]     # 8 geographic metrics
NUM = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup", "rooms_per_person", "bedroom_ratio"]
CAT = ["age_band", "size_band"]
