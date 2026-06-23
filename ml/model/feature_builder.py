import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from app.ml.model.utils import parse_area, parse_room

EPS = 1e-8


class FeatureBuilder(BaseEstimator, TransformerMixin):

    def __init__(self, debug: bool = False):
        self.debug = debug

        # =========================
        # GLOBAL STATS
        # =========================
        self.area_median_ = 80.0
        self.global_price_mean_ = 0.0

        # smoothed target encoding
        self.location_stats_ = {}

        # smoothing strength (VERY IMPORTANT)
        self.smoothing_k_ = 10.0

        # =========================
        # FEATURE COLUMNS (STABLE SET)
        # =========================
        self.feature_columns_ = [

            # RAW
            "area",
            "bedrooms",
            "toilets",
            "floors",
            "rooms",

            # LOG
            "log_area",
            "log_rooms",

            # RATIOS
            "area_per_room",
            "density",
            "rooms_per_area",

            # LOCATION (ROBUST ENCODING)
            "location_encoded",
            "location_strength",

            # INTERACTIONS (SAFE VERSION)
            "size_score",
            "structure_score",

            # FLAGS
            "is_large",
            "is_small",
            "is_villa",
            "is_apartment",
            "is_land",
            "is_legal_ok",
        ]

    # =====================================================
    def fit(self, X, y=None):

        df = X.copy()

        # =========================
        # AREA
        # =========================
        area = pd.to_numeric(df["Land Area"].map(parse_area), errors="coerce")
        self.area_median_ = float(area.median() if area.notna().any() else 80.0)

        # =========================
        # TARGET STATS
        # =========================
        if y is not None:
            tmp = df.copy()
            tmp["target"] = y.values

            self.global_price_mean_ = float(tmp["target"].mean())

            g = tmp.groupby("Location")["target"].agg(["mean", "count"])

            for loc, row in g.iterrows():
                self.location_stats_[str(loc)] = {
                    "mean": float(row["mean"]),
                    "count": int(row["count"])
                }

        return self

    # =====================================================
    def _encode_location(self, loc: str):

        stats = self.location_stats_.get(loc)

        # unseen category → global mean
        if stats is None:
            return self.global_price_mean_, 0.0

        # smoothed target encoding (ANTI OVERFIT)
        w = stats["count"]

        smooth = (
            stats["mean"] * w + self.global_price_mean_ * self.smoothing_k_
        ) / (w + self.smoothing_k_)

        return smooth, float(w)

    # =====================================================
    def transform(self, X):

        df = X.copy()

        # =========================
        # NUMERIC FEATURES
        # =========================
        df["area"] = pd.to_numeric(df["Land Area"].map(parse_area), errors="coerce")
        df["bedrooms"] = pd.to_numeric(df["Bedrooms"].map(parse_room), errors="coerce").fillna(0)
        df["toilets"] = pd.to_numeric(df["Toilets"].map(parse_room), errors="coerce").fillna(0)
        df["floors"] = pd.to_numeric(df["Total Floors"].map(parse_room), errors="coerce").fillna(0)

        df["area"] = df["area"].fillna(self.area_median_)

        # =========================
        # CORE ENGINEERING
        # =========================
        df["rooms"] = df["bedrooms"] + df["toilets"]

        df["log_area"] = np.log1p(df["area"])
        df["log_rooms"] = np.log1p(df["rooms"])

        # ratios (stable)
        df["area_per_room"] = df["area"] / (df["rooms"] + 1)
        df["density"] = df["rooms"] / (df["area"] + 1)
        df["rooms_per_area"] = df["rooms"] / (df["area"] + EPS)

        # =========================
        # LOCATION ENCODING
        # =========================
        locs = df["Location"].fillna("unknown").astype(str)

        enc = []
        strength = []

        for v in locs:
            e, s = self._encode_location(v)
            enc.append(e)
            strength.append(s)

        df["location_encoded"] = enc
        df["location_strength"] = strength

        # =========================
        # INTERACTIONS (SAFE VERSION)
        # =========================
        df["size_score"] = df["log_area"] * np.log1p(df["rooms"] + 1)
        df["structure_score"] = df["log_area"] * (df["floors"] + 1)

        # =========================
        # FLAGS
        # =========================
        df["is_large"] = (df["area"] > 120).astype(np.float32)
        df["is_small"] = (df["area"] < 30).astype(np.float32)

        house = df["Type of House"].fillna("").astype(str).str.lower()

        df["is_villa"] = house.str.contains("villa|biet thu", regex=True).astype(np.float32)
        df["is_apartment"] = house.str.contains("chung cu|can ho", regex=True).astype(np.float32)
        df["is_land"] = house.str.contains("dat|nha dat", regex=True).astype(np.float32)

        legal = df["Legal Documents"].fillna("").astype(str).str.lower()
        df["is_legal_ok"] = legal.str.contains("so hong|sổ hồng", regex=True).astype(np.float32)

        # =========================
        # FINAL ALIGNMENT
        # =========================
        for c in self.feature_columns_:
            if c not in df.columns:
                df[c] = 0.0

        out = df[self.feature_columns_]

        out = out.replace([np.inf, -np.inf], 0).fillna(0)

        if self.debug:
            print("FeatureBuilder PRO++ shape:", out.shape)

        return out.astype(np.float32)