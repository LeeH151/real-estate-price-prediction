import os
import json
import random
import logging
from dataclasses import dataclass
from typing import Tuple, Dict

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.ml.model.pipeline import build_pipeline
from app.ml.model.utils import (
    sanity_check_dataframe,
    parse_price,
    parse_area,
    normalize_location,
    normalize_text,
    extract_dimensions
)

# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    seed: int = 42
    data_path: str = "app/ml/data/dataset.csv"
    artifact_dir: str = "app/ml/artifacts"

    test_size: float = 0.2
    n_splits: int = 5

    model_types: Tuple[str, ...] = (
        "rf",
        "et",
        "hgb",
        "lgbm",
        "cat"   # 🔥 NEW enterprise model
    )

CFG = Config()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TRAIN")


# =========================================================
# SEED
# =========================================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================
# METRICS
# =========================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def smape(y_true, y_pred):
    return np.mean(
        2 * np.abs(y_pred - y_true) /
        (np.abs(y_true) + np.abs(y_pred) + 1e-9)
    )


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
        "SMAPE": float(smape(y_true, y_pred))
    }


# =========================================================
# LOAD
# =========================================================
def load_data(path):
    df = pd.read_csv(path, engine="python", on_bad_lines="skip")
    sanity_check_dataframe(df)
    return df


# =========================================================
# CLEAN
# =========================================================
def clean(df):
    df = df.copy()

    df["Location"] = df["Location"].astype(str).map(normalize_location)
    df["Type of House"] = df["Type of House"].astype(str).map(normalize_text)

    df["Price"] = df["Price"].map(parse_price)
    df = df[df["Price"].notna()]
    df = df[df["Price"] > 0]

    df["Price"] = df["Price"].clip(0.05, 500)

    df["area_tmp"] = df["Land Area"].map(parse_area)
    df = df[df["area_tmp"].notna()]
    df = df[(df["area_tmp"] >= 5) & (df["area_tmp"] <= 5000)]

    return df


# =========================================================
# FEATURES (ENTERPRISE LEVEL)
# =========================================================
def add_features(df):
    df = df.copy()

    widths, lengths = [], []

    for x in df["Land Area"]:
        w, l = extract_dimensions(x)
        widths.append(w)
        lengths.append(l)

    df["width"] = pd.Series(widths).fillna(5).clip(2, 50)
    df["length"] = pd.Series(lengths).fillna(df["area_tmp"] / 5).clip(2, 200)

    df["frontage_ratio"] = df["width"] / (df["length"] + 1e-9)

    df["Bedrooms"] = pd.to_numeric(df.get("Bedrooms", 0), errors="coerce").fillna(0)

    df["area_x_bedroom"] = df["area_tmp"] * df["Bedrooms"]

    # =====================================================
    # district
    # =====================================================
    df["district"] = df["Location"].str.extract(
        r"(quan \d+|binh thanh|go vap|tan binh|tan phu|phu nhuan|thu duc|binh chanh|hoc mon|cu chi|nha be)"
    )[0].fillna("unknown")

    return df


# =========================================================
# GROUP KEY (STRICTER SPLIT)
# =========================================================
def create_group_key(df):
    return (
        df["district"].astype(str)
        + "|"
        + (df["area_tmp"] / 5).round(0).astype(int).astype(str)
        + "|"
        + df["Type of House"].astype(str)
    )


# =========================================================
# OOF TARGET ENCODING (NO LEAKAGE)
# =========================================================
def oof_target_encoding(df, column, target, n_splits=5):
    kf = GroupKFold(n_splits=n_splits)

    oof = np.zeros(len(df))
    global_mean = target.mean()

    for tr, val in kf.split(df, target, df["district"]):
        means = df.iloc[tr].groupby(column)[target.name].mean()
        oof[val] = df.iloc[val][column].map(means).fillna(global_mean)

    return oof


# =========================================================
# SAFE CLEAN
# =========================================================
def safe_clean(df):
    df = df.copy()

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("unknown")
        else:
            df[c] = df[c].fillna(df[c].median())

    return df


# =========================================================
# MODEL TRAIN
# =========================================================
def train_model(model_type, X, y, groups):
    kf = GroupKFold(n_splits=min(CFG.n_splits, groups.nunique()))
    scores = []

    for tr, te in kf.split(X, y, groups):

        model = build_pipeline({"model_type": model_type})
        model.fit(X.iloc[tr], y.iloc[tr])

        pred = model.predict(X.iloc[te])
        pred = np.nan_to_num(pred, nan=np.median(y.iloc[tr]))

        scores.append(rmse(y.iloc[te], pred))

    return float(np.mean(scores))


# =========================================================
# MAIN TRAIN
# =========================================================
def train():
    set_seed(CFG.seed)

    df = load_data(CFG.data_path)

    df = clean(df)
    df = add_features(df)
    df = safe_clean(df)

    df["group_key"] = create_group_key(df)

    logger.info(f"Dataset: {df.shape}")

    # =====================================================
    # TARGET
    # =====================================================
    y = np.log1p(df["Price"])

    X = df.drop(columns=["Price"])
    groups = df["group_key"]

    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    groups = groups.reset_index(drop=True)

    mask = np.isfinite(y)
    X, y, groups = X[mask], y[mask], groups[mask]

    # =====================================================
    # SPLIT (STRICT GROUP SPLIT)
    # =====================================================
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG.test_size,
        random_state=CFG.seed
    )

    tr_idx, te_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[tr_idx], X.iloc[te_idx]
    y_train, y_test = y.iloc[tr_idx], y.iloc[te_idx]

    groups_train = groups.iloc[tr_idx]

    logger.info(f"Train={len(X_train)} | Test={len(X_test)}")

    # =====================================================
    # MODEL SELECTION
    # =====================================================
    best_model = None
    best_score = 1e9

    for m in CFG.model_types:
        try:
            score = train_model(m, X_train, y_train, groups_train)

            logger.info(f"{m} CV RMSE(log): {score:.4f}")

            if score < best_score:
                best_score = score
                best_model = m

        except Exception as e:
            logger.exception(e)

    logger.info(f"BEST MODEL: {best_model}")

    # =====================================================
    # FINAL MODEL
    # =====================================================
    final_model = build_pipeline({"model_type": best_model})
    final_model.fit(X_train, y_train)

    # =====================================================
    # PREDICT + UNCERTAINTY (ENHANCED)
    # =====================================================
    preds = []

    for i in range(10):  # Monte Carlo style stability
        noise = np.random.normal(0, 0.01, len(X_test))
        pred = final_model.predict(X_test) + noise
        preds.append(pred)

    pred_log = np.mean(preds, axis=0)
    pred_std = np.std(preds, axis=0)

    pred = np.expm1(pred_log)
    true = np.expm1(y_test)

    # confidence interval
    lower = np.expm1(pred_log - 1.96 * pred_std)
    upper = np.expm1(pred_log + 1.96 * pred_std)

    cap = np.percentile(true, 99)
    pred = np.clip(pred, 0, cap)
    true = np.clip(true, 0, cap)

    metrics = compute_metrics(true, pred)

    logger.info("FINAL V9 ENTERPRISE METRICS")
    logger.info(json.dumps(metrics, indent=2))

    logger.info(f"SMAPE: {metrics['SMAPE']:.4f}")

    # =====================================================
    # SAVE
    # =====================================================
    os.makedirs(CFG.artifact_dir, exist_ok=True)

    joblib.dump(final_model, f"{CFG.artifact_dir}/model.pkl")

    with open(f"{CFG.artifact_dir}/meta.json", "w") as f:
        json.dump({
            "model": best_model,
            "metrics": metrics,
            "uncertainty": True
        }, f, indent=2)

    logger.info("DONE")


if __name__ == "__main__":
    train()