import os
import json
import random
import logging
from dataclasses import dataclass
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from app.ml.model.pipeline import build_pipeline
from app.ml.model.utils import (
    sanity_check_dataframe,
    parse_price,
    parse_area,
    normalize_location,
    make_property_fingerprint
)

# =========================================================
# CONFIG
# =========================================================
@dataclass
class Config:
    seed: int = 42
    artifact_dir: str = "app/models"
    data_path: str = "data/dataset.csv"
    test_size: float = 0.2
    n_splits: int = 5

    model_types: Tuple[str, ...] = ("rf", "hgb", "et")

    price_clip_min: float = 0.01
    price_clip_max: float = 200.0

    min_area: float = 5
    max_area: float = 5000

CFG = Config()


# =========================================================
# LOGGING
# =========================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger("TRAIN")


# =========================================================
# SEED (FULL REPRODUCIBILITY)
# =========================================================
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# =========================================================
# METRICS
# =========================================================
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def compute_metrics(y_true, y_pred):
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = np.array(y_true)[mask]
    y_pred = np.array(y_pred)[mask]

    if len(y_true) == 0:
        return {"MAE": np.nan, "RMSE": np.nan, "R2": np.nan}

    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(rmse(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred))
    }


# =========================================================
# DATA LOADER (SAFE PATH + VALIDATION)
# =========================================================
def load_data(path: str):
    path = Path(path)

    base_dir = Path(__file__).resolve().parents[1]
    data_path = base_dir / "data" / path.name

    logger.info(f"📥 Loading: {data_path}")

    if not data_path.exists():
        raise FileNotFoundError(data_path)

    df = pd.read_csv(
        data_path,
        encoding="utf-8",
        engine="python",
        on_bad_lines="skip"
    )

    sanity_check_dataframe(df)
    logger.info(f"📊 Raw shape: {df.shape}")

    return df


# =========================================================
# CLEAN (ROBUST VERSION)
# =========================================================
def basic_clean(df: pd.DataFrame):
    df = df.copy()

    required = ["Location", "Price", "Land Area"]

    for c in required:
        if c not in df.columns:
            df[c] = np.nan

    # location normalization
    df["Location"] = (
        df["Location"]
        .fillna("")
        .astype(str)
        .map(normalize_location)
    )

    # price
    df["Price"] = df["Price"].map(parse_price)
    df = df[df["Price"].notna()]
    df = df[df["Price"] > 0]

    df["Price"] = df["Price"].clip(
        CFG.price_clip_min,
        CFG.price_clip_max
    )

    # area
    df["area_tmp"] = df["Land Area"].map(parse_area)
    df = df[df["area_tmp"].notna()]

    df = df[
        (df["area_tmp"] >= CFG.min_area) &
        (df["area_tmp"] <= CFG.max_area)
    ]

    return df


# =========================================================
# GROUP KEY (ANTI-LEAKAGE VERSION 2.0)
# =========================================================
def create_groups(df):
    """
    PRO FIX:
    - fingerprint stable + reduces same-property leakage
    - stronger than Location+Area
    """
    return make_property_fingerprint(df).astype(str)


# =========================================================
# CV SCORING (STABILITY UPGRADE)
# =========================================================
def cv_score(scores):
    """
    PRO UPGRADE:
    - use median instead of mean (robust to outlier fold spike)
    """
    return float(np.median(scores)), float(np.std(scores))


# =========================================================
# TRAIN ONE MODEL (ROBUST CV)
# =========================================================
def train_one_model(model_type, X, y, groups):

    kf = GroupKFold(n_splits=CFG.n_splits)

    scores = []

    for fold, (tr, te) in enumerate(kf.split(X, y, groups), 1):

        logger.info(f"📂 Fold {fold}")

        model = build_pipeline({"model_type": model_type})

        model.fit(X.iloc[tr], y.iloc[tr])

        pred = model.predict(X.iloc[te])

        # stability fixes
        pred = np.nan_to_num(pred, nan=np.mean(y.iloc[tr]))
        pred = np.clip(pred, 0, None)

        score = rmse(y.iloc[te], pred)
        scores.append(score)

        logger.info(f"   RMSE(log): {score:.4f}")

    median_score, std_score = cv_score(scores)

    logger.info(f"📊 CV median: {median_score:.4f} ± {std_score:.4f}")

    # stronger penalty (reduce unstable models)
    return median_score + 0.25 * std_score


# =========================================================
# TRAIN PIPELINE
# =========================================================
def train():

    logger.info("=" * 60)
    logger.info("🚀 TRAINING START (PRO++ STABLE VERSION)")
    logger.info("=" * 60)

    set_seed(CFG.seed)

    # LOAD
    raw = load_data(CFG.data_path)
    df = basic_clean(raw)

    # TARGET (LOG SCALE)
    y = np.log1p(df["Price"])

    X = df.drop(columns=["Price", "area_tmp"])

    # GROUPS
    groups = create_groups(df)

    # SPLIT (STRONGER CONTROL)
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CFG.test_size,
        random_state=CFG.seed
    )

    train_idx, test_idx = next(splitter.split(X, y, groups))

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    groups_train = groups.iloc[train_idx]

    # BASELINE
    baseline = np.mean(y_train)
    baseline_rmse = rmse(y_train, np.full_like(y_train, baseline))

    logger.info(f"📉 Baseline RMSE(log): {baseline_rmse:.4f}")

    # MODEL SEARCH
    best_model = None
    best_score = float("inf")

    for m in CFG.model_types:

        logger.info("\n" + "=" * 50)
        logger.info(f"🚀 Model: {m}")
        logger.info("=" * 50)

        score = train_one_model(m, X_train, y_train, groups_train)

        logger.info(f"📉 CV Score: {score:.4f}")

        if score < best_score:
            best_score = score
            best_model = m

    # FINAL MODEL
    logger.info(f"\n🏆 BEST MODEL: {best_model}")

    final_model = build_pipeline({"model_type": best_model})
    final_model.fit(X_train, y_train)

    # HOLDOUT
    pred_log = final_model.predict(X_test)

    pred_log = np.nan_to_num(pred_log, nan=np.mean(y_train))
    pred_log = np.clip(pred_log, 0, None)

    # LOG METRICS
    result_log = compute_metrics(y_test.values, pred_log)

    # REAL METRICS
    true_real = np.expm1(y_test.values)
    pred_real = np.expm1(pred_log)

    result_real = compute_metrics(true_real, pred_real)

    logger.info("\n📊 HOLDOUT (LOG SCALE)")
    logger.info(json.dumps(result_log, indent=4))

    logger.info("\n📊 HOLDOUT (REAL SCALE)")
    logger.info(json.dumps(result_real, indent=4))

    # SAVE ARTIFACTS
    os.makedirs(CFG.artifact_dir, exist_ok=True)

    joblib.dump(final_model, f"{CFG.artifact_dir}/model.pkl")

    with open(f"{CFG.artifact_dir}/metadata.json", "w") as f:
        json.dump({
            "best_model": best_model,
            "cv_score": best_score,
            "cv_median": cv_score([best_score])[0],
            "baseline_rmse": baseline_rmse,
            "holdout_log": result_log,
            "holdout_real": result_real,
            "seed": CFG.seed,
            "n_train": len(X_train),
            "n_test": len(X_test),
            "n_total": len(df)
        }, f, indent=4)

    logger.info("🎉 TRAINING DONE")


# =========================================================
if __name__ == "__main__":
    train()