import joblib
import numpy as np
import pandas as pd

from functools import lru_cache
from pathlib import Path
import sys
import types
import app.ml.model.feature_builder as feature_builder
import app.ml.model.pipeline as pipeline
import app.ml.model.utils as utils
# =========================================================
# FIX PICKLE MODULE "model"
# =========================================================
def register_model_module():

    # nếu đã tồn tại thì bỏ qua
    if "model" in sys.modules:
        return

    # tạo root module
    model = types.ModuleType("model")
    model.__path__ = []   # 👈 QUAN TRỌNG: biến thành package

    # tạo submodules
    model.feature_builder = feature_builder
    model.pipeline = pipeline
    model.utils = utils

    # đăng ký vào sys.modules (QUAN TRỌNG)
    sys.modules["model"] = model
    sys.modules["model.feature_builder"] = feature_builder
    sys.modules["model.pipeline"] = pipeline
    sys.modules["model.utils"] = utils


# =========================================================
# PATH
# =========================================================
BASE_DIR = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    BASE_DIR
    / "ml"
    / "artifacts"
    / "model.pkl"
)


# =========================================================
# LOAD MODEL
# =========================================================
@lru_cache()
def get_model():

    # 🔥 MUST REGISTER BEFORE LOAD
    register_model_module()

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"❌ Model not found: {MODEL_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    print("✅ MODEL LOADED:", type(model))

    return model


# =========================================================
# PREPROCESS
# =========================================================
def preprocess(payload):

    house_map = {
        "townhouse": "nha hem",
        "apartment": "can ho chung cu",
        "villa": "biet thu",
    }

    house_type = house_map.get(payload.house_type, "nha hem")

    data = {
        "Location": f"{payload.ward} {payload.district}",
        "Type of House": house_type,
        "Land Area": float(payload.area_m2),
        "Bedrooms": float(payload.bedrooms),
        "Toilets": float(payload.bathrooms),
        "Total Floors": float(payload.floors),
        "Legal Documents": payload.legal,
    }

    df = pd.DataFrame([data])

    print("\n========== FEATURES ==========")
    print(df.columns.tolist())
    print(df.head())

    return df


# =========================================================
# PREDICT
# =========================================================
def predict_price(payload):

    model = get_model()

    df = preprocess(payload)

    try:
        preds = model.predict(df)
    except Exception as e:
        raise ValueError(f"Prediction failed: {str(e)}")

    preds = np.asarray(preds).reshape(-1)

    if len(preds) == 0:
        raise ValueError("Empty prediction")

    pred = float(preds[0])

    # log1p inverse
    pred = float(np.expm1(pred))

    if not np.isfinite(pred):
        pred = 0.0

    pred = max(pred, 0.0)

    return round(pred, 2)