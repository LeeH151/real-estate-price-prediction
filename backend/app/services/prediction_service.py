import uuid
import pandas as pd
import numpy as np

from datetime import datetime, timezone
from typing import List, Dict, Any, Optional

from app.schemas.prediction import PredictionInput
from app.core.database import prediction_col, user_col
from app.core.model import get_model
from app.ml.model.feature_builder import FeatureBuilder

# =========================================================
# TIME
# =========================================================
def now():
    return datetime.now(timezone.utc)


# =========================================================
# UUID
# =========================================================
def generate_prediction_id():
    return str(uuid.uuid4())


# =========================================================
# SAFE FLOAT
# =========================================================
def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        value = float(v)
        if np.isnan(value) or np.isinf(value):
            return default
        return value
    except Exception:
        return default


# =========================================================
# CONFIDENCE (FIXED STABLE)
# =========================================================
def estimate_confidence(payload: PredictionInput, predicted_price: float):

    score = 0.80

    if payload.bedrooms > 0:
        score += 0.02
    if payload.bathrooms > 0:
        score += 0.02
    if payload.floors > 0:
        score += 0.02

    legal = (payload.legal or "").lower()
    if "sổ hồng" in legal or "so hong" in legal:
        score += 0.03

    # penalty extreme values
    if predicted_price > 50:
        score -= 0.06
    elif predicted_price > 30:
        score -= 0.03

    return float(np.clip(score, 0.55, 0.95))


# =========================================================
# PREDICT CORE
# =========================================================
async def predict_price(payload: PredictionInput) -> Dict[str, Any]:

    model = get_model()
    df = payload.to_model_input()
    
    '''location = f"{payload.ward}, {payload.district}"

    df_raw = pd.DataFrame([{
        "Location": location,
        "Land Area": f"{payload.area_m2} m²",
        "Bedrooms": payload.bedrooms,
        "Toilets": payload.bathrooms,
        "Total Floors": payload.floors,
        "Type of House": payload.house_type,
        "Legal Documents": payload.legal or "unknown"
    }])
    builder = FeatureBuilder()
    X = builder.transform(df_raw)

    pred_log = model.predict(X)   '''
    pred_log = model.predict(df)

    if len(pred_log) == 0:
        raise ValueError("Empty prediction")

    # =====================================================
    # LOG -> PRICE
    # =====================================================
    predicted_price = float(np.expm1(pred_log[0]))

    if not np.isfinite(predicted_price):
        predicted_price = 0.0

    predicted_price = max(predicted_price, 0.0)

    # =====================================================
    # 🔥 FIX SCALE BIAS (IMPORTANT FOR VIET NAM REAL ESTATE)
    # =====================================================
    if predicted_price < 1:
        predicted_price *= 1.7
    elif predicted_price < 3:
        predicted_price *= 1.3

    # =====================================================
    # CONFIDENCE
    # =====================================================
    confidence = estimate_confidence(payload, predicted_price)

    # =====================================================
    # RANGE (REALISTIC)
    # =====================================================
    spread = max(0.10, 0.22 - confidence * 0.1)

    min_price = predicted_price * (1 - spread)
    max_price = predicted_price * (1 + spread)

    # =====================================================
    # COMPARISON (FIXED SEMANTIC)
    # =====================================================
    comparison = [
        {
            "label": "Giá thấp",
            "value": round(predicted_price * 0.78, 2)
        },
        {
            "label": "Giá thị trường",
            "value": round(predicted_price, 2)
        },
        {
            "label": "Giá cao",
            "value": round(predicted_price * 1.28, 2)
        }
    ]

    # =====================================================
    # TREND
    # =====================================================
    trend = [
        {
            "period": "3 tháng trước",
            "value": round(predicted_price * 0.93, 2)
        },
        {
            "period": "Hiện tại",
            "value": round(predicted_price, 2)
        },
        {
            "period": "3 tháng tới",
            "value": round(predicted_price * 1.07, 2)
        }
    ]

    return {
        "predicted_price_billion_vnd": round(predicted_price, 2),
        "min_price": round(min_price, 2),
        "max_price": round(max_price, 2),
        "confidence_score": round(confidence, 2),
        "comparison": comparison,
        "trend": trend,
        "model_version": "v5.1.0-fixed"
    }


# =========================================================
# SAVE (FIXED - STORE FULL RESULT)
# =========================================================
async def save_prediction(
    user_id: str,
    payload: PredictionInput,
    result: Dict[str, Any],
):

    collection = prediction_col()
    users = user_col()

    user_name = "Khách"

    if user_id != "guest":
        user = await users.find_one({"user_id": user_id})
        if user:
            user_name = user.get("name", "Khách")

    doc = {
        "prediction_id": generate_prediction_id(),
        "user_id": str(user_id),
        "user_name": user_name,
        "input_data": payload.model_dump(),

        # 🔥 FIX: store FULL result (critical)
        "result": result,

        "created_at": now(),
    }

    await collection.insert_one(doc)

    return {
        "prediction_id": doc["prediction_id"]
    }


# =========================================================
# HISTORY (UNCHANGED BUT SAFE)
# =========================================================
async def get_history(user_id: Optional[str], limit: int = 50):

    collection = prediction_col()

    cursor = (
        collection.find({"user_id": str(user_id)})
        .sort("created_at", -1)
        .limit(limit)
    )

    results = []

    async for doc in cursor:
        results.append({
            "prediction_id": doc.get("prediction_id"),
            "user_name": doc.get("user_name"),
            "input_data": doc.get("input_data"),
            "result": doc.get("result"),
            "created_at": doc.get("created_at"),
        })

    return results


# =========================================================
# DELETE
# =========================================================
async def delete_prediction(user_id: str, prediction_id: str):

    collection = prediction_col()

    result = await collection.delete_one({
        "prediction_id": prediction_id,
        "user_id": str(user_id)
    })

    return result.deleted_count > 0


# =========================================================
# ADMIN DELETE
# =========================================================
async def admin_delete_prediction(prediction_id: str):

    collection = prediction_col()

    result = await collection.delete_one({
        "prediction_id": prediction_id
    })

    return result.deleted_count > 0


# =========================================================
# COMPARE (FIX SAFE)
# =========================================================
async def compare_predictions(payloads: List[PredictionInput]):

    results = []

    for idx, payload in enumerate(payloads):

        try:
            pred = await predict_price(payload)

            results.append({
                "index": idx,
                "input": payload.model_dump(),
                **pred
            })

        except Exception as e:
            results.append({
                "index": idx,
                "error": str(e)
            })

    valid = [r for r in results if "error" not in r]

    prices = [r["predicted_price_billion_vnd"] for r in valid]

    summary = (
        {
            "highest_price": max(prices),
            "lowest_price": min(prices),
            "average_price": round(sum(prices) / len(prices), 2),
            "price_difference": round(max(prices) - min(prices), 2),
        }
        if prices else {}
    )

    return {
        "individual_results": results,
        "summary": summary
    }