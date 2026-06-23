from fastapi import (
    APIRouter,
    Depends,
    HTTPException,
    Path,
    Query,
)

from typing import List, Optional
import json

from app.schemas.prediction import (
    PredictionInput,
    PredictionOutput,
    SavePredictionRequest,
)

from app.services.prediction_service import (
    predict_price,
    save_prediction,
    get_history,
    delete_prediction,
    admin_delete_prediction,
    compare_predictions,
)

from app.core.security import (
    get_current_user,
    get_current_user_optional,
)

from app.core.database import prediction_col


router = APIRouter(
    prefix="/predictions",
    tags=["Predictions"]
)


# =========================================================
# HELPER
# =========================================================
def extract_user(current_user: Optional[dict]):

    if not current_user:
        return "guest"

    uid = current_user.get("user_id")

    return str(uid) if uid else "guest"


# =========================================================
# PREDICT
# =========================================================
@router.post(
    "/predict",
    response_model=PredictionOutput
)
async def predict(

    payload: PredictionInput,

    current_user: Optional[dict] = Depends(
        get_current_user_optional
    ),
):

    try:

        user_id = extract_user(current_user)

        result = await predict_price(payload)

        pred = result.get(
            "predicted_price_billion_vnd",
            0
        )

        if pred <= 0:
            raise HTTPException(
                status_code=400,
                detail="Invalid prediction result"
            )

        collection = prediction_col()

        # =====================================================
        # STABLE DEDUP KEY (FIXED)
        # =====================================================
        def normalize_input(payload: PredictionInput):
            return {
                **payload.model_dump(),
                "ward": payload.ward.strip().lower(),
                "district": payload.district.strip().lower(),
                "house_type": payload.house_type.strip(),
                "legal": (payload.legal or "").strip().lower(),
            }
        payload_dict = json.loads(
            json.dumps(
                normalize_input(payload),
                sort_keys=True
            )
        )

        existing = await collection.find_one({

            "user_id": user_id,
            "input_data": payload_dict
        })

        # =====================================================
        # RETURN EXISTING (FULL CONSISTENCY FIX)
        # =====================================================
        if existing:

            old_result = existing.get("result", {})

            return PredictionOutput(

                predicted_price_billion_vnd=
                    old_result.get("predicted_price_billion_vnd", 0),

                min_price=
                    old_result.get("min_price", 0),

                max_price=
                    old_result.get("max_price", 0),

                confidence_score=
                    old_result.get("confidence_score", 0),

                comparison=
                    old_result.get("comparison", []),

                trend=
                    old_result.get("trend", []),

                model_version=
                    old_result.get("model_version", "v5.0.0")
            )

        # =====================================================
        # AUTO SAVE
        # =====================================================
        await save_prediction(

            user_id=user_id,
            payload=payload,
            result=result,
        )

        return PredictionOutput(**result)

    except ValueError as e:

        raise HTTPException(
            status_code=400,
            detail=str(e)
        )

    except HTTPException:
        raise

    except Exception as e:

        print("PREDICT ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail="Prediction failed"
        )


# =========================================================
# SAVE
# =========================================================
@router.post("/save")
async def save_prediction_endpoint(

    data: SavePredictionRequest,

    current_user: Optional[dict] = Depends(
        get_current_user_optional
    ),
):

    try:

        user_id = extract_user(current_user)

        collection = prediction_col()

        payload_dict = json.loads(
            json.dumps(
                data.input_data.model_dump(),
                sort_keys=True
            )
        )

        existing = await collection.find_one({

            "user_id": user_id,
            "input_data": payload_dict
        })

        if existing:

            return {
                "success": False,
                "message": "Prediction already saved",
                "prediction_id": existing.get("prediction_id")
            }

        doc = await save_prediction(

            user_id=user_id,
            payload=data.input_data,
            result=data.result,
        )

        return {
            "success": True,
            **doc
        }

    except HTTPException:
        raise

    except Exception as e:

        print("SAVE ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail="Save failed"
        )


# =========================================================
# HISTORY
# =========================================================
@router.get("/history")
async def history(

    current_user: Optional[dict] = Depends(
        get_current_user_optional
    ),

    limit: int = Query(50, ge=1, le=200),
):

    try:

        user_id = extract_user(current_user)

        data = await get_history(
            user_id=user_id,
            limit=limit
        )

        return {
            "user_id": user_id,
            "count": len(data),
            "history": data
        }

    except Exception as e:

        print("HISTORY ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail="History failed"
        )


# =========================================================
# DELETE
# =========================================================
@router.delete("/history/{prediction_id}")
async def delete_history_item(

    prediction_id: str = Path(...),

    current_user: dict = Depends(
        get_current_user
    ),
):

    try:

        collection = prediction_col()

        doc = await collection.find_one({
            "prediction_id": prediction_id
        })

        if not doc:
            raise HTTPException(
                status_code=404,
                detail="Prediction not found"
            )

        if current_user.get("role") == "admin":

            ok = await admin_delete_prediction(
                prediction_id
            )

            return {"success": ok}

        if str(current_user.get("user_id")) != str(doc.get("user_id")):

            raise HTTPException(
                status_code=403,
                detail="No permission"
            )

        ok = await delete_prediction(
            current_user["user_id"],
            prediction_id
        )

        return {"success": ok}

    except HTTPException:
        raise

    except Exception as e:

        print("DELETE ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail="Delete failed"
        )


# =========================================================
# COMPARE
# =========================================================
@router.post("/compare")
async def compare(payloads: List[PredictionInput]):

    if len(payloads) < 2:
        raise HTTPException(
            status_code=400,
            detail="Need at least 2 items"
        )

    try:
        return await compare_predictions(payloads)

    except Exception as e:

        print("COMPARE ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail="Compare failed"
        )


# =========================================================
# COUNT
# =========================================================
@router.get("/count")
async def count(

    current_user: Optional[dict] = Depends(
        get_current_user_optional
    )
):

    try:

        user_id = extract_user(current_user)

        collection = prediction_col()

        query = (
            {}
            if current_user and current_user.get("role") == "admin"
            else {"user_id": user_id}
        )

        total = await collection.count_documents(query)

        return {
            "user_id": user_id,
            "count": total
        }

    except Exception as e:

        print("COUNT ERROR:", e)

        raise HTTPException(
            status_code=500,
            detail="Count failed"
        )