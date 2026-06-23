from fastapi import APIRouter, Depends, Query, HTTPException
from app.api.deps import require_admin
from app.core.database import prediction_col, user_col
from app.schemas.prediction import PredictionHistoryItem, PredictionInput, PredictionResult
import datetime

router = APIRouter(
    prefix="/admin/predictions",
    tags=["Admin Predictions"]
)

# =========================
# HELPER
# =========================
def serialize_prediction(doc) -> PredictionHistoryItem:
    input_data = PredictionInput(**doc.get("input_data", {}))
    result = PredictionResult(**doc.get("result", {}))
    created_at = doc.get("created_at", datetime.datetime.now())

    return PredictionHistoryItem(
        prediction_id=doc["prediction_id"],
        user_id=doc.get("user_id"),
        input_data=input_data,
        result=result,
        created_at=created_at
    )


# =========================
# GET PREDICTIONS LIST (Admin)
# =========================
@router.get("")
async def get_predictions(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=100),
    search: str = Query("", alias="search"),
    admin_user=Depends(require_admin)
):
    try:
        collection = prediction_col()
        users_collection = user_col()

        skip = (page - 1) * page_size

        # =========================
        # QUERY
        # =========================
        query = {}
        if search:
            query["$or"] = [
                {"input_data.district": {"$regex": search, "$options": "i"}},
                {"input_data.ward": {"$regex": search, "$options": "i"}},
                {"input_data.house_type": {"$regex": search, "$options": "i"}}
            ]

        total = await collection.count_documents(query)

        cursor = (
            collection.find(query)
            .sort("created_at", -1)
            .skip(skip)
            .limit(page_size)
        )

        docs = await cursor.to_list(length=page_size)

        # =========================
        # LẤY USER IDS
        # =========================
        user_ids = list(set([
            doc.get("user_id")
            for doc in docs
            if doc.get("user_id") and doc.get("user_id") != "guest"
        ]))

        # =========================
        # FETCH USERS (JOIN đúng bằng user_id)
        # =========================
        users = []
        if user_ids:
            users = await users_collection.find({
                "user_id": {"$in": user_ids}
            }).to_list(length=None)

        # =========================
        # MAP user_id -> name
        # =========================
        user_map = {
            user["user_id"]: user.get("name", "Unknown")
            for user in users
        }

        # =========================
        # BUILD RESPONSE
        # =========================
        items = []
        for doc in docs:
            item = serialize_prediction(doc).model_dump()

            uid = doc.get("user_id")

            # 🔥 ƯU TIÊN LẤY TỪ USERS COLLECTION
            if uid in user_map:
                name = user_map[uid]

            # guest
            elif uid == "guest" or not uid:
                name = "Khách"

            # fallback
            else:
                name = f"User-{str(uid)[:6]}"

            item["user_name"] = name
            items.append(item)

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to retrieve predictions: {str(e)}"
        )