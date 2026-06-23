from typing import Dict, Any
from app.core.database import model_col


# =========================
# GET MODELS
# =========================
async def get_models_service(
    page: int = 1,
    page_size: int = 10,
    search: str = ""
) -> Dict[str, Any]:

    collection = model_col()

    query = {}
    if search:
        query["name"] = {"$regex": search, "$options": "i"}

    skip = (page - 1) * page_size

    cursor = (
        collection.find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(page_size)
    )

    items = []

    async for doc in cursor:
        items.append({
            "model_id": doc.get("model_id"),
            "name": doc.get("name"),
            "version": doc.get("version"),
            "status": doc.get("status", "inactive"),
            "accuracy": doc.get("accuracy"),
            "created_at": doc.get("created_at"),
        })

    total = await collection.count_documents(query)

    return {
        "items": items,
        "total": total
    }