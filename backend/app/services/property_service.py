from app.core.database import get_db
from datetime import datetime
from bson import ObjectId
from fastapi import HTTPException


# =========================
# FORMAT RESPONSE
# =========================
def format_property(doc):
    return {
        "id": str(doc["_id"]),
        "title": doc.get("title"),
        "district": doc.get("district"),
        "ward": doc.get("ward"),
        "price": doc.get("price"),
        "area": doc.get("area"),
        "type": doc.get("type"),
        "created_at": doc.get("created_at"),
    }


# =========================
# GET PROPERTIES (PRO)
# =========================
async def get_properties_service(page, page_size, search, district, ward):
    db = get_db()

    query = {}

    # 🔍 SEARCH MULTI FIELD
    if search:
        query["$or"] = [
            {"title": {"$regex": search, "$options": "i"}},
            {"district": {"$regex": search, "$options": "i"}},
            {"ward": {"$regex": search, "$options": "i"}},
        ]

    if district:
        query["district"] = district

    if ward:
        query["ward"] = ward

    skip = (page - 1) * page_size

    cursor = (
        db["properties"]
        .find(query)
        .sort("created_at", -1)
        .skip(skip)
        .limit(page_size)
    )

    raw = await cursor.to_list(length=page_size)

    # =========================
    # 🔥 REMOVE DUPLICATE (IMPORTANT)
    # =========================
    unique_map = {}
    for item in raw:
        key = f"{item.get('title')}_{item.get('district')}_{item.get('ward')}"
        unique_map[key] = item

    data = [format_property(v) for v in unique_map.values()]

    total = await db["properties"].count_documents(query)

    return {
        "data": data,
        "pagination": {
            "page": page,
            "page_size": page_size,
            "total": total,
            "total_pages": (total + page_size - 1) // page_size
        }
    }


# =========================
# CREATE PROPERTY (SAFE)
# =========================
async def create_property_service(payload):
    db = get_db()

    data = payload.model_dump(exclude_none=True)

    # =========================
    # AUTO TITLE
    # =========================
    if not data.get("title"):
        data["title"] = f"{data.get('type', 'Property')} {data.get('district', '')}"

    # =========================
    # 🔥 PREVENT DUPLICATE
    # =========================
    existing = await db["properties"].find_one({
        "title": data.get("title"),
        "district": data.get("district"),
        "ward": data.get("ward")
    })

    if existing:
        raise HTTPException(
            status_code=400,
            detail="Property already exists"
        )

    # =========================
    # META DATA
    # =========================
    data["created_at"] = datetime.utcnow()

    result = await db["properties"].insert_one(data)

    data["id"] = str(result.inserted_id)

    return {
        "message": "Property created successfully",
        "data": data
    }