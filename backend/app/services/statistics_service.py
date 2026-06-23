import asyncio
from typing import Dict, Any, List
from app.core.database import prediction_col

# =========================
# DISTRICT COORDS
# =========================
DISTRICT_COORDS = {
    "Quận 1": (10.7756587, 106.7004243),
    "Quận 3": (10.7828581, 106.6865025),
    "Quận 4": (10.757826, 106.701239),
    "Quận 5": (10.754027, 106.663374),
    "Quận 6": (10.748096, 106.634332),
    "Quận 7": (10.736573, 106.722172),
    "Quận 8": (10.722741, 106.628610),
    "Bình Thạnh": (10.801465, 106.707709),
    "Thủ Đức": (10.850637, 106.771332),
}

def safe_float(v, default=0.0):
    try:
        return float(v)
    except:
        return default


# =========================
# HEATMAP
# =========================
async def build_heatmap(collection):
    pipeline = [
        {
            "$group": {
                "_id": "$input_data.district",
                "avg_price": {"$avg": "$result.predicted_price_billion_vnd"},
            }
        }
    ]

    rows = await collection.aggregate(pipeline).to_list(None)

    heatmap = []
    max_price = 0

    for r in rows:
        district = r["_id"]
        avg_price = safe_float(r["avg_price"])

        max_price = max(max_price, avg_price)

        lat, lng = DISTRICT_COORDS.get(district, (10.762622, 106.660172))

        heatmap.append({
            "district": district,
            "lat": lat,
            "lng": lng,
            "avg_price": round(avg_price * 1_000_000_000),
        })

    for item in heatmap:
        base = item["avg_price"] / 1_000_000_000
        item["intensity"] = round(base / max_price, 2) if max_price else 0

    return heatmap


# =========================
# TOP DISTRICTS
# =========================
async def build_top_districts(collection):
    pipeline = [
        {
            "$group": {
                "_id": "$input_data.district",
                "avg_price": {"$avg": "$result.predicted_price_billion_vnd"},
            }
        },
        {"$sort": {"avg_price": -1}},
        {"$limit": 5}
    ]

    rows = await collection.aggregate(pipeline).to_list(None)

    return [
        {
            "district": r["_id"],
            "avg_price": round(safe_float(r["avg_price"]) * 1_000_000_000)
        }
        for r in rows
    ]


# =========================
# PRICE DISTRIBUTION
# =========================
async def build_price_distribution(collection):
    pipeline = [
        {
            "$bucket": {
                "groupBy": "$result.predicted_price_billion_vnd",
                "boundaries": [0, 1, 3, 5, 10, 20],
                "default": "20+",
                "output": {"count": {"$sum": 1}}
            }
        }
    ]

    rows = await collection.aggregate(pipeline).to_list(None)

    return [
        {
            "range": str(r["_id"]),
            "count": r["count"]
        }
        for r in rows
    ]


# =========================
# HOURLY
# =========================
async def build_hourly(collection):
    pipeline = [
        {"$project": {"hour": {"$hour": "$created_at"}}},
        {"$group": {"_id": "$hour", "count": {"$sum": 1}}},
        {"$sort": {"_id": 1}}
    ]

    rows = await collection.aggregate(pipeline).to_list(None)

    max_count = max([r["count"] for r in rows], default=1)

    return [
        {
            "hour": str(r["_id"]),
            "count": r["count"],
            "is_peak": r["count"] == max_count
        }
        for r in rows
    ]


# =========================
# MAIN SERVICE
# =========================
async def get_statistics_service() -> Dict[str, Any]:
    collection = prediction_col()

    heatmap, top, dist, hourly = await asyncio.gather(
        build_heatmap(collection),
        build_top_districts(collection),
        build_price_distribution(collection),
        build_hourly(collection)
    )

    return {
        "heatmap_areas": heatmap,
        "top_districts": top,
        "price_distribution": dist,
        "hourly_predictions": hourly
    }