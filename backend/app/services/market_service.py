from typing import Dict, List
from datetime import datetime
from app.core.database import get_db


# =========================
# DISTRICT COORDINATES (HCM)
# =========================
DISTRICT_COORDS = {
    "Quận 1": (10.7756587, 106.7004243),
    "Quận 3": (10.7828581, 106.6865025),
    "Quận 4": (10.757826, 106.701239),
    "Quận 5": (10.754027, 106.663374),
    "Quận 6": (10.748096, 106.634332),
    "Quận 7": (10.736573, 106.722172),
    "Quận 8": (10.722741, 106.628610),
    "Quận 10": (10.774596, 106.667954),
    "Quận 11": (10.762973, 106.650084),
    "Quận 12": (10.867153, 106.641332),
    "Bình Tân": (10.737548, 106.603946),
    "Bình Thạnh": (10.801465, 106.707709),
    "Phú Nhuận": (10.800110, 106.679350),
    "Thủ Đức": (10.850637, 106.771332),
    "Tân Bình": (10.802000, 106.652000),
    "Tân Phú": (10.791640, 106.629684),
    "Gò Vấp": (10.838678, 106.665290),
    "Nhà Bè": (10.695264, 106.704874),
    "Hóc Môn": (10.888219, 106.595090),
    "Củ Chi": (11.006685, 106.513852),
    "Bình Chánh": (10.687153, 106.593853),
}


# =========================
# SAFE FLOAT
# =========================
def safe_float(v, default=0.0):
    try:
        if v is None:
            return default
        v = float(v)
        if v != v:  # NaN check
            return default
        return v
    except:
        return default


# =========================
# EMPTY RESPONSE
# =========================
def empty_response():
    return {
        "avg_price": 0,
        "growth": 0,
        "transactions": 0,
        "line_chart": [],
        "top_growth": [],
        "district_chart": [],
        "heatmap": []
    }


# =========================
# MARKET STATS (FAST - AGGREGATION)
# =========================
async def get_market_stats() -> Dict:
    db = get_db()

    pipeline = [
        {
            "$group": {
                "_id": None,
                "avg_price": {"$avg": "$result.predicted_price_billion_vnd"},
                "total_listings": {"$sum": 1},
            }
        }
    ]

    data = await db.prediction_history.aggregate(pipeline).to_list(1)

    if not data:
        return {"avg_price": 0, "total_listings": 0, "prediction_count": 0}

    avg_price = safe_float(data[0].get("avg_price"))
    total = data[0].get("total_listings", 0)

    return {
        "avg_price": round(avg_price, 2),
        "total_listings": total,
        "prediction_count": total
    }


# =========================
# HEATMAP (REAL LOCATION MAPPING FIXED)
# =========================
async def get_heatmap() -> Dict:
    db = get_db()

    pipeline = [
        {
            "$group": {
                "_id": "$input_data.district",
                "avg_price": {"$avg": "$result.predicted_price_billion_vnd"},
            }
        }
    ]

    rows = await db.prediction_history.aggregate(pipeline).to_list(None)

    points = []

    for r in rows:
        district = r.get("_id")
        avg_price = safe_float(r.get("avg_price"))

        # fallback coords nếu district lạ
        lat, lng = DISTRICT_COORDS.get(
            district,
            (10.762622, 106.660172)
        )

        points.append({
            "district": district,
            "lat": lat,
            "lng": lng,
            "avg_price_million_per_m2": round(avg_price, 2),
            "growth_pct": 0
        })

    return {
        "points": points,
        "max_value": max([p["avg_price_million_per_m2"] for p in points], default=0)
    }


# =========================
# MARKET ANALYSIS (FIXED + SAFE + SORTED)
# =========================
async def get_market_analysis(year: int, quarter: str) -> Dict:
    db = get_db()

    quarter_map = {
        "Q1": [1, 2, 3],
        "Q2": [4, 5, 6],
        "Q3": [7, 8, 9],
        "Q4": [10, 11, 12],
    }

    months = quarter_map.get(quarter.upper(), [])
    if not months:
        return empty_response()

    data = await db.prediction_history.find({}).to_list(None)

    filtered = []

    for d in data:
        created = d.get("created_at")

        # robust datetime parse
        try:
            if isinstance(created, str):
                created = datetime.fromisoformat(created.replace("Z", ""))
            elif not isinstance(created, datetime):
                continue
        except:
            continue

        price = d.get("result", {}).get("predicted_price_billion_vnd")

        if (
            created.year == int(year)
            and created.month in months
            and price is not None
        ):
            filtered.append(d)

    if len(filtered) == 0:
        return empty_response()

    # ================= LINE =================
    line_map = {}

    for d in filtered:
        created = d["created_at"]
        month = f"Month {created.month}"        
        price = safe_float(d.get("result", {}).get("predicted_price_billion_vnd"))
        line_map.setdefault(month, []).append(price)

    line_chart = [
        {"period": k, "price": round(sum(v) / len(v), 2)}
        for k, v in sorted(
            line_map.items(),
            key=lambda x: int(x[0].split()[-1])
)    ]

    # ================= DISTRICT =================
    district_map = {}

    for d in filtered:
        district = d.get("input_data", {}).get("district", "Unknown")
        price = safe_float(d.get("result", {}).get("predicted_price_billion_vnd"))
        district_map.setdefault(district, []).append(price)

    top_growth = []

    for district, prices in district_map.items():
        if len(prices) == 0:
            continue
        avg_price = sum(prices) / len(prices)
        growth = 0
        if len(prices) > 1 and prices[0] != 0:
            growth = ((prices[-1] - prices[0]) / prices[0]) * 100

        top_growth.append({
            "district": district,
            "price": round(avg_price, 2),
            "growth": round(growth, 2)
        })

    top_growth.sort(key=lambda x: x["growth"], reverse=True)

    district_chart = [
        {"district": x["district"], "price": x["price"]}
        for x in top_growth[:5]
    ]

    # ================= SUMMARY =================
    prices = [
        safe_float(d.get("result", {}).get("predicted_price_billion_vnd"))
        for d in filtered
        if d.get("result", {}).get("predicted_price_billion_vnd") is not None
    ]

    avg_price = round(sum(prices) / len(prices), 2) if prices else 0
    transactions = len(prices)

    growth = 0
    if len(prices) > 1 and prices[0] != 0:
        growth = ((prices[-1] - prices[0]) / prices[0]) * 100

    # ================= HEATMAP FIX =================
    heatmap_raw = await get_heatmap()

    return {
        "avg_price": avg_price,
        "transactions": transactions,
        "growth": round(growth, 2),
        "line_chart": line_chart,
        "top_growth": top_growth,
        "district_chart": district_chart,
        "heatmap": heatmap_raw["points"]
    }