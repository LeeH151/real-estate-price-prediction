import re
import unicodedata
import numpy as np
import pandas as pd
import logging
import hashlib

logger = logging.getLogger(__name__)

# =========================================================
# CONFIG
# =========================================================
INVALID_PRICE_KEYWORDS = {
    "thỏa thuận",
    "thoả thuận",
    "liên hệ",
    "contact",
    "deal",
    "giá thỏa thuận",
    "thương lượng",
    "tl"
}

MIN_PRICE = 0.001
MAX_PRICE = 200.0

MIN_AREA = 1
MAX_AREA = 10000

# =========================================================
# REGEX
# =========================================================
AREA_PATTERN = re.compile(
    r"([\d\.,]+)\s*(m2|m²)",
    re.I
)

DIM_PATTERN = re.compile(
    r"\(?\s*([\d\.,]+)\s*[x×]\s*([\d\.,]+)\s*\)?",
    re.I
)

NUM_PATTERN = re.compile(r"[\d\.,]+")

PRICE_PATTERNS = {
    "ty": re.compile(r"([\d\.,]+)\s*tỷ", re.I),
    "trieu": re.compile(r"([\d\.,]+)\s*triệu", re.I),
    "nghin": re.compile(r"([\d\.,]+)\s*nghìn", re.I),
    "vnd": re.compile(r"([\d\.,]+)\s*(đ|vnđ|vnd)", re.I),
}

# =========================================================
# SAFE SERIES
# =========================================================
def ensure_series(value, index=None):

    if isinstance(value, pd.Series):
        return value

    if index is None:
        return pd.Series([value])

    return pd.Series([value] * len(index), index=index)

# =========================================================
# TEXT NORMALIZATION
# =========================================================
def normalize_text(text):

    if text is None:
        return ""

    if isinstance(text, float) and np.isnan(text):
        return ""

    text = str(text).strip().lower()

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = text.replace("đ", "d")

    text = re.sub(r"[^a-z0-9\s]", " ", text)

    return re.sub(r"\s+", " ", text).strip()

# =========================================================
# LOCATION NORMALIZATION
# =========================================================
def normalize_location(text):

    text = normalize_text(text)

    if not text:
        return ""

    text = re.sub(
        r"\bq[\.\s]*(\d+)\b",
        r"quan \1",
        text
    )

    text = text.replace(
        "tp thu duc",
        "thu duc"
    )

    text = text.replace(
        "tphcm",
        "ho chi minh"
    )

    return text

# =========================================================
# NUMBER NORMALIZATION
# =========================================================
def normalize_number(text):

    if text is None:
        return np.nan

    text = str(text).strip()

    if not text:
        return np.nan

    try:

        text = text.replace(" ", "")

        # VN format
        if re.match(r"^\d{1,3}(\.\d{3})+(,\d+)?$", text):
            text = text.replace(".", "")
            text = text.replace(",", ".")

        # US format
        elif re.match(r"^\d{1,3}(,\d{3})+(\.\d+)?$", text):
            text = text.replace(",", "")

        else:
            text = text.replace(",", ".")

        value = float(text)

        if np.isfinite(value):
            return value

        return np.nan

    except:
        return np.nan

# =========================================================
# PARSE DIMENSIONS
# =========================================================
def extract_dimensions(x):

    if x is None:
        return np.nan, np.nan

    text = str(x)

    m = DIM_PATTERN.search(text)

    if not m:
        return np.nan, np.nan

    width = normalize_number(m.group(1))
    length = normalize_number(m.group(2))

    return width, length

# =========================================================
# PRICE PARSER
# =========================================================
def parse_price(x):

    if x is None:
        return np.nan

    if isinstance(x, float) and np.isnan(x):
        return np.nan

    text = str(x).lower().strip()

    if not text:
        return np.nan

    if any(k in text for k in INVALID_PRICE_KEYWORDS):
        return np.nan

    value = 0.0
    found = False

    for v in PRICE_PATTERNS["ty"].findall(text):

        num = normalize_number(v)

        if np.isfinite(num):
            value += num
            found = True

    for v in PRICE_PATTERNS["trieu"].findall(text):

        num = normalize_number(v)

        if np.isfinite(num):
            value += num / 1000.0
            found = True

    if not found:

        nums = NUM_PATTERN.findall(text)

        if nums:

            num = normalize_number(nums[0])

            if np.isfinite(num):

                if num > 500:
                    value = num / 1e9
                else:
                    value = num

                found = True

    if not found:
        return np.nan

    if value <= 0:
        return np.nan

    return float(np.clip(
        value,
        MIN_PRICE,
        MAX_PRICE
    ))

# =========================================================
# AREA PARSER
# =========================================================
def parse_area(x):

    if x is None:
        return np.nan

    text = str(x).lower()

    # -----------------------------------------------------
    # ƯU TIÊN lấy diện tích chính xác từ "42 m²"
    # -----------------------------------------------------
    m = AREA_PATTERN.search(text)

    if m:

        area = normalize_number(m.group(1))

        if np.isfinite(area):

            if MIN_AREA <= area <= MAX_AREA:
                return float(area)

    # -----------------------------------------------------
    # FALLBACK dùng width x length
    # -----------------------------------------------------
    w, h = extract_dimensions(text)

    if np.isfinite(w) and np.isfinite(h):

        area = w * h

        if MIN_AREA <= area <= MAX_AREA:
            return float(area)

    return np.nan

# =========================================================
# ROOM PARSER
# =========================================================
def parse_room(x):

    if x is None:
        return np.nan

    nums = NUM_PATTERN.findall(str(x))

    if not nums:
        return np.nan

    val = normalize_number(nums[0])

    if np.isfinite(val):
        return float(np.clip(val, 0, 50))

    return np.nan

# =========================================================
# SAFE DIVIDE
# =========================================================
def safe_divide(a, b, eps=1e-8):

    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)

    with np.errstate(divide="ignore", invalid="ignore"):

        out = np.divide(a, b + eps)

    return np.nan_to_num(
        out,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

# =========================================================
# PROPERTY FINGERPRINT
# =========================================================
def make_property_fingerprint(df: pd.DataFrame):

    idx = df.index

    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------
    loc = ensure_series(
        df.get("Location"),
        idx
    ).astype(str)

    loc = (
        loc.fillna("")
        .map(normalize_location)
    )

    # -----------------------------------------------------
    # HOUSE TYPE
    # -----------------------------------------------------
    house = ensure_series(
        df.get("Type of House"),
        idx
    ).astype(str)

    house = (
        house.fillna("")
        .map(normalize_text)
    )

    # -----------------------------------------------------
    # AREA
    # FIX CHÍNH: parse_area trước khi fingerprint
    # -----------------------------------------------------
    area_raw = ensure_series(
        df.get("Land Area"),
        idx
    )

    area = (
        area_raw
        .map(parse_area)
    )

    area = pd.to_numeric(
        area,
        errors="coerce"
    ).fillna(-1)

    area = area.round(1)

    # -----------------------------------------------------
    # BEDROOM
    # -----------------------------------------------------
    bed = ensure_series(
        df.get("Bedrooms"),
        idx
    )

    bed = (
        bed.map(parse_room)
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # -----------------------------------------------------
    # TOILET
    # -----------------------------------------------------
    toilet = ensure_series(
        df.get("Toilets"),
        idx
    )

    toilet = (
        toilet.map(parse_room)
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # -----------------------------------------------------
    # FLOORS
    # -----------------------------------------------------
    floors = ensure_series(
        df.get("Total Floors"),
        idx
    )

    floors = (
        floors.map(parse_room)
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # -----------------------------------------------------
    # LEGAL
    # -----------------------------------------------------
    legal = ensure_series(
        df.get("Legal Documents"),
        idx
    ).astype(str)

    legal = (
        legal.fillna("")
        .map(normalize_text)
    )

    # -----------------------------------------------------
    # BUILD FINGERPRINT
    # -----------------------------------------------------
    base = (
        loc
        + "_"
        + house
        + "_"
        + area.astype(str)
        + "_"
        + bed
        + "_"
        + toilet
        + "_"
        + floors
        + "_"
        + legal
    )

    return base.apply(
        lambda x: hashlib.md5(
            x.encode("utf-8")
        ).hexdigest()[:16]
    )

# =========================================================
# CLUSTER KEY
# =========================================================
def create_cluster_key(df: pd.DataFrame):

    fp = make_property_fingerprint(df)

    return fp.str[:12]

# =========================================================
# SANITY CHECK DATAFRAME
# =========================================================
def sanity_check_dataframe(df):

    logger.info("=" * 60)

    logger.info(f"📊 SHAPE: {df.shape}")

    miss = (
        df.isna()
        .mean()
        .sort_values(ascending=False)
    )

    logger.info(
        f"\n🔎 Missing rate (top 10):\n{miss.head(10)}"
    )

    dup_rate = df.duplicated().mean()

    logger.info(
        f"\n🔁 Exact duplicate rate: {dup_rate:.4f}"
    )

    if {
        "Location",
        "Land Area",
        "Price"
    }.issubset(df.columns):

        semantic_dup = (
            df.groupby(
                ["Location", "Land Area", "Price"],
                dropna=False
            )
            .size()
            .mean()
        )

        logger.info(
            f"🧠 Semantic duplication signal: {semantic_dup:.2f}"
        )

    if "Price" in df.columns:

        logger.info(
            f"\n💰 Price stats:\n{df['Price'].describe()}"
        )

    if "Land Area" in df.columns:

        logger.info(
            f"\n📐 Area stats:\n{df['Land Area'].describe()}"
        )

    logger.info("=" * 60)

    return True