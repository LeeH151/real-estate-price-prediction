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
MAX_PRICE = 1000

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
    parts = [p.strip() for p in text.split(",") if p.strip()]
    return ",".join(parts)

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

    a = normalize_number(m.group(1))
    b = normalize_number(m.group(2))

    if not np.isfinite(a) or not np.isfinite(b):
        return np.nan, np.nan

    width = min(a, b)
    length = max(a, b)

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
                if "tỷ" in text:
                    value = num
                elif "triệu" in text:
                    value = num / 1000.0
                elif "nghìn" in text:
                    value = num / 1_000_000.0
                else:
                    return np.nan
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
    m = DIM_PATTERN.search(text)

    if m:

        w = normalize_number(m.group(1))
        h = normalize_number(m.group(2))

        if np.isfinite(w) and np.isfinite(h):

            area = w * h

            if MIN_AREA <= area <= MAX_AREA:
                return float(area)
    return np.nan
# =========================================================
# ROOM PARSER
# =========================================================
ROOM_PATTERN = re.compile(
    r"(\d+)"
)

def parse_room(text):

    if text is None:
        return np.nan

    nums = ROOM_PATTERN.findall(str(text))

    if not nums:
        return np.nan

    return float(nums[0])
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
def make_property_fingerprint(df):

    def col(name):
        if name in df.columns:
            return df[name]
        return pd.Series([""] * len(df))
    
    loc = col("Location").map(normalize_location)
    house = col("Type of House").map(normalize_text)

    area = col("Land Area").map(parse_area).fillna(-1).round(0).astype(int)
    bed = col("Bedrooms").map(parse_room).fillna(-1).astype(int)
    toilet = col("Toilets").map(parse_room).fillna(-1).astype(int)
    floors = col("Total Floors").map(parse_room).fillna(-1).astype(int)
    legal = col("Legal Documents").map(normalize_text)

    width = []
    length = []

    for x in col("Land Area"):
        w,l = extract_dimensions(x)
        width.append(w)
        length.append(l)

    width = pd.Series(width).fillna(-1).round().astype(int)
    length = pd.Series(length).fillna(-1).round().astype(int)
    ward = (
        loc
        .str.extract(
            r"(phuong \d+|phuong \w+|xa \w+)"
        )[0]
        .fillna("unknown")
    )
    base = (
        loc
        + "|"
        + ward
        + "|"
        + house
        + "|"
        + area.astype(str)
        + "|"
        + width.astype(str)
        + "|"
        + length.astype(str)
        + "|"
        + bed.astype(str)
        + "|"
        + toilet.astype(str)
        + "|"
        + floors.astype(str)
    )
    return base.map(lambda x: hashlib.md5(x.encode()).hexdigest())

# =========================================================
# CLUSTER KEY
# =========================================================
def create_cluster_key(df):

    if "Location" not in df.columns:
        loc = pd.Series(["unknown"] * len(df))
    else:
        loc = (
            df["Location"]
            .fillna("")
            .map(normalize_location)
        )

    if "Type of House" not in df.columns:
        house = pd.Series(["unknown"] * len(df))
    else:
        house = (
            df["Type of House"]
            .fillna("")
            .map(normalize_text)
        )

    if "Land Area" not in df.columns:
        area = pd.Series([-1] * len(df))
    else:
        area = (
            df["Land Area"]
            .map(parse_area)
            .round(-1)
        )

    return (
        loc
        + "|"
        + house
        + "|"
        + area.astype(str)
    )
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