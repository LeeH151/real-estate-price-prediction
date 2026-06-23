import pandas as pd

from model.utils import (
    parse_price,
    parse_area,
    normalize_location,
    normalize_text
)

# =====================================================
# LOAD
# =====================================================
df = pd.read_csv("data/dataset.csv")

# =====================================================
# CLEAN
# =====================================================
df["price_num"] = df["Price"].map(parse_price)

df["area_num"] = df["Land Area"].map(parse_area)

df["Location"] = (
    df["Location"]
    .fillna("")
    .astype(str)
    .map(normalize_location)
)

df["Type of House"] = (
    df["Type of House"]
    .fillna("")
    .astype(str)
    .map(normalize_text)
)

# =====================================================
# FILTER SIMILAR
# =====================================================
result = df[

    df["Location"].str.contains("binh thanh")

    &

    df["Type of House"].str.contains("mat tien")

    &

    (df["area_num"] >= 35)

    &

    (df["area_num"] <= 50)

]

# =====================================================
# SHOW
# =====================================================
print("\nSố mẫu tương tự:", len(result))

print("\nGiá trung bình:")
print(result["price_num"].describe())

print("\nTop giá:")
print(
    result[
        [
            "Location",
            "Price",
            "Land Area",
            "Bedrooms",
            "Toilets"
        ]
    ]
    .head(20)
)