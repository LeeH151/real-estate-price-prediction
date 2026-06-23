# =========================================================
# REAL ESTATE DATA VISUALIZATION
# HANDLE DUPLICATED DATASET (26K -> REAL 1K1)
# =========================================================

import os
import hashlib
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# IMPORT UTILS
# =========================================================
from app.ml.model.utils import (
    parse_price,
    parse_area,
    parse_room,
    normalize_location,
    normalize_text
)

# =========================================================
# CONFIG
# =========================================================
DATA_PATH = "app/ml/data/dataset.csv"

MODEL_PATH = "app/ml/artifacts/model.pkl"

OUTPUT_DIR = "artifacts"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================================================
# STYLE
# =========================================================
sns.set_style("whitegrid")

plt.rcParams["figure.figsize"] = (10, 6)

# =========================================================
# PROPERTY FINGERPRINT
# REMOVE DUPLICATES
# =========================================================
def make_property_fingerprint(df):

    # -----------------------------------------------------
    # LOCATION
    # -----------------------------------------------------
    location = (
        df["Location"]
        .fillna("")
        .astype(str)
        .map(normalize_location)
    )

    # -----------------------------------------------------
    # TYPE
    # -----------------------------------------------------
    house_type = (
        df["Type of House"]
        .fillna("")
        .astype(str)
        .map(normalize_text)
    )

    # -----------------------------------------------------
    # AREA
    # -----------------------------------------------------
    area = (
        df["Land Area"]
        .map(parse_area)
        .fillna(-1)
        .round(1)
        .astype(str)
    )

    # -----------------------------------------------------
    # BEDROOM
    # -----------------------------------------------------
    bedrooms = (
        df["Bedrooms"]
        .map(parse_room)
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # -----------------------------------------------------
    # TOILETS
    # -----------------------------------------------------
    toilets = (
        df["Toilets"]
        .map(parse_room)
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # -----------------------------------------------------
    # FLOORS
    # -----------------------------------------------------
    floors = (
        df["Total Floors"]
        .map(parse_room)
        .fillna(-1)
        .astype(int)
        .astype(str)
    )

    # -----------------------------------------------------
    # LEGAL
    # -----------------------------------------------------
    legal = (
        df["Legal Documents"]
        .fillna("")
        .astype(str)
        .map(normalize_text)
    )

    # -----------------------------------------------------
    # COMBINE
    # -----------------------------------------------------
    text = (
        location
        + "_"
        + house_type
        + "_"
        + area
        + "_"
        + bedrooms
        + "_"
        + toilets
        + "_"
        + floors
        + "_"
        + legal
    )

    # -----------------------------------------------------
    # HASH
    # -----------------------------------------------------
    return text.apply(
        lambda x: hashlib.md5(
            x.encode("utf-8")
        ).hexdigest()
    )

# =========================================================
# LOAD DATA
# =========================================================
print("📥 Loading dataset...")

df = pd.read_csv(DATA_PATH)

print(f"📊 Raw dataset shape: {df.shape}")

# =========================================================
# REQUIRED COLUMNS
# =========================================================
required = [

    "Location",
    "Price",
    "Type of House",
    "Land Area",

    "Bedrooms",
    "Toilets",
    "Total Floors",

    "Legal Documents"
]

for col in required:

    if col not in df.columns:

        df[col] = np.nan

# =========================================================
# PARSE NUMERIC
# =========================================================
print("⚙️ Parsing numeric data...")

df["Price"] = df["Price"].map(parse_price)

df["Land Area"] = df["Land Area"].map(parse_area)

df["Bedrooms"] = df["Bedrooms"].map(parse_room)

df["Toilets"] = df["Toilets"].map(parse_room)

df["Total Floors"] = df["Total Floors"].map(parse_room)

# =========================================================
# REMOVE INVALID
# =========================================================
print("🧹 Removing invalid rows...")

df = df.dropna(subset=[
    "Price",
    "Land Area"
])

# remove impossible values
df = df[

    (df["Price"] > 0)
    &
    (df["Land Area"] > 5)
    &
    (df["Land Area"] < 5000)

]

print(f"📊 After cleaning: {df.shape}")

# =========================================================
# REMOVE DUPLICATES
# =========================================================
print("🔥 Removing duplicated properties...")

df["fingerprint"] = make_property_fingerprint(df)

before = len(df)

# keep first unique property
df = df.drop_duplicates(
    subset=["fingerprint"]
)

after = len(df)

removed = before - after

print(f"✅ Removed duplicates: {removed}")

print(f"✅ Final dataset: {after}")

# =========================================================
# SAVE CLEAN DATASET
# =========================================================
df.to_csv(
    f"{OUTPUT_DIR}/clean_dataset.csv",
    index=False
)

print("💾 Saved clean dataset")

# =========================================================
# SUMMARY
# =========================================================
print("\n📊 DATA SUMMARY")

print(df[[
    "Price",
    "Land Area",
    "Bedrooms",
    "Toilets",
    "Total Floors"
]].describe())

# =========================================================
# 1. PRICE DISTRIBUTION
# =========================================================
print("\n📈 Plotting price distribution...")

plt.figure(figsize=(10,6))

sns.histplot(
    df["Price"],
    bins=30,
    kde=True
)

plt.title("Phân bố giá bất động sản")

plt.xlabel("Giá (tỷ VNĐ)")

plt.ylabel("Số lượng")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/price_distribution.png",
    dpi=300
)

plt.close()

# =========================================================
# 2. AREA DISTRIBUTION
# =========================================================
print("📈 Plotting area distribution...")

plt.figure(figsize=(10,6))

sns.histplot(
    df["Land Area"],
    bins=30,
    kde=True
)

plt.title("Phân bố diện tích bất động sản")

plt.xlabel("Diện tích (m²)")

plt.ylabel("Số lượng")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/area_distribution.png",
    dpi=300
)

plt.close()

# =========================================================
# 3. CORRELATION HEATMAP
# =========================================================
print("📈 Plotting correlation heatmap...")

features = [

    "Price",
    "Land Area",
    "Bedrooms",
    "Toilets",
    "Total Floors"
]

corr = df[features].corr()

plt.figure(figsize=(8,6))

sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/correlation_heatmap.png",
    dpi=300
)

plt.close()

# =========================================================
# 4. SCATTER PLOT
# =========================================================
print("📈 Plotting scatter plot...")

plt.figure(figsize=(10,6))

sns.scatterplot(
    x=df["Land Area"],
    y=df["Price"],
    alpha=0.7
)

plt.title("Scatter Plot giữa diện tích và giá bán")

plt.xlabel("Diện tích (m²)")

plt.ylabel("Giá (tỷ VNĐ)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/scatter_area_price.png",
    dpi=300
)

plt.close()

# =========================================================
# 5. PRICE BY BEDROOMS
# =========================================================
print("📈 Plotting price by bedrooms...")

plt.figure(figsize=(10,6))

sns.boxplot(
    x=df["Bedrooms"],
    y=df["Price"]
)

plt.title("Giá nhà theo số phòng ngủ")

plt.xlabel("Số phòng ngủ")

plt.ylabel("Giá (tỷ VNĐ)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/price_by_bedrooms.png",
    dpi=300
)

plt.close()

# =========================================================
# 6. PRICE BY FLOORS
# =========================================================
print("📈 Plotting price by floors...")

plt.figure(figsize=(10,6))

sns.boxplot(
    x=df["Total Floors"],
    y=df["Price"]
)

plt.title("Giá nhà theo số tầng")

plt.xlabel("Số tầng")

plt.ylabel("Giá (tỷ VNĐ)")

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/price_by_floors.png",
    dpi=300
)

plt.close()

# =========================================================
# 7. TOP LOCATIONS
# =========================================================
print("📈 Plotting top locations...")

top_locations = (

    df["Location"]

    .value_counts()

    .head(10)
)

plt.figure(figsize=(12,6))

top_locations.plot(kind="bar")

plt.title("Top khu vực có nhiều bất động sản")

plt.xlabel("Khu vực")

plt.ylabel("Số lượng")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/top_locations.png",
    dpi=300
)

plt.close()

# =========================================================
# 8. AVG PRICE BY LOCATION
# =========================================================
print("📈 Plotting average price by location...")

avg_price = (

    df.groupby("Location")["Price"]

    .mean()

    .sort_values(ascending=False)

    .head(10)
)

plt.figure(figsize=(12,6))

avg_price.plot(kind="bar")

plt.title("Giá trung bình theo khu vực")

plt.xlabel("Khu vực")

plt.ylabel("Giá trung bình (tỷ VNĐ)")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    f"{OUTPUT_DIR}/avg_price_location.png",
    dpi=300
)

plt.close()

# =========================================================
# 9. FEATURE IMPORTANCE
# =========================================================
print("📈 Plotting feature importance...")

try:

    model_pipeline = joblib.load(MODEL_PATH)

    model = model_pipeline.named_steps["model"]

    feature_names = [

        "Land Area",
        "Bedrooms",
        "Toilets",
        "Total Floors",

        "rooms",

        "log_area",
        "log_rooms",
        "log_floors",

        "area_per_room",
        "area_per_floor",
        "rooms_per_area",

        "district_freq",
        "district_area_median",
        "area_vs_district",

        "cluster_freq",

        "complex_score",
        "density_score",
        "luxury_score",

        "is_large_area",
        "is_small_area",

        "is_land",
        "is_hem",
        "is_mat_tien",
        "is_villa",
        "is_apartment",

        "is_so_hong"
    ]

    if hasattr(model, "feature_importances_"):

        importance = model.feature_importances_

        idx = np.argsort(importance)[-15:]

        plt.figure(figsize=(10,6))

        plt.barh(
            range(len(idx)),
            importance[idx]
        )

        plt.yticks(
            range(len(idx)),
            np.array(feature_names)[idx]
        )

        plt.title("Feature Importance")

        plt.tight_layout()

        plt.savefig(
            f"{OUTPUT_DIR}/feature_importance.png",
            dpi=300
        )

        plt.close()

        print("✅ Feature importance saved")

except Exception as e:

    print(f"❌ Feature importance error: {e}")

# =========================================================
# SAVE SUMMARY
# =========================================================
summary = {

    "raw_dataset_size": before,

    "final_dataset_size": after,

    "duplicates_removed": removed,

    "mean_price": float(df["Price"].mean()),

    "median_price": float(df["Price"].median()),

    "mean_area": float(df["Land Area"].mean()),

    "max_price": float(df["Price"].max()),

    "min_price": float(df["Price"].min())
}

summary_df = pd.DataFrame([summary])

summary_df.to_csv(
    f"{OUTPUT_DIR}/dataset_summary.csv",
    index=False
)

# =========================================================
# FINISH
# =========================================================
print("\n🎉 ALL VISUALIZATIONS COMPLETED")

print(f"📁 Output folder: {OUTPUT_DIR}")

print("\nGenerated files:")

for file in os.listdir(OUTPUT_DIR):

    print(" -", file)
