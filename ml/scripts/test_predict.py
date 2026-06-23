import joblib
import pandas as pd
import numpy as np

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("app/ml/artifacts/model.pkl")

# =========================================================
# TEST DATA (RAW INPUT - OK)
# =========================================================
samples = [
    {
        "Location": "Quận 1",
        "Type of House": "Nhà mặt tiền",
        "Land Area": "85 m²",
        "Bedrooms": "5 phòng",
        "Toilets": "5 WC",
        "Total Floors": "4",
        "Legal Documents": "Sổ hồng"
    },
    {
        "Location": "Quận 3",
        "Type of House": "Nhà hẻm",
        "Land Area": "45 m²",
        "Bedrooms": "3 phòng",
        "Toilets": "3 WC",
        "Total Floors": "3",
        "Legal Documents": "Sổ hồng"
    },
    {
        "Location": "Quận 7",
        "Type of House": "Căn hộ chung cư",
        "Land Area": "72 m²",
        "Bedrooms": "2 phòng",
        "Toilets": "2 WC",
        "Total Floors": "18",
        "Legal Documents": "Hợp đồng mua bán"
    }
]

df = pd.DataFrame(samples)

# =========================================================
# IMPORTANT FIX: PIPELINE HANDLES FEATURE ENGINEERING
# =========================================================
pred_log = model.predict(df)

# convert back to real price
pred_price = np.expm1(pred_log)

# =========================================================
# OUTPUT
# =========================================================
print("\n" + "=" * 80)
print("🏠 REAL ESTATE PRICE PREDICTION (PRO MAX FIXED)")
print("=" * 80)

for i, (sample, price) in enumerate(zip(samples, pred_price), 1):
    print(f"\n📌 PROPERTY #{i}")
    print("-" * 60)

    for k, v in sample.items():
        print(f"{k:<20}: {v}")

    print(f"\n💰 Predicted Price : {price:.2f} tỷ")

print("\n" + "=" * 80)