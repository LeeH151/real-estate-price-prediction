import joblib
import numpy as np
import pandas as pd

# =========================================================
# LOAD MODEL
# =========================================================
model = joblib.load("artifacts/model.pkl")

# =========================================================
# TEST CASES
# =========================================================
samples = [

    # =====================================================
    # LOW END / NGOẠI Ô
    # =====================================================

    {
        "Location": "Hóc Môn",
        "Type of House": "Nhà hẻm",
        "Land Area": "45 m²",
        "Bedrooms": "2 phòng",
        "Toilets": "1 WC",
        "Total Floors": "1",
        "Legal Documents": "Sổ hồng"
    },

    {
        "Location": "Bình Chánh",
        "Type of House": "Nhà hẻm",
        "Land Area": "60 m²",
        "Bedrooms": "2 phòng",
        "Toilets": "2 WC",
        "Total Floors": "1",
        "Legal Documents": "Sổ hồng"
    },

    # =====================================================
    # MID RANGE
    # =====================================================

    {
        "Location": "Quận 12",
        "Type of House": "Nhà hẻm",
        "Land Area": "70 m²",
        "Bedrooms": "3 phòng",
        "Toilets": "2 WC",
        "Total Floors": "2",
        "Legal Documents": "Sổ hồng"
    },

    {
        "Location": "Tân Phú",
        "Type of House": "Nhà mặt tiền",
        "Land Area": "65 m²",
        "Bedrooms": "3 phòng",
        "Toilets": "3 WC",
        "Total Floors": "3",
        "Legal Documents": "Sổ hồng"
    },

    {
        "Location": "Bình Thạnh",
        "Type of House": "Nhà hẻm",
        "Land Area": "50 m²",
        "Bedrooms": "2 phòng",
        "Toilets": "2 WC",
        "Total Floors": "2",
        "Legal Documents": "Sổ hồng"
    },

    # =====================================================
    # APARTMENTS
    # =====================================================

    {
        "Location": "Quận 7",
        "Type of House": "Căn hộ chung cư",
        "Land Area": "65 m²",
        "Bedrooms": "2 phòng",
        "Toilets": "2 WC",
        "Total Floors": "20",
        "Legal Documents": "Hợp đồng mua bán"
    },

    {
        "Location": "Quận 2",
        "Type of House": "Căn hộ chung cư",
        "Land Area": "90 m²",
        "Bedrooms": "3 phòng",
        "Toilets": "2 WC",
        "Total Floors": "25",
        "Legal Documents": "Sổ hồng"
    },

    # =====================================================
    # HIGH END / THẢO ĐIỀN / Q1
    # =====================================================

    {
        "Location": "Phường Thảo Điền, Quận 2",
        "Type of House": "Biệt thự",
        "Land Area": "250 m²",
        "Bedrooms": "5 phòng",
        "Toilets": "6 WC",
        "Total Floors": "3",
        "Legal Documents": "Sổ hồng"
    },

    {
        "Location": "Quận 1",
        "Type of House": "Nhà mặt tiền",
        "Land Area": "80 m²",
        "Bedrooms": "4 phòng",
        "Toilets": "4 WC",
        "Total Floors": "5",
        "Legal Documents": "Sổ hồng"
    },

    {
        "Location": "Quận 1",
        "Type of House": "Nhà mặt tiền",
        "Land Area": "120 m²",
        "Bedrooms": "6 phòng",
        "Toilets": "6 WC",
        "Total Floors": "6",
        "Legal Documents": "Sổ hồng"
    },

    # =====================================================
    # EDGE CASES
    # =====================================================

    {
        "Location": "Thủ Đức",
        "Type of House": "Nhà hẻm",
        "Land Area": "35 m²",
        "Bedrooms": "1 phòng",
        "Toilets": "1 WC",
        "Total Floors": "1",
        "Legal Documents": "Sổ hồng"
    },

    {
        "Location": "Quận 10",
        "Type of House": "Nhà mặt tiền",
        "Land Area": "40 m²",
        "Bedrooms": "2 phòng",
        "Toilets": "2 WC",
        "Total Floors": "4",
        "Legal Documents": "Sổ hồng"
    },

    {
        "Location": "Quận 3",
        "Type of House": "Nhà hẻm",
        "Land Area": "55 m²",
        "Bedrooms": "3 phòng",
        "Toilets": "2 WC",
        "Total Floors": "3",
        "Legal Documents": "Sổ hồng"
    },

    {
        "Location": "Quận 7",
        "Type of House": "Căn hộ chung cư",
        "Land Area": "110 m²",
        "Bedrooms": "3 phòng",
        "Toilets": "3 WC",
        "Total Floors": "30",
        "Legal Documents": "Hợp đồng mua bán"
    },

    {
        "Location": "Phú Nhuận",
        "Type of House": "Nhà mặt tiền",
        "Land Area": "75 m²",
        "Bedrooms": "4 phòng",
        "Toilets": "3 WC",
        "Total Floors": "4",
        "Legal Documents": "Sổ hồng"
    }
]
# =========================================================
# PREDICT
# =========================================================
df = pd.DataFrame(samples)

pred_log = model.predict(df)

pred_price = np.expm1(pred_log)

# =========================================================
# RESULT
# =========================================================
print("\n" + "=" * 70)
print("🏠 REAL ESTATE PRICE PREDICTION")
print("=" * 70)

for i, (sample, price) in enumerate(
    zip(samples, pred_price),
    1
):

    print(f"\n📌 PROPERTY #{i}")
    print("-" * 50)

    for k, v in sample.items():
        print(f"{k}: {v}")

    print(f"\n💰 Predicted Price: {price:.2f} tỷ")

print("\n" + "=" * 70)