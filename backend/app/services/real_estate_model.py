import pandas as pd
import numpy as np
import re
import joblib

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error


class RealEstatePredictor:

    def __init__(self):
        self.model = None
        self.district_price_map = {}
        self.global_median_price = 0.0  # FIX: global fallback


    # ========================
    # LOAD DATA
    # ========================
    def load_data(self, path):
        df = pd.read_csv(path)
        print(f"📊 Raw shape: {df.shape}")
        return df


    # ========================
    # CLEAN DATA
    # ========================
    def clean_data(self, df):

        df = df.copy()

        parts = df['Location'].astype(str).str.split(', ')
        df['ward'] = parts.str[0]
        df['district'] = parts.str[1]

        df.drop(columns=['Location'], inplace=True)

        df['district'] = df['district'].str.replace(
            r'TP\.\s*Thủ Đức\s*-\s*', '', regex=True
        ).str.strip()

        df['Price'] = df['Price'].apply(self._normalize_price)
        df['Land Area'] = df['Land Area'].apply(self._normalize_area)

        for col in ['Bedrooms', 'Toilets', 'Total Floors']:
            df[col] = df[col].apply(self._extract_number)

        df = df[(df['Price'] > 0.5) & (df['Price'] < 60)]
        df = df[(df['Land Area'] > 10) & (df['Land Area'] < 2000)]

        df['is_central'] = df['district'].isin(
            ['Quận 1', 'Quận 3', 'Quận 5', 'Quận 10', 'Phú Nhuận']
        ).astype(int)

        df['rooms_total'] = df['Bedrooms'].fillna(0) + df['Toilets'].fillna(0)

        df['density'] = df['rooms_total'] / df['Land Area']
        df['density'] = df['density'].replace([np.inf, -np.inf], np.nan).fillna(0)

        df['Price_log'] = np.log1p(df['Price'])

        print(f"📊 Clean shape: {df.shape}")
        return df


    # ========================
    # HELPERS
    # ========================
    def _normalize_price(self, x):
        if pd.isna(x):
            return np.nan

        text = str(x).lower()
        nums = re.findall(r'[\d.,]+', text)

        if not nums:
            return np.nan

        raw = nums[0].replace(',', '')

        try:
            num = float(raw)
        except:
            return np.nan

        if 'tỷ' in text:
            return num
        elif 'triệu' in text:
            return num / 1000

        return num / 1e9


    def _normalize_area(self, x):
        nums = re.findall(r'[\d,.]+', str(x))
        return float(nums[0].replace('.', '').replace(',', '.')) if nums else np.nan


    def _extract_number(self, x):
        nums = re.findall(r'\d+', str(x))
        return float(nums[0]) if nums else np.nan


    # ========================
    # FEATURES
    # ========================
    def prepare_features(self, df):

        X = df[[
            'district', 'Type of House', 'Land Area',
            'Bedrooms', 'Toilets', 'Total Floors',
            'Legal Documents', 'is_central', 'density'
        ]].copy()

        y = df['Price_log']

        return X, y


    # ========================
    # TRAIN
    # ========================
    def train(self, X, y):

        X = X.reset_index(drop=True)
        y = y.reset_index(drop=True)

        bins = pd.qcut(np.expm1(y), q=5, labels=False, duplicates='drop')

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=bins, random_state=42
        )

        # =========================
        # DISTRICT PRICE MAP (TRAIN ONLY)
        # =========================
        price_train = np.expm1(y_train)

        df_train = X_train.copy()
        df_train["Price"] = price_train

        self.district_price_map = (
            df_train.groupby("district")["Price"]
            .median()
            .to_dict()
        )

        self.global_median_price = float(np.median(price_train))  # FIX

        # =========================
        # SAFE FEATURE ADDITION
        # =========================
        def map_district_price(x):
            return self.district_price_map.get(x, self.global_median_price)

        X_train = X_train.copy()
        X_test = X_test.copy()

        X_train["district_price"] = X_train["district"].apply(map_district_price)
        X_test["district_price"] = X_test["district"].apply(map_district_price)

        # =========================
        # PIPELINE
        # =========================
        num_cols = [
            'Land Area', 'Bedrooms', 'Toilets',
            'Total Floors', 'is_central', 'density', 'district_price'
        ]

        cat_cols = [
            'district', 'Type of House', 'Legal Documents'
        ]

        preprocessor = ColumnTransformer([
            ('num', SimpleImputer(strategy='median'), num_cols),
            ('cat', Pipeline([
                ('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                ('ohe', OneHotEncoder(handle_unknown='ignore'))
            ]), cat_cols)
        ])

        self.model = Pipeline([
            ('prep', preprocessor),
            ('rf', RandomForestRegressor(
                n_estimators=300,     # FIX: stronger model
                max_depth=15,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            ))
        ])

        self.model.fit(X_train, y_train)

        # =========================
        # EVALUATION
        # =========================
        pred = self.model.predict(X_test)

        rmse = np.sqrt(mean_squared_error(y_test, pred))
        print(f"✅ RMSE (log): {rmse:.4f}")

        rmse_real = np.sqrt(
            mean_squared_error(np.expm1(y_test), np.expm1(pred))
        )
        print(f"✅ RMSE (real): {rmse_real:.4f}")


    # ========================
    # PREDICT (FIXED)
    # ========================
    def predict(self, input_data):

        if isinstance(input_data, dict):
            X = pd.DataFrame([input_data])
        else:
            X = pd.DataFrame(input_data)

        # SAFE MAP (vectorized, no lambda)
        def map_district_price(x):
            return self.district_price_map.get(x, self.global_median_price)

        X = X.copy()
        X["district_price"] = X["district"].apply(map_district_price)

        pred_log = self.model.predict(X)
        pred_log = np.asarray(pred_log).ravel()

        if len(pred_log) == 0:
            return 0.0

        result = float(np.expm1(pred_log[0]))

        if not np.isfinite(result):
            return 0.0

        return max(result, 0.0)


    # ========================
    # SAVE / LOAD
    # ========================
    def save(self, path="app/models/model.pkl"):
        joblib.dump(self, path)
        print(f"💾 Saved: {path}")


    @staticmethod
    def load(path="app/models/model.pkl"):
        obj = joblib.load(path)
        print("✅ Model loaded")
        return obj