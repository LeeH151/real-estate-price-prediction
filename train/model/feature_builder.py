import logging
import numpy as np
import pandas as pd

from sklearn.base import (
    BaseEstimator,
    TransformerMixin
)

from model.utils import (
    parse_area,
    parse_room,
    safe_divide,
    normalize_text,
    normalize_location,
    extract_dimensions
)

logger = logging.getLogger(__name__)

EPS = 1e-8


# =========================================================
# SAFE NUMERIC
# =========================================================
def safe_numeric(series):

    return pd.to_numeric(
        series,
        errors="coerce"
    ).astype(np.float32)


# =========================================================
# FEATURE BUILDER
# =========================================================
class FeatureBuilder(
    BaseEstimator,
    TransformerMixin
):

    def __init__(self, debug=False):

        self.debug = debug

        # =================================================
        # GLOBAL STATS
        # =================================================
        self.area_median = 80.0
        self.area_clip_high = 500.0
        # =================================================
        # DISTRICT MAPS
        # =================================================
        self.district_freq_map = {}
        self.district_area_median_map = {}
        self.district_strength_map = {}
        self.district_centroid_map = {}
        # =================================================
        # FINAL FEATURE SCHEMA
        # =================================================
        self.ward_freq_map={}
        self.default_ward_freq = 0.001
        self.feature_columns_ = [

            # =================================================
            # RAW
            # =================================================
            "Land Area",
            "Bedrooms",
            "Toilets",
            "Total Floors",

            # =================================================
            # CORE
            # =================================================
            "rooms",
            "ward_freq",
            # =================================================
            # LOG
            # =================================================
            "log_area",
            "log_rooms",
            "log_floors",

            # =================================================
            # RATIOS
            # =================================================
            "area_per_room",
            "area_per_floor",
            "rooms_per_area",
            # DIMENSIONS
            "width",
            "length",
            "frontage_ratio",
            "width_ratio",
            "width_x_area",
            "is_wide_house",
            "is_long_house",
            "square_ratio",
            "is_square_house",
            "big_frontage",
            # =================================================
            # DISTRICT
            # =================================================
            "district_freq",
            "district_area_median",
            "district_strength",
            "district_score",
            "district_centroid_score",
            "area_vs_district",

            # =================================================
            # CLUSTER
            # =================================================
            "cluster_freq",

            # =================================================
            # ADVANCED
            # =================================================
            "complex_score",
            "density_score",
            "luxury_score",

            # =================================================
            # FLAGS
            # =================================================
            "is_large_area",
            "is_small_area",
            "shape_score",
            # =================================================
            # HOUSE TYPE
            # =================================================
            "is_land",
            "is_hem",
            "is_mat_tien",
            "is_villa",
            "is_apartment",

            # =================================================
            # LEGAL
            # =================================================
            "is_so_hong",

            # =================================================
            # MISSING FLAGS
            # =================================================
            "area_missing",
            "bedroom_missing",
            "toilet_missing",
            "floors_missing"
        ]

    # =====================================================
    # DISTRICT EXTRACTOR
    # =====================================================
    def _extract_district(self, df):

        if "Location" not in df.columns:

            return pd.Series(
                "unknown",
                index=df.index
            )

        text = (

            df["Location"]

            .fillna("")

            .astype(str)

            .map(normalize_location)
        )
        district = text.str.extract(
        r"(quan \d+|huyen [a-z ]+|binh thanh|go vap|tan binh|tan phu|phu nhuan|thu duc|binh chanh|hoc mon|cu chi|nha be|can gio|binh tan)"
        )[0]
        district = district.fillna("unknown")

        district = (
            district
            .str.replace(" ", "_")
        )

        return district

        '''mapping = {

            "quan_1": r"quan 1",
            "quan_2": r"quan 2",
            "quan_3": r"quan 3",
            "quan_4": r"quan 4",
            "quan_5": r"quan 5",
            "quan_6": r"quan 6",
            "quan_7": r"quan 7",
            "quan_8": r"quan 8",
            "quan_9": r"quan 9",
            "quan_10": r"quan 10",
            "quan_11": r"quan 11",
            "quan_12": r"quan 12",

            "tan_binh": r"tan binh",
            "tan_phu": r"tan phu",
            "binh_thanh": r"binh thanh",
            "phu_nhuan": r"phu nhuan",
            "go_vap": r"go vap",
            "thu_duc": r"thu duc",

            "binh_tan": r"binh tan",
            "hoc_mon": r"hoc mon",
            "cu_chi": r"cu chi",
            "nha_be": r"nha be",
            "binh_chanh": r"binh chanh"
        }

        out = pd.Series(
            "unknown",
            index=df.index
        )

        for k, pattern in mapping.items():

            mask = text.str.contains(
                pattern,
                regex=True,
                na=False
            )

            out.loc[mask] = k

        return out'''

    # =====================================================
    # FIT
    # =====================================================
    def fit(self, X, y=None):

        df = X.copy()

        # =================================================
        # REQUIRED COLUMNS
        # =================================================
        required = [

            "Location",
            "Land Area"
        ]

        for c in required:

            if c not in df.columns:
                df[c] = np.nan

        # =================================================
        # DISTRICT
        # =================================================
        df["district"] = (
            self._extract_district(df)
        )
        df["ward"] = (
            df["Location"]
            .fillna("")
            .astype(str)
            .map(normalize_location)
            .str.extract(
                r"(phuong [a-z0-9 ]+|xa [a-z0-9 ]+)"
            )
            .fillna("unknown")
        )
        ward_count = df["ward"].value_counts()

        self.ward_freq_map = (
            ward_count / len(df)
        ).to_dict()

        self.default_ward_freq = 1.0 / max(len(self.ward_freq_map), 1)
        # =================================================
        # AREA
        # =================================================
        df["area"] = safe_numeric(
            df["Land Area"].map(parse_area)
        )

        area_values = (
            df["area"]
            .dropna()
        )

        # =================================================
        # AREA STATS
        # =================================================
        if len(area_values) > 0:

            self.area_median = float(

                np.nanmedian(
                    area_values
                )
            )

            self.area_clip_high = float(

                np.nanpercentile(
                    area_values,
                    99
                )
            )

        self.area_clip_high = float(

            np.clip(
                self.area_clip_high,
                100,
                5000
            )
        )

        # =================================================
        # DISTRICT FREQUENCY
        # =================================================
        freq = (
            df["district"]
            .value_counts()
        )

        total = max(len(df), 1)

        self.district_freq_map = (
            freq / total
        ).to_dict()

        # =================================================
        # DISTRICT AREA MEDIAN
        # =================================================
        grouped = df.groupby("district")["area"]

        for district, values in grouped:

            values = values.dropna()

            if len(values) == 0:
                continue

            cnt = len(values)

            median_area = values.median()

            self.district_area_median_map[district] = float(
                (
                    median_area * cnt
                    + self.area_median * 5
                )
                /(cnt+5)
            )

            self.district_strength_map[district] = float(cnt)

            self.district_centroid_map[district] = float(
                np.median(values)
            )

        return self

    # =====================================================
    # TRANSFORM
    # =====================================================
    def transform(self, X):

        df = X.copy().reset_index(drop=True)

        # =================================================
        # REQUIRED COLUMNS
        # =================================================
        required = [

            "Land Area",
            "Bedrooms",
            "Toilets",
            "Total Floors",

            "Type of House",
            "Legal Documents",
            "Location"
        ]

        for c in required:

            if c not in df.columns:
                df[c] = np.nan

        # =================================================
        # MISSING FLAGS
        # =================================================
        df["area_missing"] = (
            df["Land Area"].isna()
        ).astype(np.float32)

        df["bedroom_missing"] = (
            df["Bedrooms"].isna()
        ).astype(np.float32)

        df["toilet_missing"] = (
            df["Toilets"].isna()
        ).astype(np.float32)

        df["floors_missing"] = (
            df["Total Floors"].isna()
        ).astype(np.float32)

        # =================================================
        # DISTRICT
        # =================================================
        df["district"] = (
            self._extract_district(df)
        )

        # =================================================
        # PARSE NUMERIC
        # =================================================
        raw_area = df["Land Area"].copy()

        df["Land Area"] = safe_numeric(
            raw_area.map(parse_area)
        )

        widths = []
        lengths = []

        '''for x in raw_area:

            w, l = extract_dimensions(x)

            widths.append(w)
            lengths.append(l)
        raw_area = df["Land Area"].copy()'''

        widths, lengths = zip(*[
            extract_dimensions(x if isinstance(x, str) else "")
            for x in df["Land Area"].fillna("")
        ])

        df["width"] = (
            pd.Series(widths)
            .apply(pd.to_numeric, errors="coerce")
            .fillna(5)
            .clip(2, 50)
            .values
        )
        lengths = pd.to_numeric(lengths, errors="coerce")
        lengths = pd.Series(lengths, index=df.index)

        df["length"] = (
            lengths
            .fillna(df["Land Area"].fillna(self.area_median) / 5)
            .clip(2, 200)
            .astype(np.float32)
        )
        df["width_x_area"] = np.log1p(df["width"]) * np.log1p(df["Land Area"])
        df["is_wide_house"] = (
            df["width"] >= 5
        ).astype(int)

        df["is_long_house"] = (
            df["length"] >= 15
        ).astype(int)
        df["width_ratio"] = (
            df["width"]
            /
            (df["length"] + EPS)
        )
        df["Bedrooms"] = safe_numeric(
            df["Bedrooms"].map(parse_room)
        )

        df["Toilets"] = safe_numeric(
            df["Toilets"].map(parse_room)
        )

        df["Total Floors"] = safe_numeric(
            df["Total Floors"].map(parse_room)
        )

        bedrooms = df["Bedrooms"].fillna(0)
        toilets = df["Toilets"].fillna(0)

        rooms = bedrooms + 0.5 * toilets
        df["rooms"] = rooms
        # =================================================
        # FILL MISSING
        # =================================================
        df["Land Area"] = (
            df["Land Area"]
            .fillna(self.area_median)
        )

        df["Bedrooms"] = (
            df["Bedrooms"]
            .fillna(0)
        )

        df["Toilets"] = (
            df["Toilets"]
            .fillna(0)
        )

        df["Total Floors"] = (
            df["Total Floors"]
            .fillna(0)
        )
        df["ward"] = (
            df["Location"]
            .fillna("")
            .astype(str)
            .map(normalize_location)
            .str.extract(
                r"(phuong [a-z0-9 ]+|xa [a-z0-9 ]+)"
            )
            .fillna("unknown")
        )
        df["ward_freq"] = (
            df["ward"]
            .map(self.ward_freq_map)
            .fillna(self.default_ward_freq)
        )
        # =================================================
        # CLIP
        # =================================================
        df["Land Area"] = (
            df["Land Area"]
            .clip(5, self.area_clip_high)
        )

        for c in [

            "Bedrooms",
            "Toilets",
            "Total Floors"
        ]:

            df[c] = (
                df[c]
                .clip(0, 50)
            )

        # =================================================
        # LOG FEATURES
        # =================================================
        df["log_area"] = np.log1p(
            df["Land Area"]
        )

        df["log_rooms"] = np.log1p(rooms)
        df["log_floors"] = np.log1p(
            df["Total Floors"]
        )

        # =================================================
        # RATIO FEATURES
        # =================================================
        df["area_per_room"] = safe_divide(

            df["Land Area"],

            rooms + EPS
        )

        df["area_per_floor"] = safe_divide(

            df["Land Area"],

            df["Total Floors"] + EPS
        )

        df["rooms_per_area"] = safe_divide(

            rooms,

            df["Land Area"] + EPS
        )
        df["frontage_ratio"] = safe_divide(
            df["width"],
            df["length"] + EPS
        )
        df["width_ratio"] = safe_divide(
            df["width"],
            df["length"] + EPS
        )

        df["room_density"] = safe_divide(
            rooms,
            df["Land Area"]
        )

        df["floor_density"] = safe_divide(
            df["Total Floors"],
            df["Land Area"]
        )
        df["shape_score"] = (
            np.log1p(df["width"]) *
            np.log1p(df["length"])
        )
        df["square_ratio"] = safe_divide(
            np.minimum(df["width"], df["length"]),
            np.maximum(df["width"], df["length"])
        )

        df["is_square_house"] = (
            df["square_ratio"] > 0.7
        ).astype(np.float32)

        df["big_frontage"] = (
            df["width"] >= 6
        ).astype(np.float32)
        # =================================================
        # DISTRICT FEATURES
        # =================================================
        default_freq = (

            np.mean(
                list(
                    self.district_freq_map.values()
                )
            )

            if self.district_freq_map
            else 0.001
        )

        default_strength = (

            np.mean(
                list(
                    self.district_strength_map.values()
                )
            )

            if self.district_strength_map
            else 1.0
        )

        df["district_freq"] = (

            df["district"]

            .map(self.district_freq_map)

            .fillna(default_freq)
        )

        df["district_area_median"] = (

            df["district"]

            .map(self.district_area_median_map)

            .fillna(self.area_median)
        )
        df["district_area_log"] = np.log1p(df["district_area_median"])
        df["price_anchor"] = df["district_area_median"] * np.log1p(df["Land Area"])        
        df["district_strength"] = (

            df["district"]

            .map(self.district_strength_map)

            .fillna(default_strength)
        )

        df["district_score"] = (
            np.log1p(df["district_freq"]) *
            np.log1p(df["district_strength"] + 1)
        )
        default_centroid = self.area_median
        centroid = (
            df["district"]
            .map(self.district_centroid_map)
            .fillna(default_centroid)
        )

        df["district_centroid_score"] = safe_divide(
            df["Land Area"],
            centroid + EPS
        )

        df["area_vs_district"] = safe_divide(

            df["Land Area"],

            df["district_area_median"] + EPS
        )

        # =================================================
        # CLUSTER PLACEHOLDER
        # =================================================
        if "cluster_freq" not in df.columns:
            df["cluster_freq"] = 1.0


                # =================================================
        # HOUSE TYPE
        # =================================================
        house = (

            df["Type of House"]

            .fillna("")

            .astype(str)

            .map(normalize_text)
        )

        df["is_land"] = (

            house.str.contains(
                r"dat",
                regex=True
            )

        ).astype(np.float32)

        df["is_villa"] = (

            house.str.contains(
                r"villa|biet thu",
                regex=True
            )

        ).astype(np.float32)

        df["is_mat_tien"] = (

            house.str.contains(
                r"mat tien",
                regex=True
            )

        ).astype(np.float32)

        df["is_hem"] = (

            house.str.contains(
                r"hem|ngo|ngach",
                regex=True
            )

        ).astype(np.float32)

        df["is_apartment"] = (

            house.str.contains(
                r"chung cu|can ho",
                regex=True
            )

        ).astype(np.float32)
        df["is_house"] = (1.0 - df["is_apartment"]).astype(np.float32)
        df["is_high_rise"] = (df["Total Floors"] >= 10).astype(np.float32)

        # =================================================
        # ADVANCED FEATURES
        # =================================================
        df["complex_score"] = (

            df["log_area"]

            * np.sqrt(rooms + 1)

            * (df["log_floors"] + 1)
        )

        df["density_score"] = safe_divide(
            rooms * np.log1p(df["Total Floors"]* df["is_apartment"] + 1),
            (df["Land Area"] + EPS)
        )
        df["luxury_score"] = (

            df["log_area"]

            * np.sqrt(
                rooms + 1
            )

            * np.log1p(
                df["Total Floors"]+1
            )

        )
        # =================================================
        # AREA FLAGS
        # =================================================
        df["is_large_area"] = (
            df["Land Area"] > 120
        ).astype(np.float32)

        df["is_small_area"] = (
            df["Land Area"] < 25
        ).astype(np.float32)

        # =================================================
        # LEGAL
        # =================================================
        legal = (

            df["Legal Documents"]

            .fillna("")

            .astype(str)

            .map(normalize_text)
        )

        df["is_so_hong"] = (

            legal.str.contains(
                r"so hong",
                regex=True
            )

        ).astype(np.float32)

        # =================================================
        # ENSURE SCHEMA
        # =================================================
        for col in self.feature_columns_:

            if col not in df.columns:
                df[col] = 0.0

        # =================================================
        # FINAL OUTPUT
        # =================================================
        out = (

            df[self.feature_columns_]

            .replace(
                [np.inf, -np.inf],
                0
            )

            .fillna(0)

            .astype(np.float32)
        )

        if self.debug:

            logger.info(
                f"✅ FeatureBuilder output: {out.shape}"
            )

            logger.info(
                f"📌 Columns: {list(out.columns)}"
            )

        return out