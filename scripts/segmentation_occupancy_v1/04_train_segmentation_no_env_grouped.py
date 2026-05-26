from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor


INPUT_CSV = Path("outputs/features/segmentation_density_dataset_v1.csv")
OUTPUT_DIR = Path("outputs/models/segmentation_occupancy_v1")

RESULTS_CSV = OUTPUT_DIR / "compact_segmentation_no_env_grouped_results.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "compact_segmentation_no_env_grouped_predictions.csv"

TARGET_COL = "effective_density_kg_m3"
GROUP_COL = "timestamp"

RANDOM_STATE = 42
N_SPLITS = 5


COMPACT_FEATURES = [
    # OSM / location context
    "latitude",
    "longitude",
    "osm_latitude_rounded",
    "osm_longitude_rounded",
    "commercial_count_250m",
    "marketplace_count_250m",
    "parking_count_250m",
    "restaurant_count_250m",
    "road_segment_count_250m",
    "total_road_length_250m",
    "primary_road_count_250m",
    "secondary_road_count_250m",
    "residential_road_count_250m",

    # Existing visual road ROI features
    "road_roi_brightness_mean_mean",
    "road_roi_contrast_std_mean",
    "road_roi_edge_density_mean",
    "road_roi_haze_score_mean",
    "road_roi_road_dust_score_mean",
    "road_roi_visual_dust_score_mean",

    # Compact segmentation occupancy features
    "seg_vehicle_occupancy_ratio_mean",
    "seg_vehicle_occupancy_ratio_max",
    "seg_total_vehicle_instances_mean",
    "seg_total_vehicle_instances_max",
    "seg_average_vehicle_confidence_mean",
    "seg_heavy_vehicle_occupancy_ratio_mean",
    "seg_two_wheeler_occupancy_ratio_mean",
    "seg_car_occupancy_ratio_mean",
    "seg_heavy_vehicle_area_share_mean",
    "seg_two_wheeler_area_share_mean",
    "seg_car_area_share_mean",

    # Lens-wise compact segmentation features
    "lens1_seg_vehicle_occupancy_ratio",
    "lens4_seg_vehicle_occupancy_ratio",
    "lens6_seg_vehicle_occupancy_ratio",
    "lens1_seg_heavy_vehicle_occupancy_ratio",
    "lens4_seg_heavy_vehicle_occupancy_ratio",
    "lens6_seg_heavy_vehicle_occupancy_ratio",
    "lens1_seg_two_wheeler_occupancy_ratio",
    "lens4_seg_two_wheeler_occupancy_ratio",
    "lens6_seg_two_wheeler_occupancy_ratio",
]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def build_preprocess(X, scaled=False):
    numeric_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    numeric_steps = [("imputer", SimpleImputer(strategy="median"))]

    if scaled:
        numeric_steps.append(("scaler", StandardScaler()))

    return ColumnTransformer([
        ("num", Pipeline(numeric_steps), numeric_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), categorical_cols),
    ])


def get_models(X):
    return {
        "Ridge": Pipeline([
            ("preprocess", build_preprocess(X, scaled=True)),
            ("model", Ridge(alpha=10.0)),
        ]),

        "RandomForest": Pipeline([
            ("preprocess", build_preprocess(X, scaled=False)),
            ("model", RandomForestRegressor(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "ExtraTrees": Pipeline([
            ("preprocess", build_preprocess(X, scaled=False)),
            ("model", ExtraTreesRegressor(
                n_estimators=400,
                max_depth=6,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "XGBoost": Pipeline([
            ("preprocess", build_preprocess(X, scaled=False)),
            ("model", XGBRegressor(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.5,
                reg_lambda=2.0,
                objective="reg:squarederror",
                random_state=RANDOM_STATE,
            )),
        ]),
    }


def main():
    if not INPUT_CSV.exists():
        print(f"Missing file: {INPUT_CSV}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded segmentation density dataset:")
    print(df.shape)

    available_features = [
        c for c in COMPACT_FEATURES
        if c in df.columns
    ]

    missing_features = [
        c for c in COMPACT_FEATURES
        if c not in df.columns
    ]

    print(f"\nCompact features requested: {len(COMPACT_FEATURES)}")
    print(f"Available compact features: {len(available_features)}")

    if missing_features:
        print("\nMissing compact features:")
        print(missing_features)

    X = df[available_features].copy()
    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    # Drop constants
    constant_cols = [
        c for c in X.columns
        if X[c].nunique(dropna=False) <= 1
    ]

    X = X.drop(columns=constant_cols)

    print("\nDropped constant columns:")
    print(constant_cols)

    print("\nFinal compact X shape:")
    print(X.shape)

    print("\nBenchmark to beat:")
    print("Old no-environment best: ExtraTrees R2 = 0.172, RMSE = 503.37")
    print("Raw segmentation no-env best: XGBoost R2 = 0.034, RMSE = 543.79")

    gkf = GroupKFold(n_splits=N_SPLITS)

    results = []
    pred_df = pd.DataFrame({
        "timestamp": groups,
        "ground_truth": y,
    })

    for name, model in get_models(X).items():
        print("\n" + "=" * 70)
        print(f"Running compact grouped CV: {name}")
        print("=" * 70)

        y_pred = cross_val_predict(
            model,
            X,
            y,
            groups=groups,
            cv=gkf,
            n_jobs=1,
        )

        metrics = evaluate(y, y_pred)
        metrics["model"] = name
        metrics["num_features"] = X.shape[1]

        results.append(metrics)
        pred_df[name] = y_pred

        print(metrics)

    results_df = pd.DataFrame(results).sort_values(by="RMSE")

    results_df.to_csv(RESULTS_CSV, index=False)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    print("\n" + "=" * 80)
    print("COMPACT SEGMENTATION NO-ENV GROUPED RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print(f"\nSaved results to: {RESULTS_CSV}")
    print(f"Saved predictions to: {PREDICTIONS_CSV}")


if __name__ == "__main__":
    main()