from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor


MASTER_CSV = Path("outputs/features/master_sensor_osm_visual_roi_density_dataset.csv")
CLIP_CSV = Path("outputs/features/sample_clip_embeddings_v1.csv")

OUTPUT_DIR = Path("outputs/models/clip_osm_v1")
RESULTS_CSV = OUTPUT_DIR / "clip_osm_no_env_grouped_results.csv"

TARGET_COL = "effective_density_kg_m3"
GROUP_COL = "timestamp"

RANDOM_STATE = 42
N_SPLITS = 5


OSM_COLS = [
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
]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def build_models():
    return {
        "Ridge": Ridge(alpha=10.0),

        "RandomForest": RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=400,
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBRegressor(
            n_estimators=300,
            max_depth=3,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.5,
            reg_lambda=2.0,
            objective="reg:squarederror",
            random_state=RANDOM_STATE,
        ),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    master = pd.read_csv(MASTER_CSV)
    clip = pd.read_csv(CLIP_CSV)

    merged = master.merge(
        clip,
        on=["sample_id", "timestamp", "matched_run_id"],
        how="inner",
    )

    print("\nMerged dataset:")
    print(merged.shape)

    clip_cols = [
        c for c in merged.columns
        if c.startswith("sample_clip_mean_")
    ]

    osm_cols = [
        c for c in OSM_COLS
        if c in merged.columns
    ]

    print(f"CLIP cols: {len(clip_cols)}")
    print(f"OSM cols: {len(osm_cols)}")

    y = merged[TARGET_COL]
    groups = merged[GROUP_COL]

    X_raw = merged[osm_cols + clip_cols].copy()

    results = []

    pca_options = [8, 16, 32, 64]

    gkf = GroupKFold(n_splits=N_SPLITS)

    for n_pca in pca_options:
        print("\n" + "=" * 80)
        print(f"Testing PCA components: {n_pca}")
        print("=" * 80)

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "osm",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    osm_cols,
                ),
                (
                    "clip",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(n_components=n_pca, random_state=RANDOM_STATE)),
                    ]),
                    clip_cols,
                ),
            ]
        )

        for model_name, model in build_models().items():
            print(f"\nRunning {model_name} with PCA={n_pca}")

            pipe = Pipeline([
                ("preprocess", preprocess),
                ("model", model),
            ])

            y_pred = cross_val_predict(
                pipe,
                X_raw,
                y,
                groups=groups,
                cv=gkf,
                n_jobs=1,
            )

            metrics = evaluate(y, y_pred)
            metrics["model"] = model_name
            metrics["pca_components"] = n_pca
            metrics["num_osm_features"] = len(osm_cols)
            metrics["num_clip_features"] = len(clip_cols)

            results.append(metrics)

            print(metrics)

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 80)
    print("CLIP + OSM NO-ENV GROUPED RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print(f"\nSaved to: {RESULTS_CSV}")

    print("\nBenchmarks:")
    print("Old no-env best: R2 = 0.172, RMSE = 503.37")
    print("Compact segmentation no-env best: R2 = 0.038, RMSE = 542.70")


if __name__ == "__main__":
    main()