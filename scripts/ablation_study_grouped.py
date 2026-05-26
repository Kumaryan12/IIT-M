from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import (
    GroupKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
)

from xgboost import XGBRegressor


INPUT_CSV = Path(
    "outputs/features/model_density_selected_features.csv"
)

OUTPUT_DIR = Path("outputs/models")

RESULTS_CSV = (
    OUTPUT_DIR /
    "ablation_grouped_results.csv"
)

TARGET_COL = "effective_density_kg_m3"

RANDOM_STATE = 42


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def build_pipeline():

    model = XGBRegressor(
        n_estimators=400,
        max_depth=4,
        learning_rate=0.03,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=RANDOM_STATE,
    )

    return model


def get_feature_groups(df):

    all_cols = df.columns.tolist()

    env_cols = [
        c for c in all_cols
        if any(x in c for x in [
            "temperature",
            "humidity",
            "co_ppb",
            "no2_ppb",
            "so2_ppb",
            "o3",
        ])
    ]

    osm_cols = [
        c for c in all_cols
        if (
            "_250m" in c
            or "road_count" in c
            or "road_length" in c
            or "latitude" in c
            or "longitude" in c
        )
    ]

    traffic_cols = [
        c for c in all_cols
        if any(x in c for x in [
            "vehicle",
            "car_count",
            "truck_count",
            "motorcycle",
            "traffic_load",
            "bus_count",
            "ratio",
        ])
    ]

    visual_cols = [
        c for c in all_cols
        if any(x in c for x in [
            "road_roi",
            "brightness",
            "contrast",
            "dust",
            "edge_density",
            "haze",
        ])
    ]

    return {
        "Environment": sorted(list(set(env_cols))),
        "OSM": sorted(list(set(osm_cols))),
        "Traffic": sorted(list(set(traffic_cols))),
        "Visual": sorted(list(set(visual_cols))),
    }


def run_model(X, y, groups):

    numeric_cols = X.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[

            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                numeric_cols,
            ),

            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(
                        handle_unknown="ignore"
                    )),
                ]),
                categorical_cols,
            ),
        ]
    )

    pipeline = Pipeline([
        ("preprocess", preprocess),
        ("model", build_pipeline()),
    ])

    gkf = GroupKFold(n_splits=5)

    y_pred = cross_val_predict(
        pipeline,
        X,
        y,
        groups=groups,
        cv=gkf,
        n_jobs=1,
    )

    return evaluate(y, y_pred)


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded dataset:")
    print(df.shape)

    groups = df["timestamp"]

    y = df[TARGET_COL]

    X_all = df.drop(columns=[
        TARGET_COL,
        "timestamp",
    ])

    feature_groups = get_feature_groups(X_all)

    results = []

    # ======================================================
    # INDIVIDUAL MODALITIES
    # ======================================================

    for group_name, cols in feature_groups.items():

        print("\n" + "=" * 70)
        print(f"Running: {group_name}")
        print("=" * 70)

        X = X_all[cols].copy()

        metrics = run_model(
            X,
            y,
            groups,
        )

        metrics["modality"] = group_name
        metrics["num_features"] = len(cols)

        results.append(metrics)

        print(metrics)

    # ======================================================
    # FUSION MODELS
    # ======================================================

    fusion_configs = {

        "Visual + Traffic":
        (
            feature_groups["Visual"]
            + feature_groups["Traffic"]
        ),

        "Visual + OSM":
        (
            feature_groups["Visual"]
            + feature_groups["OSM"]
        ),

        "Traffic + Environment":
        (
            feature_groups["Traffic"]
            + feature_groups["Environment"]
        ),

        "All Fusion":
        X_all.columns.tolist(),
    }

    for fusion_name, cols in fusion_configs.items():

        cols = sorted(list(set(cols)))

        print("\n" + "=" * 70)
        print(f"Running: {fusion_name}")
        print("=" * 70)

        X = X_all[cols].copy()

        metrics = run_model(
            X,
            y,
            groups,
        )

        metrics["modality"] = fusion_name
        metrics["num_features"] = len(cols)

        results.append(metrics)

        print(metrics)

    # ======================================================
    # SAVE RESULTS
    # ======================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="R2",
        ascending=False,
    )

    results_df.to_csv(
        RESULTS_CSV,
        index=False,
    )

    print("\n")
    print("=" * 80)
    print("FINAL ABLATION RESULTS")
    print("=" * 80)

    print(results_df.to_string(index=False))

    print(f"\nSaved to: {RESULTS_CSV}")


if __name__ == "__main__":
    main()