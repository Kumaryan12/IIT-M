from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


INPUT_CSV = Path("outputs/features/temporal_density_dataset_no_env_v1.csv")

OUTPUT_DIR = Path("outputs/features")
OUTPUT_CSV = OUTPUT_DIR / "minimal_temporal_no_env_top20.csv"

IMPORTANCE_CSV = Path("outputs/models/temporal_v1/minimal_no_env_feature_importance.csv")

TARGET_COL = "effective_density_kg_m3"
GROUP_COL = "timestamp"

TOP_K = 20
RANDOM_STATE = 42

METADATA_COLS = [
    "sample_id",
    "timestamp",
    "matched_run_id",
    "osm_location_key",
    "osm_error",
    "lenses_used",
    "road_roi_lenses_used",
]

LEAKAGE_COLS = [
    "pm1_mass",
    "pm2_5_mass",
    "pm4_mass",
    "pm10_mass",
    "npm1_count",
    "npm2_5_count",
    "npm4_count",
    "npm10_count",
    "pm25_mass_kg_m3",
    "npm2_5_particles_m3",
    "total_particle_volume_m3",
    "single_particle_volume_m3",
    "effective_radius_m",
    "effective_diameter_um",
]

ENV_KEYWORDS = [
    "temperature_c",
    "relative_humidity",
    "co_ppb",
    "no2_ppb",
    "so2_ppb",
    "o3_ppb_compensated",
]


def is_env_related(col):
    return any(k in col for k in ENV_KEYWORDS)


def drop_constant_columns(X):
    drop_cols = [
        c for c in X.columns
        if X[c].nunique(dropna=False) <= 1
    ]
    return X.drop(columns=drop_cols), drop_cols


def drop_high_corr_numeric_columns(X, threshold=0.95):
    numeric_cols = X.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    if len(numeric_cols) <= 1:
        return X, []

    corr = X[numeric_cols].corr().abs()

    upper = corr.where(
        np.triu(np.ones(corr.shape), k=1).astype(bool)
    )

    to_drop = [
        c for c in upper.columns
        if any(upper[c] > threshold)
    ]

    return X.drop(columns=to_drop), to_drop


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    IMPORTANCE_CSV.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded:")
    print(df.shape)

    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    drop_cols = [
        c for c in METADATA_COLS + LEAKAGE_COLS
        if c in df.columns
    ]

    X = df.drop(columns=drop_cols + [TARGET_COL])

    env_cols = [
        c for c in X.columns
        if is_env_related(c)
    ]

    X = X.drop(columns=env_cols)

    print("\nRemoved environment-related columns:")
    print(env_cols)

    X, constant_cols = drop_constant_columns(X)
    X, corr_cols = drop_high_corr_numeric_columns(X, threshold=0.95)

    print("\nAfter cleanup:")
    print(X.shape)
    print(f"Dropped constants: {len(constant_cols)}")
    print(f"Dropped high-corr: {len(corr_cols)}")

    numeric_cols = X.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocess = ColumnTransformer([
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
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]),
            categorical_cols,
        ),
    ])

    model = ExtraTreesRegressor(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", model),
    ])

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    train_idx, test_idx = next(
        splitter.split(X, y, groups=groups)
    )

    X_train = X.iloc[train_idx]
    X_test = X.iloc[test_idx]
    y_train = y.iloc[train_idx]
    y_test = y.iloc[test_idx]

    print("\nTrain/test:")
    print(X_train.shape, X_test.shape)

    pipe.fit(X_train, y_train)

    perm = permutation_importance(
        pipe,
        X_test,
        y_test,
        n_repeats=20,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
    )

    importance_df = pd.DataFrame({
        "feature": X.columns,
        "permutation_importance_mean": perm.importances_mean,
        "permutation_importance_std": perm.importances_std,
    }).sort_values(
        by="permutation_importance_mean",
        ascending=False,
    )

    importance_df.to_csv(IMPORTANCE_CSV, index=False)

    top_features = (
        importance_df
        .head(TOP_K)["feature"]
        .tolist()
    )

    final_df = df[
        ["sample_id", "timestamp", TARGET_COL] + top_features
    ].copy()

    final_df.to_csv(OUTPUT_CSV, index=False)

    print("\nTop selected features:")
    print(top_features)

    print("\nDone.")
    print(f"Saved minimal dataset to: {OUTPUT_CSV}")
    print(f"Shape: {final_df.shape}")
    print(f"Saved feature importance to: {IMPORTANCE_CSV}")


if __name__ == "__main__":
    main()