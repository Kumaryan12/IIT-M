from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


INPUT_CSV = Path("outputs/features/minimal_temporal_no_env_top20.csv")

OUTPUT_DIR = Path("outputs/models/temporal_v1")
RESULTS_CSV = OUTPUT_DIR / "minimal_no_env_grouped_results.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "minimal_no_env_grouped_predictions.csv"

TARGET_COL = "effective_density_kg_m3"
GROUP_COL = "timestamp"

RANDOM_STATE = 42
N_SPLITS = 5


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def build_preprocess(X):
    numeric_cols = X.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    return ColumnTransformer([
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


def get_models(X):
    return {
        "RandomForest": Pipeline([
            ("preprocess", build_preprocess(X)),
            ("model", RandomForestRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "ExtraTrees": Pipeline([
            ("preprocess", build_preprocess(X)),
            ("model", ExtraTreesRegressor(
                n_estimators=500,
                max_depth=6,
                min_samples_leaf=5,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),

        "XGBoost": Pipeline([
            ("preprocess", build_preprocess(X)),
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

    print("\nLoaded minimal no-env dataset:")
    print(df.shape)

    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    X = df.drop(
        columns=[
            "sample_id",
            GROUP_COL,
            TARGET_COL,
        ]
    )

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nFinal X shape:")
    print(X.shape)

    numeric_cols = X.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print(f"\nNumeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print("Categorical columns:", categorical_cols)

    gkf = GroupKFold(n_splits=N_SPLITS)

    results = []

    pred_df = pd.DataFrame({
        "sample_id": df["sample_id"],
        "timestamp": groups,
        "ground_truth": y,
    })

    for model_name, model in get_models(X).items():
        print("\n" + "=" * 80)
        print(f"Running grouped CV: {model_name}")
        print("=" * 80)

        y_pred = cross_val_predict(
            model,
            X,
            y,
            groups=groups,
            cv=gkf,
            n_jobs=1,
        )

        metrics = evaluate(y, y_pred)
        metrics["model"] = model_name
        metrics["num_features"] = X.shape[1]

        results.append(metrics)

        pred_df[model_name] = y_pred

        print(metrics)

    results_df = pd.DataFrame(results).sort_values(
        by="RMSE"
    )

    results_df.to_csv(RESULTS_CSV, index=False)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    print("\n" + "=" * 80)
    print("MINIMAL TRUE NO-ENV GROUPED RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print(f"\nSaved results to: {RESULTS_CSV}")
    print(f"Saved predictions to: {PREDICTIONS_CSV}")

    print("\nBenchmarks:")
    print("Old true no-env best: R2 ≈ 0.172, RMSE ≈ 503.37")
    print("Manual compact no-env: R2 ≈ 0.148, RMSE ≈ 510.83")


if __name__ == "__main__":
    main()