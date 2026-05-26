from pathlib import Path
import json

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


INPUT_CSV = Path("outputs/features/model_ready_density_dataset_pruned.csv")
OUTPUT_DIR = Path("outputs/models")
RESULTS_CSV = OUTPUT_DIR / "density_baseline_model_results.csv"
FEATURE_IMPORTANCE_CSV = OUTPUT_DIR / "density_extra_trees_feature_importance.csv"
METADATA_JSON = OUTPUT_DIR / "density_baseline_metadata.json"

TARGET_COL = "effective_density_kg_m3"
RANDOM_STATE = 42
TEST_SIZE = 0.2


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    return {
        "model": name,

        "train_MAE": mean_absolute_error(y_train, y_train_pred),
        "train_RMSE": rmse(y_train, y_train_pred),
        "train_R2": r2_score(y_train, y_train_pred),

        "test_MAE": mean_absolute_error(y_test, y_test_pred),
        "test_RMSE": rmse(y_test, y_test_pred),
        "test_R2": r2_score(y_test, y_test_pred),
    }, model


def main():
    if not INPUT_CSV.exists():
        print(f"Input file not found: {INPUT_CSV}")
        print("Run create_pruned_model_dataset.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded pruned model dataset:")
    print(df.shape)

    if TARGET_COL not in df.columns:
        print(f"Target column not found: {TARGET_COL}")
        return

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    numeric_cols = X.select_dtypes(include=["int64", "float64", "bool"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()

    print("\nFeature types:")
    print(f"Numeric features: {len(numeric_cols)}")
    print(f"Categorical features: {len(categorical_cols)}")
    print("Categorical columns:")
    print(categorical_cols)

    numeric_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_preprocess = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocess_scaled = ColumnTransformer(
        transformers=[
            ("num", numeric_preprocess, numeric_cols),
            ("cat", categorical_preprocess, categorical_cols),
        ]
    )

    preprocess_tree = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), numeric_cols),
            ("cat", Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]
            ), categorical_cols),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    models = [
        (
            "Mean Baseline",
            Pipeline(
                steps=[
                    ("preprocess", preprocess_scaled),
                    ("model", DummyRegressor(strategy="mean")),
                ]
            ),
        ),
        (
            "Linear Regression",
            Pipeline(
                steps=[
                    ("preprocess", preprocess_scaled),
                    ("model", LinearRegression()),
                ]
            ),
        ),
        (
            "Ridge Regression",
            Pipeline(
                steps=[
                    ("preprocess", preprocess_scaled),
                    ("model", Ridge(alpha=10.0, random_state=RANDOM_STATE)),
                ]
            ),
        ),
        (
            "Random Forest",
            Pipeline(
                steps=[
                    ("preprocess", preprocess_tree),
                    ("model", RandomForestRegressor(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )),
                ]
            ),
        ),
        (
            "Extra Trees",
            Pipeline(
                steps=[
                    ("preprocess", preprocess_tree),
                    ("model", ExtraTreesRegressor(
                        n_estimators=300,
                        max_depth=6,
                        min_samples_leaf=5,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    )),
                ]
            ),
        ),
    ]

    results = []
    fitted_models = {}

    for name, model in models:
        print(f"\nTraining: {name}")
        result, fitted_model = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
        )

        results.append(result)
        fitted_models[name] = fitted_model

        print(result)

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="test_RMSE")
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\nModel comparison:")
    print(results_df.to_string(index=False))
    print(f"\nSaved results to: {RESULTS_CSV}")

    # Feature importance from Extra Trees
    if "Extra Trees" in fitted_models:
        et_pipeline = fitted_models["Extra Trees"]
        preprocess = et_pipeline.named_steps["preprocess"]
        model = et_pipeline.named_steps["model"]

        feature_names = preprocess.get_feature_names_out()
        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values(by="importance", ascending=False)

        fi_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

        print(f"\nSaved Extra Trees feature importance to: {FEATURE_IMPORTANCE_CSV}")
        print("\nTop 25 features:")
        print(fi_df.head(25).to_string(index=False))

    metadata = {
        "input_csv": str(INPUT_CSV),
        "target_col": TARGET_COL,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "numeric_features": len(numeric_cols),
        "categorical_features": len(categorical_cols),
        "categorical_columns": categorical_cols,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "note": "PM/count columns and direct density derivation columns were removed before training.",
    }

    METADATA_JSON.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nSaved metadata to: {METADATA_JSON}")


if __name__ == "__main__":
    main()