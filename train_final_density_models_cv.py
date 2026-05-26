from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesRegressor,
    RandomForestRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import (
    ElasticNet,
    Lasso,
    LinearRegression,
    Ridge,
)
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
    StandardScaler,
)

from xgboost import XGBRegressor


INPUT_CSV = Path(
    "outputs/features/model_density_selected_features.csv"
)

OUTPUT_DIR = Path("outputs/models")

RESULTS_CSV = OUTPUT_DIR / "final_density_model_cv_results_grouped.csv"

PREDICTIONS_CSV = (
    OUTPUT_DIR / "final_density_model_predictions_grouped.csv"
)

FEATURE_IMPORTANCE_CSV = (
    OUTPUT_DIR / "final_xgb_feature_importance_grouped.csv"
)

TARGET_COL = "effective_density_kg_m3"

RANDOM_STATE = 42
N_SPLITS = 5


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate_predictions(y_true, y_pred):

    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def main():

    if not INPUT_CSV.exists():
        print(f"Missing file: {INPUT_CSV}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded selected feature dataset:")
    print(df.shape)

    # ==========================================================
    # GROUPS
    # ==========================================================

    if "timestamp" not in df.columns:
        print("\nERROR:")
        print("timestamp column missing.")
        print(
            "Regenerate model_density_selected_features.csv "
            "with timestamp preserved."
        )
        return

    groups = df["timestamp"]

    # ==========================================================
    # FEATURES / TARGET
    # ==========================================================

    X = df.drop(columns=[TARGET_COL, "timestamp"])

    y = df[TARGET_COL]

    numeric_cols = X.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    categorical_cols = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    print("\nFeature summary:")
    print(f"Numeric: {len(numeric_cols)}")
    print(f"Categorical: {len(categorical_cols)}")

    if categorical_cols:
        print("\nCategorical columns:")
        print(categorical_cols)

    # ==========================================================
    # PREPROCESSORS
    # ==========================================================

    preprocess_scaled = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                categorical_cols,
            ),
        ]
    )

    preprocess_tree = ColumnTransformer(
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
                    (
                        "onehot",
                        OneHotEncoder(handle_unknown="ignore"),
                    ),
                ]),
                categorical_cols,
            ),
        ]
    )

    # ==========================================================
    # MODELS
    # ==========================================================

    models = {

        "LinearRegression":
        Pipeline([
            ("preprocess", preprocess_scaled),
            ("model", LinearRegression()),
        ]),

        "Ridge":
        Pipeline([
            ("preprocess", preprocess_scaled),
            ("model", Ridge(alpha=10.0)),
        ]),

        "Lasso":
        Pipeline([
            ("preprocess", preprocess_scaled),
            (
                "model",
                Lasso(
                    alpha=0.01,
                    max_iter=10000,
                ),
            ),
        ]),

        "ElasticNet":
        Pipeline([
            ("preprocess", preprocess_scaled),
            (
                "model",
                ElasticNet(
                    alpha=0.01,
                    l1_ratio=0.5,
                    max_iter=10000,
                ),
            ),
        ]),

        "RandomForest":
        Pipeline([
            ("preprocess", preprocess_tree),
            (
                "model",
                RandomForestRegressor(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),

        "ExtraTrees":
        Pipeline([
            ("preprocess", preprocess_tree),
            (
                "model",
                ExtraTreesRegressor(
                    n_estimators=400,
                    max_depth=8,
                    min_samples_leaf=3,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]),

        "XGBoost":
        Pipeline([
            ("preprocess", preprocess_tree),
            (
                "model",
                XGBRegressor(
                    n_estimators=400,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.1,
                    reg_lambda=1.0,
                    objective="reg:squarederror",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]),
    }

    # ==========================================================
    # GROUPED CV
    # ==========================================================

    gkf = GroupKFold(n_splits=N_SPLITS)

    results = []

    prediction_df = pd.DataFrame({
        "timestamp": groups,
        "ground_truth": y,
    })

    fitted_models = {}

    for name, model in models.items():

        print("\n" + "=" * 70)
        print(f"Running GROUPED CV for: {name}")
        print("=" * 70)

        y_pred = cross_val_predict(
            model,
            X,
            y,
            cv=gkf,
            groups=groups,
            n_jobs=1,
        )

        metrics = evaluate_predictions(y, y_pred)

        metrics["model"] = name

        results.append(metrics)

        prediction_df[name] = y_pred

        print(metrics)

        # ------------------------------------------------------
        # Fit on full dataset
        # For feature importance extraction
        # ------------------------------------------------------

        model.fit(X, y)

        fitted_models[name] = model

    # ==========================================================
    # RESULTS TABLE
    # ==========================================================

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        by="RMSE"
    )

    results_df.to_csv(RESULTS_CSV, index=False)

    prediction_df.to_csv(PREDICTIONS_CSV, index=False)

    print("\nFINAL GROUPED CV MODEL COMPARISON")
    print("=" * 70)

    print(results_df.to_string(index=False))

    print(f"\nSaved results to: {RESULTS_CSV}")

    print(f"Saved predictions to: {PREDICTIONS_CSV}")

    # ==========================================================
    # XGBOOST FEATURE IMPORTANCE
    # ==========================================================

    if "XGBoost" in fitted_models:

        xgb_pipeline = fitted_models["XGBoost"]

        preprocess = xgb_pipeline.named_steps["preprocess"]

        model = xgb_pipeline.named_steps["model"]

        feature_names = preprocess.get_feature_names_out()

        importances = model.feature_importances_

        fi_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values(
            by="importance",
            ascending=False,
        )

        fi_df.to_csv(
            FEATURE_IMPORTANCE_CSV,
            index=False,
        )

        print("\nTop XGBoost features:")
        print(fi_df.head(25).to_string(index=False))

        print(f"\nSaved feature importance to:")
        print(FEATURE_IMPORTANCE_CSV)

    # ==========================================================
    # FINAL NOTES
    # ==========================================================

    print("\nSCIENTIFIC NOTE")
    print("=" * 70)

    print("""
This evaluation uses GroupKFold by timestamp.

This prevents leakage where samples from the
same timestamp appear in both train and test folds.

The resulting scores are significantly more
trustworthy than random KFold evaluation.
""")

    print("\nNEXT STEPS")
    print("=" * 70)

    print("""
1. Visual-only model
2. OSM-only model
3. Environment-only model
4. Traffic-only model
5. Fusion ablation comparison
6. SHAP explainability
7. Temporal holdout testing
""")


if __name__ == "__main__":
    main()