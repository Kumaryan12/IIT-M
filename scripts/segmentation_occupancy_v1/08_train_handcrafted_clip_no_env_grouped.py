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


HANDCRAFTED_CSV = Path("outputs/features/model_density_selected_features.csv")
CLIP_CSV = Path("outputs/features/sample_clip_embeddings_v1.csv")

OUTPUT_DIR = Path("outputs/models/handcrafted_clip_v1")
RESULTS_CSV = OUTPUT_DIR / "handcrafted_clip_no_env_grouped_results.csv"
PREDICTIONS_CSV = OUTPUT_DIR / "handcrafted_clip_no_env_grouped_predictions.csv"

TARGET_COL = "effective_density_kg_m3"
GROUP_COL = "timestamp"

RANDOM_STATE = 42
N_SPLITS = 5

ENV_KEYWORDS = [
    "temperature",
    "humidity",
    "co_ppb",
    "no2_ppb",
    "so2_ppb",
    "o3",
]


def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def is_environment_feature(col):
    col_lower = col.lower()
    return any(k in col_lower for k in ENV_KEYWORDS)


def diagnose_matrix(name, X):
    print("\n" + "=" * 70)
    print(f"DIAGNOSTIC: {name}")
    print("=" * 70)

    print("Shape:", X.shape)

    print("\nDtypes:")
    print(X.dtypes.value_counts())

    numeric = X.select_dtypes(
        include=["int64", "float64", "float32", "bool"]
    )

    non_numeric = X.select_dtypes(
        exclude=["int64", "float64", "float32", "bool"]
    )

    print("\nNumeric columns:", numeric.shape[1])
    print("Non-numeric columns:", non_numeric.shape[1])

    if non_numeric.shape[1] > 0:
        print("\nNon-numeric columns:")
        print(non_numeric.columns.tolist())

    X_num = numeric.apply(pd.to_numeric, errors="coerce")

    values = X_num.to_numpy(dtype=np.float64)

    nan_count = np.isnan(values).sum()
    inf_count = np.isinf(values).sum()

    print("\nNaN count:", nan_count)
    print("Inf count:", inf_count)

    print("\nValue range:")
    print("Min:", np.nanmin(values))
    print("Max:", np.nanmax(values))
    print("Mean:", np.nanmean(values))
    print("Std:", np.nanstd(values))

    huge_cols = []

    for col in X_num.columns:
        col_values = X_num[col].to_numpy(dtype=np.float64)
        max_abs = np.nanmax(np.abs(col_values))

        if max_abs > 1e6:
            huge_cols.append((col, max_abs))

    print("\nHuge-value columns > 1e6:")
    print(huge_cols[:30])


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

    hand = pd.read_csv(HANDCRAFTED_CSV)
    clip = pd.read_csv(CLIP_CSV)

    if GROUP_COL not in hand.columns:
        print("timestamp missing in handcrafted dataset.")
        return

    print("\nLoaded:")
    print(f"Handcrafted selected: {hand.shape}")
    print(f"CLIP sample embeddings: {clip.shape}")

    clip_cols = [
        c for c in clip.columns
        if c.startswith("sample_clip_mean_")
    ]

    clip_keep = [
        "sample_id",
        "timestamp",
        "matched_run_id",
    ] + clip_cols

    # ----------------------------------------------------------
    # TEMPORARY ALIGNMENT WARNING
    # ----------------------------------------------------------
    # If handcrafted file has sample_id, merge safely.
    # If not, align by sorted timestamp + row order.
    # This is acceptable only as a temporary experiment.
    # Best final version should rebuild from master dataset.
    # ----------------------------------------------------------

    if "sample_id" in hand.columns:
        merged = hand.merge(
            clip[clip_keep],
            on=["sample_id", "timestamp"],
            how="inner",
        )
    else:
        print("\nWARNING:")
        print("sample_id missing in handcrafted dataset.")
        print("Using row-order alignment after sorting by timestamp.")
        print("For final reporting, rebuild from master dataset using sample_id.")

        clip_sorted = clip.sort_values(
            ["timestamp", "sample_id"]
        ).reset_index(drop=True)

        hand_sorted = hand.sort_values(
            ["timestamp"]
        ).reset_index(drop=True)

        if len(hand_sorted) != len(clip_sorted):
            print("Length mismatch. Cannot safely align.")
            print(hand_sorted.shape, clip_sorted.shape)
            return

        merged = pd.concat(
            [
                hand_sorted.reset_index(drop=True),
                clip_sorted[clip_cols].reset_index(drop=True),
            ],
            axis=1,
        )

    print(f"\nMerged shape: {merged.shape}")

    y = merged[TARGET_COL].astype(np.float64)
    groups = merged[GROUP_COL]

    X = merged.drop(columns=[TARGET_COL])

    meta_cols = [
        "sample_id",
        "timestamp",
        "matched_run_id",
        "clip_lenses_used",
        "num_clip_lens_frames_used",
    ]

    X = X.drop(
        columns=[c for c in meta_cols if c in X.columns],
        errors="ignore",
    )

    env_cols = [
        c for c in X.columns
        if is_environment_feature(c)
    ]

    X = X.drop(columns=env_cols)

    print("\nRemoved environment columns:")
    print(env_cols)

    clip_cols_in_X = [
        c for c in X.columns
        if c.startswith("sample_clip_mean_")
    ]

    hand_cols_in_X = [
        c for c in X.columns
        if c not in clip_cols_in_X
    ]

    # ----------------------------------------------------------
    # Numeric safety cleaning
    # ----------------------------------------------------------

    X[clip_cols_in_X] = (
        X[clip_cols_in_X]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float64)
    )

    X[hand_cols_in_X] = (
        X[hand_cols_in_X]
        .replace([np.inf, -np.inf], np.nan)
        .apply(pd.to_numeric, errors="coerce")
        .astype(np.float64)
    )

    diagnose_matrix("Full X after environment removal", X)
    diagnose_matrix("Handcrafted block", X[hand_cols_in_X])
    diagnose_matrix("CLIP block", X[clip_cols_in_X])

    print("\nFeature counts:")
    print(f"Handcrafted no-env features: {len(hand_cols_in_X)}")
    print(f"CLIP features: {len(clip_cols_in_X)}")

    duplicate_cols = X.columns[X.columns.duplicated()].tolist()
    print("\nDuplicate columns:")
    print(duplicate_cols)

    if duplicate_cols:
        print("Duplicate columns found. Aborting.")
        return

    pca_options = [4, 8, 16, 32]

    gkf = GroupKFold(n_splits=N_SPLITS)

    results = []

    pred_df = pd.DataFrame({
        "timestamp": groups,
        "ground_truth": y,
    })

    for n_pca in pca_options:
        print("\n" + "=" * 80)
        print(f"Testing handcrafted + CLIP PCA={n_pca}")
        print("=" * 80)

        preprocess = ColumnTransformer(
            transformers=[
                (
                    "handcrafted",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]),
                    hand_cols_in_X,
                ),
                (
                    "clip",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                        ("pca", PCA(
                            n_components=n_pca,
                            svd_solver="full",
                        )),
                    ]),
                    clip_cols_in_X,
                ),
            ],
            remainder="drop",
        )

        for model_name, model in build_models().items():
            print(f"\nRunning {model_name}, PCA={n_pca}")

            pipe = Pipeline([
                ("preprocess", preprocess),
                ("model", model),
            ])

            y_pred = cross_val_predict(
                pipe,
                X,
                y,
                groups=groups,
                cv=gkf,
                n_jobs=1,
            )

            metrics = evaluate(y, y_pred)
            metrics["model"] = model_name
            metrics["clip_pca_components"] = n_pca
            metrics["num_handcrafted_features"] = len(hand_cols_in_X)
            metrics["num_clip_features"] = len(clip_cols_in_X)

            results.append(metrics)

            pred_df[f"{model_name}_pca_{n_pca}"] = y_pred

            print(metrics)

    results_df = pd.DataFrame(results).sort_values(by="RMSE")

    results_df.to_csv(RESULTS_CSV, index=False)
    pred_df.to_csv(PREDICTIONS_CSV, index=False)

    print("\n" + "=" * 80)
    print("HANDCRAFTED + CLIP NO-ENV GROUPED RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print(f"\nSaved results to: {RESULTS_CSV}")
    print(f"Saved predictions to: {PREDICTIONS_CSV}")

    print("\nBenchmarks:")
    print("Old no-env best: R2 = 0.172, RMSE = 503.37")
    print("CLIP + OSM best: R2 = 0.107, RMSE = 523.02")
    print("Segmentation compact best: R2 = 0.038, RMSE = 542.70")


if __name__ == "__main__":
    main()