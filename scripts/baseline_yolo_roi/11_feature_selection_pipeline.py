from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


INPUT_CSV = Path("outputs/features/model_ready_density_dataset_pruned.csv")

OUTPUT_DIR = Path("outputs/features")

CORR_PRUNED_CSV = OUTPUT_DIR / "model_density_corr_pruned.csv"
FINAL_SELECTED_CSV = OUTPUT_DIR / "model_density_selected_features.csv"

CORR_DROPPED_CSV = OUTPUT_DIR / "dropped_high_corr_features.csv"
LOW_VARIANCE_DROPPED_CSV = OUTPUT_DIR / "dropped_low_variance_features.csv"

FEATURE_IMPORTANCE_CSV = OUTPUT_DIR / "feature_selection_scores.csv"

TARGET_COL = "effective_density_kg_m3"

CORR_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 0.0001

TOP_K_FEATURES = 10

RANDOM_STATE = 42


def remove_high_correlation(df, threshold=0.95):
    corr_matrix = df.corr(numeric_only=True).abs()

    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    to_drop = [
        column
        for column in upper.columns
        if any(upper[column] > threshold)
    ]

    return to_drop


def main():

    if not INPUT_CSV.exists():
        print(f"Input file missing: {INPUT_CSV}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded dataset:")
    print(df.shape)

    y = df[TARGET_COL]
    X = df.drop(columns=[TARGET_COL])

    # ==========================================================
    # STEP 1 — Remove high correlation
    # ==========================================================

    high_corr_drop = remove_high_correlation(X, CORR_THRESHOLD)

    X_corr = X.drop(columns=high_corr_drop)

    pd.DataFrame({
        "dropped_feature": high_corr_drop
    }).to_csv(CORR_DROPPED_CSV, index=False)

    corr_pruned_df = pd.concat([X_corr, y], axis=1)

    corr_pruned_df.to_csv(CORR_PRUNED_CSV, index=False)

    print("\nSTEP 1 — Correlation pruning")
    print(f"Dropped: {len(high_corr_drop)}")
    print(f"Shape after pruning: {corr_pruned_df.shape}")

    # ==========================================================
    # STEP 2 — Remove low variance numeric columns
    # ==========================================================

    numeric_cols = X_corr.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    categorical_cols = X_corr.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numeric_df = X_corr[numeric_cols]

    selector = VarianceThreshold(threshold=VARIANCE_THRESHOLD)

    selector.fit(numeric_df)

    keep_mask = selector.get_support()

    kept_numeric_cols = numeric_df.columns[keep_mask].tolist()

    dropped_low_variance = [
        c for c in numeric_cols
        if c not in kept_numeric_cols
    ]

    pd.DataFrame({
        "dropped_feature": dropped_low_variance
    }).to_csv(LOW_VARIANCE_DROPPED_CSV, index=False)

    X_variance = pd.concat(
        [
            numeric_df[kept_numeric_cols],
            X_corr[categorical_cols]
        ],
        axis=1
    )

    print("\nSTEP 2 — Low variance pruning")
    print(f"Dropped: {len(dropped_low_variance)}")
    print(f"Shape after pruning: {X_variance.shape}")

    # ==========================================================
    # STEP 3 — Encode + preprocess
    # ==========================================================

    numeric_cols = X_variance.select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    categorical_cols = X_variance.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median"))
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
        ]
    )

    X_processed = preprocess.fit_transform(X_variance)

    feature_names = preprocess.get_feature_names_out()

    print("\nSTEP 3 — Encoded feature matrix")
    print(X_processed.shape)

    # ==========================================================
    # STEP 4 — Mutual Information
    # ==========================================================

    print("\nSTEP 4 — Computing mutual information")

    mi_scores = mutual_info_regression(
        X_processed,
        y,
        random_state=RANDOM_STATE,
    )

    mi_df = pd.DataFrame({
        "feature": feature_names,
        "mutual_information": mi_scores,
    }).sort_values(
        by="mutual_information",
        ascending=False
    )

    # ==========================================================
    # STEP 5 — Extra Trees Importance
    # ==========================================================

    print("\nSTEP 5 — Training Extra Trees")

    X_train, X_test, y_train, y_test = train_test_split(
        X_processed,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
    )

    model = ExtraTreesRegressor(
        n_estimators=400,
        max_depth=8,
        min_samples_leaf=3,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)

    et_importance = model.feature_importances_

    et_df = pd.DataFrame({
        "feature": feature_names,
        "extra_trees_importance": et_importance,
    }).sort_values(
        by="extra_trees_importance",
        ascending=False
    )

    # ==========================================================
    # STEP 6 — Permutation Importance
    # ==========================================================

    print("\nSTEP 6 — Computing permutation importance")

    perm = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    perm_df = pd.DataFrame({
        "feature": feature_names,
        "permutation_importance": perm.importances_mean,
    }).sort_values(
        by="permutation_importance",
        ascending=False
    )

    # ==========================================================
    # STEP 7 — Merge all importance metrics
    # ==========================================================

    importance_df = (
        mi_df.merge(et_df, on="feature")
             .merge(perm_df, on="feature")
    )

    importance_df["combined_score"] = (
        importance_df["mutual_information"].rank(pct=True)
        + importance_df["extra_trees_importance"].rank(pct=True)
        + importance_df["permutation_importance"].rank(pct=True)
    )

    importance_df = importance_df.sort_values(
        by="combined_score",
        ascending=False
    )

    importance_df.to_csv(FEATURE_IMPORTANCE_CSV, index=False)

    print("\nTop 30 features:")
    print(
        importance_df.head(30).to_string(index=False)
    )

        # ==========================================================
    # STEP 8 — Final selected dataset
    # ==========================================================

    top_features_encoded = (
        importance_df.head(TOP_K_FEATURES)["feature"]
        .tolist()
    )

    X_processed_df = pd.DataFrame(
        X_processed,
        columns=feature_names,
    )

    final_df = X_processed_df[top_features_encoded].copy()

    # ----------------------------------------------------------
    # Preserve timestamp ONLY for grouped CV
    # NOT as predictor
    # ----------------------------------------------------------

    final_df["timestamp"] = df["timestamp"].values

    # ----------------------------------------------------------
    # Add target
    # ----------------------------------------------------------

    final_df[TARGET_COL] = y.values

    # ----------------------------------------------------------
    # Save
    # ----------------------------------------------------------

    final_df.to_csv(FINAL_SELECTED_CSV, index=False)

    print("\nSTEP 8 — Final selected dataset")
    print(final_df.shape)

    print("\nSaved files:")
    print(f"- {CORR_PRUNED_CSV}")
    print(f"- {FINAL_SELECTED_CSV}")
    print(f"- {FEATURE_IMPORTANCE_CSV}")
    print(f"- {CORR_DROPPED_CSV}")
    print(f"- {LOW_VARIANCE_DROPPED_CSV}")

    print("\nIMPORTANT:")
    print("Use model_density_selected_features.csv for final ML training.")
    print("timestamp is preserved ONLY for GroupKFold.")

    print("\nSTEP 8 — Final selected dataset")
    print(final_df.shape)

    print("\nSaved files:")
    print(f"- {CORR_PRUNED_CSV}")
    print(f"- {FINAL_SELECTED_CSV}")
    print(f"- {FEATURE_IMPORTANCE_CSV}")
    print(f"- {CORR_DROPPED_CSV}")
    print(f"- {LOW_VARIANCE_DROPPED_CSV}")

    print("\nIMPORTANT:")
    print("Use model_density_selected_features.csv for final ML training.")


if __name__ == "__main__":
    main()