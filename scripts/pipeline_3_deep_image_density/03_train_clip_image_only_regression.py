from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

from xgboost import XGBRegressor


INPUT_CSV = Path(
    "outputs/features/sample_clip_image_embeddings_v1.csv"
)

OUTPUT_DIR = Path(
    "outputs/models/pipeline_3_clip_image_only"
)

RESULTS_CSV = OUTPUT_DIR / "clip_image_only_results.csv"

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

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded:")
    print(df.shape)

    y = df[TARGET_COL]
    groups = df[GROUP_COL]

    emb_cols = [
        c for c in df.columns
        if c.startswith("image_clip_mean_")
    ]

    X = df[emb_cols].copy()

    print("\nEmbedding features:")
    print(len(emb_cols))

    pca_options = [4, 8, 16, 32, 64]

    gkf = GroupKFold(n_splits=N_SPLITS)

    results = []

    for n_pca in pca_options:
        print("\n" + "=" * 80)
        print(f"PCA = {n_pca}")
        print("=" * 80)

        for model_name, model in build_models().items():

            print(f"\nRunning {model_name}")

            pipe = Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(
                    n_components=n_pca,
                    random_state=RANDOM_STATE,
                )),
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
            metrics["pca_components"] = n_pca
            metrics["num_embedding_features"] = len(emb_cols)

            results.append(metrics)

            print(metrics)

    results_df = (
        pd.DataFrame(results)
        .sort_values(by="RMSE")
    )

    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 80)
    print("PIPELINE 3 — IMAGE ONLY CLIP RESULTS")
    print("=" * 80)

    print(results_df.to_string(index=False))

    print(f"\nSaved to: {RESULTS_CSV}")

    print("\nBenchmarks:")
    print("True no-env handcrafted best: R2 ≈ 0.172")
    print("CLIP+OSM old best: R2 ≈ 0.107")
    print("Segmentation best: R2 ≈ 0.04")


if __name__ == "__main__":
    main()