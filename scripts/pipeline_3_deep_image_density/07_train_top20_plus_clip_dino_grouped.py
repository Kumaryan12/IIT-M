from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from xgboost import XGBRegressor


TOP20_CSV = Path("outputs/features/minimal_temporal_no_env_top20.csv")
CLIP_CSV = Path("outputs/features/sample_clip_image_embeddings_v1.csv")
DINO_CSV = Path("outputs/features/sample_dinov2_image_embeddings_v1.csv")

OUTPUT_DIR = Path("outputs/models/pipeline_3_clip_dino_fusion")
RESULTS_CSV = OUTPUT_DIR / "top20_plus_clip_dino_grouped_results.csv"

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
        "RandomForest": RandomForestRegressor(
            n_estimators=500,
            max_depth=6,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "ExtraTrees": ExtraTreesRegressor(
            n_estimators=500,
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

    top20 = pd.read_csv(TOP20_CSV)
    clip = pd.read_csv(CLIP_CSV)
    dino = pd.read_csv(DINO_CSV)

    print("\nLoaded:")
    print("Top20:", top20.shape)
    print("CLIP:", clip.shape)
    print("DINO:", dino.shape)

    clip_cols = [c for c in clip.columns if c.startswith("image_clip_mean_")]
    dino_cols = [c for c in dino.columns if c.startswith("image_dino_mean_")]

    merged = top20.merge(
        clip[["sample_id", "timestamp"] + clip_cols],
        on=["sample_id", "timestamp"],
        how="inner",
    )

    merged = merged.merge(
        dino[["sample_id", "timestamp"] + dino_cols],
        on=["sample_id", "timestamp"],
        how="inner",
    )

    print("\nMerged:")
    print(merged.shape)

    y = merged[TARGET_COL]
    groups = merged[GROUP_COL]

    X = merged.drop(columns=["sample_id", "timestamp", TARGET_COL])

    clip_cols_in_X = [c for c in X.columns if c.startswith("image_clip_mean_")]
    dino_cols_in_X = [c for c in X.columns if c.startswith("image_dino_mean_")]
    structured_cols = [
        c for c in X.columns
        if c not in clip_cols_in_X and c not in dino_cols_in_X
    ]

    print("\nFeature counts:")
    print("Structured:", len(structured_cols))
    print("CLIP:", len(clip_cols_in_X))
    print("DINO:", len(dino_cols_in_X))

    structured_num = X[structured_cols].select_dtypes(
        include=["int64", "float64", "bool"]
    ).columns.tolist()

    structured_cat = X[structured_cols].select_dtypes(
        include=["object"]
    ).columns.tolist()

    pca_options = [
        (4, 4),
        (8, 8),
        (8, 16),
        (16, 8),
        (16, 16),
    ]

    gkf = GroupKFold(n_splits=N_SPLITS)
    results = []

    for clip_pca, dino_pca in pca_options:
        print("\n" + "=" * 80)
        print(f"CLIP PCA={clip_pca}, DINO PCA={dino_pca}")
        print("=" * 80)

        preprocess = ColumnTransformer([
            (
                "structured_num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                ]),
                structured_num,
            ),
            (
                "structured_cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore")),
                ]),
                structured_cat,
            ),
            (
                "clip",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=clip_pca, svd_solver="full")),
                ]),
                clip_cols_in_X,
            ),
            (
                "dino",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("pca", PCA(n_components=dino_pca, svd_solver="full")),
                ]),
                dino_cols_in_X,
            ),
        ])

        for model_name, model in build_models().items():
            print(f"\nRunning {model_name}")

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
            metrics["clip_pca_components"] = clip_pca
            metrics["dino_pca_components"] = dino_pca
            metrics["num_structured_features"] = len(structured_cols)
            metrics["num_clip_features"] = len(clip_cols_in_X)
            metrics["num_dino_features"] = len(dino_cols_in_X)

            results.append(metrics)
            print(metrics)

    results_df = pd.DataFrame(results).sort_values(by="RMSE")
    results_df.to_csv(RESULTS_CSV, index=False)

    print("\n" + "=" * 80)
    print("TOP20 + CLIP + DINO FUSION RESULTS")
    print("=" * 80)
    print(results_df.to_string(index=False))

    print(f"\nSaved to: {RESULTS_CSV}")

    print("\nBenchmarks:")
    print("Top20 only: R2 = 0.224, RMSE = 487.56")
    print("Top20 + DINO: R2 = 0.252, RMSE = 478.44")
    print("CLIP image-only: R2 = 0.115, RMSE = 520.56")


if __name__ == '__main__':
    main()