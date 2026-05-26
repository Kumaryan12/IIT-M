from pathlib import Path

import numpy as np
import pandas as pd


INPUT_CSV = Path(
    "outputs/features/model_ready_density_dataset.csv"
)

TARGET_COL = "effective_density_kg_m3"

TOP_K = 10


def classify_feature(col_name: str) -> str:

    col = col_name.lower()

    if any(
        k in col for k in [
            "dust",
            "haze",
            "brightness",
            "contrast",
            "edge",
            "brown",
            "road_condition",
            "visual"
        ]
    ):
        return "visual"

    if any(
        k in col for k in [
            "vehicle",
            "traffic",
            "truck",
            "bus",
            "car",
            "motorcycle",
            "two_wheeler",
            "heavy_vehicle"
        ]
    ):
        return "traffic"

    if any(
        k in col for k in [
            "road",
            "commercial",
            "construction",
            "restaurant",
            "industrial",
            "park",
            "bus_stop",
            "warehouse",
            "retail",
            "marketplace",
            "school",
            "hospital"
        ]
    ):
        return "osm"

    if any(
        k in col for k in [
            "temperature",
            "humidity",
            "co_ppb",
            "so2_ppb",
            "no2_ppb",
            "o3"
        ]
    ):
        return "environment"

    if any(
        k in col for k in [
            "latitude",
            "longitude"
        ]
    ):
        return "spatial"

    return "other"


def main():

    if not INPUT_CSV.exists():
        print(f"Dataset not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    print("\nMODEL READY DATASET ANALYSIS")
    print("=" * 70)

    print("\n1. Dataset Shape")
    print(df.shape)

    if TARGET_COL not in df.columns:
        print(f"Missing target column: {TARGET_COL}")
        return

    # ---------------------------------------------------
    # Feature categorization
    # ---------------------------------------------------

    feature_categories = {}

    for col in df.columns:
        if col == TARGET_COL:
            continue

        category = classify_feature(col)
        feature_categories[col] = category

    category_counts = (
        pd.Series(feature_categories)
        .value_counts()
    )

    print("\n2. Feature Category Counts")
    print(category_counts)

    # ---------------------------------------------------
    # Missing values
    # ---------------------------------------------------

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    print("\n3. Missing Value Columns")
    print(missing.head(30))

    # ---------------------------------------------------
    # Numeric correlation with target
    # ---------------------------------------------------

    numeric_df = df.select_dtypes(
        include=[np.number]
    )

    correlations = (
        numeric_df.corr()[TARGET_COL]
        .drop(TARGET_COL)
        .sort_values(
            key=lambda s: s.abs(),
            ascending=False
        )
    )

    print("\n4. Top Positive Correlations")
    print(correlations.head(TOP_K))

    print("\n5. Top Negative Correlations")
    print(correlations.tail(TOP_K))

    # ---------------------------------------------------
    # Redundant feature pairs
    # ---------------------------------------------------

    corr_matrix = numeric_df.corr().abs()

    redundant_pairs = []

    cols = corr_matrix.columns

    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):

            c1 = cols[i]
            c2 = cols[j]

            if c1 == TARGET_COL or c2 == TARGET_COL:
                continue

            val = corr_matrix.iloc[i, j]

            if val > 0.95:
                redundant_pairs.append(
                    (c1, c2, val)
                )

    redundant_pairs = sorted(
        redundant_pairs,
        key=lambda x: x[2],
        reverse=True
    )

    print("\n6. Highly Redundant Feature Pairs (>0.95)")
    print(f"Count: {len(redundant_pairs)}")

    for pair in redundant_pairs[:40]:
        print(pair)

    # ---------------------------------------------------
    # Near-constant columns
    # ---------------------------------------------------

    near_constant = []

    for col in numeric_df.columns:

        if col == TARGET_COL:
            continue

        unique_ratio = (
            numeric_df[col]
            .nunique(dropna=True)
            / len(numeric_df)
        )

        if unique_ratio < 0.01:
            near_constant.append(
                (col, unique_ratio)
            )

    print("\n7. Near-Constant Columns")
    print(near_constant)

    # ---------------------------------------------------
    # Category-wise target correlation
    # ---------------------------------------------------

    category_scores = []

    for category in sorted(set(feature_categories.values())):

        cols = [
            c for c, cat in feature_categories.items()
            if cat == category
            and c in correlations.index
        ]

        if not cols:
            continue

        avg_abs_corr = (
            correlations[cols]
            .abs()
            .mean()
        )

        max_abs_corr = (
            correlations[cols]
            .abs()
            .max()
        )

        category_scores.append({
            "category": category,
            "num_features": len(cols),
            "avg_abs_corr": avg_abs_corr,
            "max_abs_corr": max_abs_corr
        })

    category_df = pd.DataFrame(category_scores)

    print("\n8. Category-wise Correlation Strength")
    print(
        category_df.sort_values(
            by="avg_abs_corr",
            ascending=False
        )
    )

    # ---------------------------------------------------
    # Target distribution
    # ---------------------------------------------------

    print("\n9. Target Distribution")
    print(
        df[TARGET_COL]
        .describe()
    )

    print("\n10. Suggested Next Steps")

    print("""
- Remove highly redundant features
- Build baseline regressors
- Perform ablation studies:
    * visual only
    * traffic only
    * OSM only
    * environment only
    * all features
- Use RandomForest + XGBoost first
- Use SHAP for explainability
- Compare against mean-predictor baseline
""")

    print("\nDone.")


if __name__ == "__main__":
    main()