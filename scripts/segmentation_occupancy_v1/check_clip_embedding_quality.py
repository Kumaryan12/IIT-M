from pathlib import Path

import numpy as np
import pandas as pd


CLIP_CSV = Path("outputs/features/sample_clip_embeddings_v1.csv")


def main():
    df = pd.read_csv(CLIP_CSV)

    clip_cols = [
        c for c in df.columns
        if c.startswith("sample_clip_mean_")
    ]

    print("\nDataset shape:")
    print(df.shape)

    print("\nCLIP embedding columns:")
    print(len(clip_cols))

    X = df[clip_cols].copy()

    # Convert safely to numeric
    X_numeric = X.apply(pd.to_numeric, errors="coerce")

    nan_count = X_numeric.isna().sum().sum()
    inf_count = np.isinf(X_numeric.to_numpy()).sum()

    print("\nCorruption summary:")
    print(f"NaN count: {nan_count}")
    print(f"Inf count: {inf_count}")

    print("\nPer-column NaN count top 20:")
    print(X_numeric.isna().sum().sort_values(ascending=False).head(20))

    print("\nPer-row NaN count top 20:")
    row_nan_counts = X_numeric.isna().sum(axis=1)
    print(row_nan_counts.sort_values(ascending=False).head(20))

    print("\nValue range:")
    print(f"Min: {np.nanmin(X_numeric.to_numpy())}")
    print(f"Max: {np.nanmax(X_numeric.to_numpy())}")
    print(f"Mean: {np.nanmean(X_numeric.to_numpy())}")
    print(f"Std: {np.nanstd(X_numeric.to_numpy())}")

    bad_rows = df.loc[row_nan_counts > 0, ["sample_id", "timestamp"]]

    print("\nRows with corrupted CLIP values:")
    print(bad_rows.head(30).to_string(index=False))

    print("\nDone.")


if __name__ == "__main__":
    main()