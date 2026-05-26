from pathlib import Path
import pandas as pd

INPUT_CSV = Path("outputs/features/model_ready_density_dataset.csv")
OUTPUT_CSV = Path("outputs/features/model_ready_density_dataset_pruned.csv")

TARGET_COL = "effective_density_kg_m3"

DROP_SUFFIXES = [
    "_sum",
    "_max",
    "_min",
]

ALWAYS_KEEP = [
    TARGET_COL,
    "temperature_c",
    "relative_humidity",
    "co_ppb",
    "no2_ppb",
    "so2_ppb",
    "o3_ppb_compensated",
    "latitude",
    "longitude",
    "osm_latitude_rounded",
    "osm_longitude_rounded",
]


def main():
    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded:")
    print(df.shape)

    drop_cols = []

    for col in df.columns:
        if col in ALWAYS_KEEP:
            continue

        if any(col.endswith(suffix) for suffix in DROP_SUFFIXES):
            drop_cols.append(col)

    pruned_df = df.drop(columns=drop_cols)

    # Drop highly redundant duplicate columns after suffix pruning
    constant_cols = [
        col for col in pruned_df.columns
        if col != TARGET_COL and pruned_df[col].nunique(dropna=False) <= 1
    ]

    pruned_df = pruned_df.drop(columns=constant_cols)

    pruned_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Original shape: {df.shape}")
    print(f"Pruned shape: {pruned_df.shape}")

    print("\nDropped suffix-based columns:")
    print(len(drop_cols))

    print("\nDropped constant columns:")
    print(constant_cols)

    print("\nRemaining columns:")
    print(list(pruned_df.columns))


if __name__ == "__main__":
    main()