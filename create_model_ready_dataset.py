from pathlib import Path

import pandas as pd


INPUT_CSV = Path("outputs/features/master_sensor_osm_visual_roi_density_dataset.csv")
OUTPUT_CSV = Path("outputs/features/model_ready_density_dataset.csv")

TARGET_COL = "effective_density_kg_m3"

LEAKAGE_COLS = [
    "pm1_mass",
    "pm2_5_mass",
    "pm4_mass",
    "pm10_mass",
    "npm1_count",
    "npm2_5_count",
    "npm4_count",
    "npm10_count",
    "pm25_mass_kg_m3",
    "npm2_5_particles_m3",
    "total_particle_volume_m3",
    "single_particle_volume_m3",
    "effective_radius_m",
    "effective_diameter_um",
]

DROP_META_COLS = [
    "sample_id",
    #"timestamp",
    "osm_location_key",
    "osm_error",
    "matched_run_id",
    "lenses_used",
    "road_roi_lenses_used",
]


def main():
    if not INPUT_CSV.exists():
        print(f"Input file not found: {INPUT_CSV}")
        print("Run compute_effective_density.py first.")
        return

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded density dataset:")
    print(df.shape)

    if TARGET_COL not in df.columns:
        print(f"Target column not found: {TARGET_COL}")
        return

    # Keep only rows with visual and road ROI features
    if "has_visual_features" in df.columns:
        df = df[df["has_visual_features"] == True].copy()

    if "has_road_roi_features" in df.columns:
        df = df[df["has_road_roi_features"] == True].copy()

    # Remove leakage and metadata columns
    drop_cols = [c for c in LEAKAGE_COLS + DROP_META_COLS if c in df.columns]
    df = df.drop(columns=drop_cols)

    # Drop columns that are completely empty
    df = df.dropna(axis=1, how="all")

    # Drop constant columns
    constant_cols = []
    for col in df.columns:
        if col == TARGET_COL:
            continue
        if df[col].nunique(dropna=False) <= 1:
            constant_cols.append(col)

    df = df.drop(columns=constant_cols)

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved model-ready dataset to: {OUTPUT_CSV}")
    print(f"Output shape: {df.shape}")

    print("\nDropped leakage/meta columns:")
    print(drop_cols)

    print("\nDropped constant columns:")
    print(constant_cols)

    print("\nTarget summary:")
    print(df[TARGET_COL].describe())


if __name__ == "__main__":
    main()