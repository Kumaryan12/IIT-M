from pathlib import Path

import pandas as pd


BASE_DATASET = Path(
    "outputs/features/master_sensor_osm_visual_roi_density_dataset.csv"
)

SEG_DATASET = Path(
    "outputs/features/sample_segmentation_occupancy_features_v1.csv"
)

OUTPUT_CSV = Path(
    "outputs/features/segmentation_density_dataset_v1.csv"
)


ENVIRONMENT_COLS = [
    "temperature_c",
    "relative_humidity",
    "co_ppb",
    "no2_ppb",
    "so2_ppb",
    "o3_ppb_compensated",
]


DROP_COLS = [
    "sample_id",
    "timestamp",
    "matched_run_id",
]

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

def main():

    if not BASE_DATASET.exists():
        print(f"Missing: {BASE_DATASET}")
        return

    if not SEG_DATASET.exists():
        print(f"Missing: {SEG_DATASET}")
        return

    base = pd.read_csv(BASE_DATASET)
    seg = pd.read_csv(SEG_DATASET)

    print("\nLoaded:")
    print(f"Base dataset: {base.shape}")
    print(f"Seg dataset: {seg.shape}")

    keep_cols = [
        c for c in seg.columns
        if c not in DROP_COLS
    ]

    merge_cols = [
        "sample_id",
        "timestamp",
    ]

    seg_merge = seg[merge_cols + keep_cols].copy()

    merged = base.merge(
        seg_merge,
        on=["sample_id", "timestamp"],
        how="inner",
    )

    print(f"\nMerged shape: {merged.shape}")

    # Remove environment features
    env_cols_present = [
        c for c in ENVIRONMENT_COLS
        if c in merged.columns
    ]

    merged_no_env = merged.drop(
        columns=env_cols_present
    )

    print("\nRemoved environment columns:")
    print(env_cols_present)

    drop_cols = [
    c for c in LEAKAGE_COLS + ENVIRONMENT_COLS
    if c in merged.columns
]

    merged_no_env = merged.drop(columns=drop_cols)

    merged_no_env.to_csv(
        OUTPUT_CSV,
        index=False,
    )

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nFinal dataset shape:")
    print(merged_no_env.shape)

    print("\nSegmentation feature examples:")

    seg_cols = [
        c for c in merged_no_env.columns
        if c.startswith("seg_")
    ]

    print(seg_cols[:20])


if __name__ == "__main__":
    main()