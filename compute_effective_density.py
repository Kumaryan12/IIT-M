import math
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("outputs/features/master_sensor_osm_visual_roi_dataset.csv")
OUTPUT_CSV = Path("outputs/features/master_sensor_osm_visual_roi_density_dataset.csv")

# Assumption from team formula pipeline
EFFECTIVE_DIAMETER_UM = 0.35

PM25_COL = "pm2_5_mass"
NPM25_COL = "npm2_5_count"


def main():
    if not INPUT_CSV.exists():
        print(f"Input file not found: {INPUT_CSV}")
        return

    df = pd.read_csv(INPUT_CSV)

    print("\nLoaded master dataset:")
    print(df.shape)

    required_cols = [PM25_COL, NPM25_COL]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        print("Missing required columns:")
        print(missing)
        return

    # PM2.5 mass concentration: µg/m³ → kg/m³
    df["pm25_mass_kg_m3"] = df[PM25_COL] * 1e-9

    # Effective particle diameter assumption
    df["effective_diameter_um"] = EFFECTIVE_DIAMETER_UM

    effective_diameter_m = EFFECTIVE_DIAMETER_UM * 1e-6
    effective_radius_m = effective_diameter_m / 2.0

    df["effective_radius_m"] = effective_radius_m

    # Spherical particle volume
    single_particle_volume_m3 = (4.0 / 3.0) * math.pi * (effective_radius_m ** 3)
    df["single_particle_volume_m3"] = single_particle_volume_m3

    # Particle count: particles/cm³ → particles/m³
    df["npm2_5_particles_m3"] = df[NPM25_COL] * 1e6

    # Total particle volume concentration
    df["total_particle_volume_m3"] = (
        df["single_particle_volume_m3"] * df["npm2_5_particles_m3"]
    )

    # Target variable: effective density parameter
    epsilon = 1e-18
    df["effective_density_kg_m3"] = (
        df["pm25_mass_kg_m3"] / (df["total_particle_volume_m3"] + epsilon)
    )

    df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")

    print("\nEffective density target summary:")
    print(df["effective_density_kg_m3"].describe())

    print("\nPreview:")
    print(
        df[
            [
                "sample_id",
                "timestamp",
                PM25_COL,
                NPM25_COL,
                "effective_density_kg_m3",
            ]
        ].head(10)
    )

    print("\nImportant:")
    print("effective_density_kg_m3 is the ML target.")
    print("Do NOT use PM/count columns as predictors while training.")


if __name__ == "__main__":
    main()