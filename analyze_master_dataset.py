from pathlib import Path

import pandas as pd


INPUT_PATH = Path("outputs/features/master_sensor_osm_visual_dataset.csv")

OUTPUT_DIR = Path("outputs/features")
SUMMARY_TXT_PATH = OUTPUT_DIR / "master_dataset_summary.txt"
NUMERIC_SUMMARY_PATH = OUTPUT_DIR / "master_numeric_summary.csv"
CORRELATION_PATH = OUTPUT_DIR / "master_correlation_with_pm.csv"


PM_TARGET_COLS = [
    "pm1_mass",
    "pm2_5_mass",
    "pm4_mass",
    "pm10_mass",
    "npm1_count",
    "npm2_5_count",
    "npm4_count",
    "npm10_count",
]


VISUAL_KEY_COLS = [
    "has_visual_features",
    "matched_run_id",
    "num_lens_frames_used",
    "lenses_used",
    "total_vehicles_sum",
    "traffic_load_score_sum",
    "road_dust_score_mean",
    "road_dust_score_max",
    "road_condition_mode",
    "road_dust_level_mode",
    "lens1_total_vehicles",
    "lens4_total_vehicles",
    "lens6_total_vehicles",
    "lens1_traffic_load_score",
    "lens4_traffic_load_score",
    "lens6_traffic_load_score",
]


OSM_KEY_COLS = [
    "fuel_station_count_250m",
    "restaurant_count_250m",
    "bus_stop_count_250m",
    "parking_count_250m",
    "construction_count_250m",
    "industrial_count_250m",
    "park_count_250m",
    "commercial_count_250m",
    "road_segment_count_250m",
    "total_road_length_250m",
    "primary_road_count_250m",
    "secondary_road_count_250m",
    "residential_road_count_250m",
]


def write_line(lines, text=""):
    lines.append(str(text))


def main():
    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        print("Run merge_sensor_osm_visual.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    lines = []

    write_line(lines, "MASTER SENSOR + OSM + VISUAL DATASET SUMMARY")
    write_line(lines, "=" * 60)

    write_line(lines, "\n1. Basic Shape")
    write_line(lines, f"Rows: {df.shape[0]}")
    write_line(lines, f"Columns: {df.shape[1]}")

    write_line(lines, "\n2. Columns")
    write_line(lines, list(df.columns))

    write_line(lines, "\n3. Missing Values: Top 30")
    missing = df.isna().sum().sort_values(ascending=False)
    write_line(lines, missing.head(30).to_string())

    write_line(lines, "\n4. Visual Feature Availability")
    if "has_visual_features" in df.columns:
        write_line(lines, df["has_visual_features"].value_counts(dropna=False).to_string())
    else:
        write_line(lines, "Column has_visual_features not found.")

    write_line(lines, "\n5. Matched Run Counts")
    if "matched_run_id" in df.columns:
        write_line(lines, df["matched_run_id"].value_counts(dropna=False).to_string())
    else:
        write_line(lines, "Column matched_run_id not found.")

    write_line(lines, "\n6. Timestamp Range")
    if "timestamp" in df.columns:
        time_series = pd.to_datetime(df["timestamp"], errors="coerce")
        write_line(lines, f"Start: {time_series.min()}")
        write_line(lines, f"End:   {time_series.max()}")
        write_line(lines, f"Unique timestamps: {time_series.nunique()}")
    else:
        write_line(lines, "timestamp column not found.")

    write_line(lines, "\n7. Coordinate Range")
    if "latitude" in df.columns and "longitude" in df.columns:
        write_line(lines, f"Latitude:  {df['latitude'].min()} to {df['latitude'].max()}")
        write_line(lines, f"Longitude: {df['longitude'].min()} to {df['longitude'].max()}")
    else:
        write_line(lines, "latitude/longitude columns not found.")

    write_line(lines, "\n8. PM / Particle Summary")
    available_pm_cols = [col for col in PM_TARGET_COLS if col in df.columns]
    if available_pm_cols:
        write_line(lines, df[available_pm_cols].describe().to_string())
    else:
        write_line(lines, "No PM/count columns found.")

    write_line(lines, "\n9. Key Visual Feature Summary")
    available_visual_cols = [col for col in VISUAL_KEY_COLS if col in df.columns]
    if available_visual_cols:
        write_line(lines, df[available_visual_cols].describe(include="all").to_string())
    else:
        write_line(lines, "No key visual columns found.")

    write_line(lines, "\n10. Key OSM Feature Summary")
    available_osm_cols = [col for col in OSM_KEY_COLS if col in df.columns]
    if available_osm_cols:
        write_line(lines, df[available_osm_cols].describe().to_string())
    else:
        write_line(lines, "No key OSM columns found.")

    write_line(lines, "\n11. Road Condition Counts")
    if "road_condition_mode" in df.columns:
        write_line(lines, df["road_condition_mode"].value_counts(dropna=False).to_string())
    else:
        write_line(lines, "road_condition_mode column not found.")

    write_line(lines, "\n12. Road Dust Level Counts")
    if "road_dust_level_mode" in df.columns:
        write_line(lines, df["road_dust_level_mode"].value_counts(dropna=False).to_string())
    else:
        write_line(lines, "road_dust_level_mode column not found.")

    # Numeric summary
    numeric_df = df.select_dtypes(include=["int64", "float64", "bool"])
    numeric_summary = numeric_df.describe().T
    numeric_summary.to_csv(NUMERIC_SUMMARY_PATH)

    # Correlation with PM columns
    corr_rows = []

    numeric_cols = numeric_df.columns.tolist()

    for target_col in available_pm_cols:
        if target_col not in numeric_df.columns:
            continue

        correlations = numeric_df[numeric_cols].corrwith(numeric_df[target_col])
        correlations = correlations.dropna()

        for feature_name, corr_value in correlations.items():
            if feature_name == target_col:
                continue

            corr_rows.append({
                "target": target_col,
                "feature": feature_name,
                "correlation": corr_value,
                "abs_correlation": abs(corr_value),
            })

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        corr_df = corr_df.sort_values(
            by=["target", "abs_correlation"],
            ascending=[True, False]
        )
        corr_df.to_csv(CORRELATION_PATH, index=False)

        write_line(lines, "\n13. Top Correlations With PM Columns")
        for target_col in available_pm_cols:
            top_df = corr_df[corr_df["target"] == target_col].head(15)
            write_line(lines, f"\nTop correlations for {target_col}:")
            write_line(lines, top_df[["feature", "correlation"]].to_string(index=False))
    else:
        write_line(lines, "\n13. Correlation file not created because no numeric PM columns were available.")

    # Save text summary
    SUMMARY_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"Summary text saved to: {SUMMARY_TXT_PATH}")
    print(f"Numeric summary saved to: {NUMERIC_SUMMARY_PATH}")

    if corr_rows:
        print(f"Correlation file saved to: {CORRELATION_PATH}")

    print("\nQuick status:")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    if "has_visual_features" in df.columns:
        print("\nVisual feature availability:")
        print(df["has_visual_features"].value_counts(dropna=False))

    if "road_condition_mode" in df.columns:
        print("\nRoad condition counts:")
        print(df["road_condition_mode"].value_counts(dropna=False))

    if available_pm_cols:
        print("\nPM summary:")
        print(df[available_pm_cols].describe())


if __name__ == "__main__":
    main()