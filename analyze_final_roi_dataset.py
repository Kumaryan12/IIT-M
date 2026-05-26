from pathlib import Path

import pandas as pd
import numpy as np


INPUT_PATH = Path("outputs/features/master_sensor_osm_visual_roi_dataset.csv")

OUTPUT_DIR = Path("outputs/final_analysis")
SUMMARY_TXT_PATH = OUTPUT_DIR / "final_roi_dataset_summary.txt"
NUMERIC_SUMMARY_PATH = OUTPUT_DIR / "final_roi_numeric_summary.csv"
CORR_PM_PATH = OUTPUT_DIR / "final_roi_correlation_with_pm.csv"
CORR_NON_LEAKAGE_PATH = OUTPUT_DIR / "final_roi_non_leakage_feature_correlation.csv"
TIMESTAMP_SUMMARY_PATH = OUTPUT_DIR / "final_roi_timestamp_summary.csv"


PM_COUNT_COLS = [
    "pm1_mass",
    "pm2_5_mass",
    "pm4_mass",
    "pm10_mass",
    "npm1_count",
    "npm2_5_count",
    "npm4_count",
    "npm10_count",
]


GAS_ENV_COLS = [
    "temperature_c",
    "relative_humidity",
    "co_ppb",
    "no2_ppb",
    "so2_ppb",
    "o3_ppb_compensated",
]


ID_META_COLS = [
    "sample_id",
    "timestamp",
    "latitude",
    "longitude",
    "osm_location_key",
    "osm_latitude_rounded",
    "osm_longitude_rounded",
    "osm_status",
    "osm_error",
    "matched_run_id",
    "lenses_used",
    "road_roi_lenses_used",
    "road_condition_mode",
    "road_dust_level_mode",
]


def write_line(lines, text=""):
    lines.append(str(text))


def get_existing_cols(df, cols):
    return [c for c in cols if c in df.columns]


def main():
    if not INPUT_PATH.exists():
        print(f"Dataset not found: {INPUT_PATH}")
        print("Run merge_sensor_osm_visual_roi.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    lines = []
    write_line(lines, "FINAL ROI MASTER DATASET ANALYSIS")
    write_line(lines, "=" * 70)

    write_line(lines, "\n1. Dataset Shape")
    write_line(lines, f"Rows: {df.shape[0]}")
    write_line(lines, f"Columns: {df.shape[1]}")

    write_line(lines, "\n2. Visual Feature Availability")
    if "has_visual_features" in df.columns:
        write_line(lines, df["has_visual_features"].value_counts(dropna=False).to_string())
    else:
        write_line(lines, "has_visual_features column not found.")

    write_line(lines, "\n3. Road ROI Feature Availability")
    if "has_road_roi_features" in df.columns:
        write_line(lines, df["has_road_roi_features"].value_counts(dropna=False).to_string())
    else:
        write_line(lines, "has_road_roi_features column not found.")

    write_line(lines, "\n4. Matched Run Counts")
    if "matched_run_id" in df.columns:
        write_line(lines, df["matched_run_id"].value_counts(dropna=False).to_string())

    write_line(lines, "\n5. Timestamp Summary")
    if "timestamp" in df.columns:
        df["timestamp_parsed"] = pd.to_datetime(df["timestamp"], errors="coerce")
        write_line(lines, f"Start: {df['timestamp_parsed'].min()}")
        write_line(lines, f"End: {df['timestamp_parsed'].max()}")
        write_line(lines, f"Unique timestamps: {df['timestamp_parsed'].nunique()}")

        timestamp_summary = (
            df.groupby("timestamp")
            .agg(
                rows_per_timestamp=("sample_id", "count"),
                mean_pm2_5=("pm2_5_mass", "mean") if "pm2_5_mass" in df.columns else ("sample_id", "count"),
                mean_pm10=("pm10_mass", "mean") if "pm10_mass" in df.columns else ("sample_id", "count"),
                visual_available=("has_visual_features", "max") if "has_visual_features" in df.columns else ("sample_id", "count"),
            )
            .reset_index()
        )
        timestamp_summary.to_csv(TIMESTAMP_SUMMARY_PATH, index=False)

        write_line(lines, "\nRows per timestamp:")
        write_line(lines, timestamp_summary["rows_per_timestamp"].describe().to_string())

    write_line(lines, "\n6. Missing Values: Top 40")
    missing = df.isna().sum().sort_values(ascending=False)
    write_line(lines, missing.head(40).to_string())

    write_line(lines, "\n7. PM and Particle Count Summary")
    pm_cols = get_existing_cols(df, PM_COUNT_COLS)
    if pm_cols:
        write_line(lines, df[pm_cols].describe().to_string())
    else:
        write_line(lines, "No PM/count columns found.")

    write_line(lines, "\n8. Gas/Environment Summary")
    gas_cols = get_existing_cols(df, GAS_ENV_COLS)
    if gas_cols:
        write_line(lines, df[gas_cols].describe().to_string())
    else:
        write_line(lines, "No gas/environment columns found.")

    # Key ROI visual columns
    key_visual_cols = [
        "total_vehicles_sum",
        "traffic_load_score_sum",
        "vehicle_box_area_ratio_sum",
        "road_roi_road_dust_score_mean",
        "road_roi_road_dust_score_max",
        "road_roi_visual_dust_score_mean",
        "road_roi_haze_score_mean",
        "road_roi_brown_pixel_ratio_mean",
        "road_roi_edge_density_mean",
        "lens1_total_vehicles",
        "lens4_total_vehicles",
        "lens6_total_vehicles",
        "lens1_road_roi_road_dust_score",
        "lens6_road_roi_road_dust_score",
    ]

    key_visual_cols = get_existing_cols(df, key_visual_cols)

    write_line(lines, "\n9. Key Visual/ROI Feature Summary")
    if key_visual_cols:
        write_line(lines, df[key_visual_cols].describe().to_string())
    else:
        write_line(lines, "No key visual/ROI columns found.")

    # Key OSM columns
    key_osm_cols = [
        "fuel_station_count_250m",
        "restaurant_count_250m",
        "bus_stop_count_250m",
        "parking_count_250m",
        "construction_count_250m",
        "industrial_count_250m",
        "commercial_count_250m",
        "retail_count_250m",
        "road_segment_count_250m",
        "total_road_length_250m",
        "primary_road_count_250m",
        "secondary_road_count_250m",
        "residential_road_count_250m",
        "service_road_count_250m",
    ]

    key_osm_cols = get_existing_cols(df, key_osm_cols)

    write_line(lines, "\n10. Key OSM Feature Summary")
    if key_osm_cols:
        write_line(lines, df[key_osm_cols].describe().to_string())
    else:
        write_line(lines, "No key OSM columns found.")

    write_line(lines, "\n11. Road Condition Counts")
    if "road_condition_mode" in df.columns:
        write_line(lines, df["road_condition_mode"].value_counts(dropna=False).to_string())

    write_line(lines, "\n12. Road Dust Level Counts")
    if "road_dust_level_mode" in df.columns:
        write_line(lines, df["road_dust_level_mode"].value_counts(dropna=False).to_string())

    # Numeric summary
    numeric_df = df.select_dtypes(include=[np.number, bool]).copy()
    numeric_df.describe().T.to_csv(NUMERIC_SUMMARY_PATH)

    # Correlation with PM/count columns
    corr_rows = []
    for target in pm_cols:
        if target not in numeric_df.columns:
            continue
        corrs = numeric_df.corrwith(numeric_df[target]).dropna()
        for feature, corr in corrs.items():
            if feature == target:
                continue
            corr_rows.append({
                "target": target,
                "feature": feature,
                "correlation": corr,
                "abs_correlation": abs(corr),
            })

    if corr_rows:
        corr_df = pd.DataFrame(corr_rows)
        corr_df = corr_df.sort_values(["target", "abs_correlation"], ascending=[True, False])
        corr_df.to_csv(CORR_PM_PATH, index=False)

        write_line(lines, "\n13. Top Correlations with PM/Count Columns")
        for target in pm_cols:
            top = corr_df[corr_df["target"] == target].head(20)
            write_line(lines, f"\nTop correlations for {target}:")
            write_line(lines, top[["feature", "correlation"]].to_string(index=False))

    # Non-leakage correlation table:
    # Exclude PM/count columns as features.
    non_leakage_numeric_cols = [
        col for col in numeric_df.columns
        if col not in PM_COUNT_COLS
    ]

    non_leakage_rows = []
    for target in pm_cols:
        if target not in numeric_df.columns:
            continue

        for feature in non_leakage_numeric_cols:
            if feature == target:
                continue

            valid = numeric_df[[feature, target]].dropna()
            if len(valid) < 10:
                continue

            corr = valid[feature].corr(valid[target])
            if pd.isna(corr):
                continue

            non_leakage_rows.append({
                "target": target,
                "feature": feature,
                "correlation": corr,
                "abs_correlation": abs(corr),
            })

    if non_leakage_rows:
        nl_corr_df = pd.DataFrame(non_leakage_rows)
        nl_corr_df = nl_corr_df.sort_values(
            ["target", "abs_correlation"],
            ascending=[True, False]
        )
        nl_corr_df.to_csv(CORR_NON_LEAKAGE_PATH, index=False)

        write_line(lines, "\n14. Top Non-Leakage Correlations")
        write_line(lines, "Note: PM/count columns are excluded as predictor features here.")
        for target in pm_cols:
            top = nl_corr_df[nl_corr_df["target"] == target].head(20)
            write_line(lines, f"\nTop non-leakage correlations for {target}:")
            write_line(lines, top[["feature", "correlation"]].to_string(index=False))

    # Dataset readiness verdict
    write_line(lines, "\n15. Readiness Verdict")
    if "has_visual_features" in df.columns:
        visual_count = int(df["has_visual_features"].sum())
        write_line(lines, f"Rows usable for visual/OSM modeling: {visual_count}/{len(df)}")
    if "has_road_roi_features" in df.columns:
        road_roi_count = int(df["has_road_roi_features"].sum())
        write_line(lines, f"Rows with road ROI features: {road_roi_count}/{len(df)}")

    write_line(lines, "\nRecommended modeling dataset:")
    write_line(lines, "Use rows where has_visual_features == True and has_road_roi_features == True.")
    write_line(lines, "Use PM/count columns only to compute effective density target.")
    write_line(lines, "Exclude PM/count columns from predictors to avoid data leakage.")

    SUMMARY_TXT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print("\nDone.")
    print(f"Summary saved to: {SUMMARY_TXT_PATH}")
    print(f"Numeric summary saved to: {NUMERIC_SUMMARY_PATH}")
    print(f"PM correlation saved to: {CORR_PM_PATH}")
    print(f"Non-leakage correlation saved to: {CORR_NON_LEAKAGE_PATH}")
    print(f"Timestamp summary saved to: {TIMESTAMP_SUMMARY_PATH}")

    print("\nQuick view:")
    print(f"Shape: {df.shape}")

    if "has_visual_features" in df.columns:
        print("\nVisual features:")
        print(df["has_visual_features"].value_counts(dropna=False))

    if "has_road_roi_features" in df.columns:
        print("\nRoad ROI features:")
        print(df["has_road_roi_features"].value_counts(dropna=False))

    if "road_condition_mode" in df.columns:
        print("\nRoad condition:")
        print(df["road_condition_mode"].value_counts(dropna=False))

    if key_visual_cols:
        print("\nKey visual feature summary:")
        print(df[key_visual_cols].describe())


if __name__ == "__main__":
    main()