from pathlib import Path

import pandas as pd


SENSOR_OSM_CANDIDATES = [
    Path("outputs/features/Sensor_data_with_osm.csv"),
    Path("outputs/features/clean_sensor_data_with_osm.csv"),
    Path("outputs/features/clean_sensor_data.csv"),
]

VISUAL_FEATURES_PATH = Path("outputs/features/sample_visual_features_roi.csv")

OUTPUT_DIR = Path("outputs/features")
OUTPUT_PATH = OUTPUT_DIR / "master_sensor_osm_visual_roi_dataset.csv"


def find_first_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def main():
    sensor_path = find_first_existing_path(SENSOR_OSM_CANDIDATES)

    if sensor_path is None:
        print("No sensor/OSM file found.")
        for path in SENSOR_OSM_CANDIDATES:
            print(f"  {path}")
        return

    if not VISUAL_FEATURES_PATH.exists():
        print(f"Visual ROI features file not found: {VISUAL_FEATURES_PATH}")
        print("Run aggregate_video_features_by_sample_roi.py first.")
        return

    sensor_df = pd.read_csv(sensor_path)
    visual_df = pd.read_csv(VISUAL_FEATURES_PATH)

    if "sample_id" not in sensor_df.columns:
        print(f"'sample_id' missing in sensor file: {sensor_path}")
        return

    if "sample_id" not in visual_df.columns:
        print(f"'sample_id' missing in visual ROI file: {VISUAL_FEATURES_PATH}")
        return

    print("\nInput files:")
    print(f"Sensor/OSM file: {sensor_path}")
    print(f"Sensor/OSM shape: {sensor_df.shape}")
    print(f"Visual ROI features file: {VISUAL_FEATURES_PATH}")
    print(f"Visual ROI features shape: {visual_df.shape}")

    visual_df_clean = visual_df.drop(columns=["timestamp"], errors="ignore")

    merged_df = sensor_df.merge(
        visual_df_clean,
        on="sample_id",
        how="left",
        suffixes=("", "_visual")
    )

    if "num_lens_frames_used" in merged_df.columns:
        merged_df["has_visual_features"] = merged_df["num_lens_frames_used"].notna()
    else:
        merged_df["has_visual_features"] = False

    if "num_road_roi_lens_frames_used" in merged_df.columns:
        merged_df["has_road_roi_features"] = merged_df["num_road_roi_lens_frames_used"].notna()
    else:
        merged_df["has_road_roi_features"] = False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("\nROI master merge completed.")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Output shape: {merged_df.shape}")

    print("\nVisual feature availability:")
    print(merged_df["has_visual_features"].value_counts(dropna=False))

    print("\nRoad ROI feature availability:")
    print(merged_df["has_road_roi_features"].value_counts(dropna=False))

    if "matched_run_id" in merged_df.columns:
        print("\nMatched run counts:")
        print(merged_df[merged_df["has_visual_features"]]["matched_run_id"].value_counts())

    print("\nPreview:")
    preview_cols = [
        "sample_id",
        "timestamp",
        "latitude",
        "longitude",
        "has_visual_features",
        "has_road_roi_features",
        "matched_run_id",
        "num_lens_frames_used",
        "lenses_used",
        "num_road_roi_lens_frames_used",
        "road_roi_lenses_used",
        "total_vehicles_sum",
        "traffic_load_score_sum",
        "road_roi_road_dust_score_mean",
        "road_condition_mode",
        "road_dust_level_mode",
    ]

    preview_cols = [col for col in preview_cols if col in merged_df.columns]
    print(merged_df[preview_cols].head(15).to_string(index=False))

    print("\nNote:")
    print("This ROI master dataset is preferred over the previous non-ROI visual dataset.")
    print("Use PM/count columns only for density target generation, not as model inputs.")


if __name__ == "__main__":
    main()