from pathlib import Path

import pandas as pd


# Prefer the OSM-enriched sensor file if available.
SENSOR_OSM_CANDIDATES = [
    Path("outputs/features/Sensor_data_with_osm.csv"),
    Path("outputs/features/clean_sensor_data_with_osm.csv"),
    Path("outputs/features/clean_sensor_data.csv"),
]

VISUAL_FEATURES_PATH = Path("outputs/features/sample_visual_features.csv")

OUTPUT_DIR = Path("outputs/features")
OUTPUT_PATH = OUTPUT_DIR / "master_sensor_osm_visual_dataset.csv"


def find_first_existing_path(paths):
    for path in paths:
        if path.exists():
            return path
    return None


def main():
    sensor_path = find_first_existing_path(SENSOR_OSM_CANDIDATES)

    if sensor_path is None:
        print("No sensor/OSM file found.")
        print("Expected one of:")
        for path in SENSOR_OSM_CANDIDATES:
            print(f"  {path}")
        return

    if not VISUAL_FEATURES_PATH.exists():
        print(f"Visual features file not found: {VISUAL_FEATURES_PATH}")
        print("Run aggregate_video_features_by_sample.py first.")
        return

    sensor_df = pd.read_csv(sensor_path)
    visual_df = pd.read_csv(VISUAL_FEATURES_PATH)

    if "sample_id" not in sensor_df.columns:
        print(f"'sample_id' missing in sensor file: {sensor_path}")
        print("Available columns:")
        print(list(sensor_df.columns))
        return

    if "sample_id" not in visual_df.columns:
        print(f"'sample_id' missing in visual features file: {VISUAL_FEATURES_PATH}")
        print("Available columns:")
        print(list(visual_df.columns))
        return

    print("\nInput files:")
    print(f"Sensor/OSM file: {sensor_path}")
    print(f"Sensor/OSM shape: {sensor_df.shape}")
    print(f"Visual features file: {VISUAL_FEATURES_PATH}")
    print(f"Visual features shape: {visual_df.shape}")

    # Avoid duplicate timestamp columns causing confusion.
    visual_cols_to_drop = []
    for col in ["timestamp"]:
        if col in visual_df.columns:
            visual_cols_to_drop.append(col)

    visual_df_clean = visual_df.drop(columns=visual_cols_to_drop)

    merged_df = sensor_df.merge(
        visual_df_clean,
        on="sample_id",
        how="left",
        suffixes=("", "_visual")
    )

    # Add match flag
    if "num_lens_frames_used" in merged_df.columns:
        merged_df["has_visual_features"] = merged_df["num_lens_frames_used"].notna()
    else:
        merged_df["has_visual_features"] = False

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    merged_df.to_csv(OUTPUT_PATH, index=False)

    print("\nMerge completed.")
    print(f"Output file: {OUTPUT_PATH}")
    print(f"Output shape: {merged_df.shape}")

    print("\nVisual feature availability:")
    print(merged_df["has_visual_features"].value_counts())

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
        "matched_run_id",
        "num_lens_frames_used",
        "lenses_used",
        "total_vehicles_sum",
        "traffic_load_score_sum",
        "road_dust_score_mean",
        "road_condition_mode",
    ]

    preview_cols = [col for col in preview_cols if col in merged_df.columns]
    print(merged_df[preview_cols].head(15).to_string(index=False))

    print("\nNote:")
    print("PM/count sensor columns are still present for target generation.")
    print("When training image/video/OSM → density, exclude raw PM/count columns from model inputs.")


if __name__ == "__main__":
    main()