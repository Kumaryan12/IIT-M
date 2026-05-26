from pathlib import Path

import pandas as pd


SAMPLE_FRAME_MAPPING_PATH = Path("outputs/features/sample_frame_mapping.csv")
PROCESSED_FEATURES_PATH = Path("outputs/features/processed_frame_visual_features.csv")
OUTPUT_PATH = Path("outputs/features/sample_visual_features.csv")


NUMERIC_AGG_COLS = [
    "total_vehicles",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "bicycle_count",
    "vehicle_box_area_ratio",
    "average_vehicle_confidence",
    "small_vehicle_count",
    "medium_vehicle_count",
    "large_vehicle_count",
    "heavy_vehicle_count",
    "two_wheeler_count",
    "motor_vehicle_count",
    "non_motor_vehicle_count",
    "heavy_vehicle_ratio",
    "two_wheeler_ratio",
    "motor_vehicle_ratio",
    "car_ratio",
    "truck_ratio",
    "bus_ratio",
    "traffic_load_score",
    "road_condition_confidence",
    "clip_road_dust_score",
    "brightness_mean",
    "contrast_std",
    "brown_pixel_ratio",
    "edge_density",
    "haze_score",
    "visual_dust_score",
    "road_dust_score",
    "dust_traffic_interaction_score",
    "heavy_vehicle_dust_score",
]


def safe_mode(series):
    series = series.dropna()
    if series.empty:
        return ""
    return series.mode().iloc[0]


def main():
    if not SAMPLE_FRAME_MAPPING_PATH.exists():
        print(f"Missing file: {SAMPLE_FRAME_MAPPING_PATH}")
        print("Run extract_frames_from_alignment.py first.")
        return

    if not PROCESSED_FEATURES_PATH.exists():
        print(f"Missing file: {PROCESSED_FEATURES_PATH}")
        print("Run run_processed_frame_feature_extraction.py first.")
        return

    mapping_df = pd.read_csv(SAMPLE_FRAME_MAPPING_PATH)
    features_df = pd.read_csv(PROCESSED_FEATURES_PATH)

    success_df = features_df[features_df["feature_status"] == "success"].copy()

    print("\nInput files:")
    print(f"Sample-frame mapping rows: {len(mapping_df)}")
    print(f"Processed frame feature rows: {len(features_df)}")
    print(f"Successful feature rows: {len(success_df)}")

    # processed_frame_visual_features has source_frame_key.
    # sample_frame_mapping has frame_key.
    merged = mapping_df.merge(
        success_df,
        left_on=["frame_key", "lens_id"],
        right_on=["source_frame_key", "lens_id"],
        how="inner",
        suffixes=("_map", "_feat"),
    )

    print(f"\nMerged mapping + features rows: {len(merged)}")

    if merged.empty:
        print("No rows matched. Check frame_key/source_frame_key columns.")
        return

    grouped_rows = []

    for sample_id, group in merged.groupby("sample_id"):
        out = {
            "sample_id": sample_id,
            "timestamp": group["timestamp"].iloc[0],
            "matched_run_id": safe_mode(group["matched_run_id_map"]),
            "num_lens_frames_used": len(group),
            "lenses_used": ",".join(map(str, sorted(group["lens_id"].unique()))),
        }

        # Overall aggregations across selected lenses
        for col in NUMERIC_AGG_COLS:
            if col in group.columns:
                out[f"{col}_sum"] = group[col].sum()
                out[f"{col}_mean"] = group[col].mean()
                out[f"{col}_max"] = group[col].max()
                out[f"{col}_min"] = group[col].min()

        # Lens-specific features
        for lens_id in sorted(group["lens_id"].unique()):
            lens_group = group[group["lens_id"] == lens_id]

            prefix = f"lens{lens_id}"

            for col in [
                "total_vehicles",
                "car_count",
                "motorcycle_count",
                "bus_count",
                "truck_count",
                "traffic_load_score",
                "vehicle_box_area_ratio",
                "road_dust_score",
                "visual_dust_score",
                "haze_score",
                "brightness_mean",
                "contrast_std",
            ]:
                if col in lens_group.columns:
                    out[f"{prefix}_{col}"] = lens_group[col].mean()

            if "road_condition" in lens_group.columns:
                out[f"{prefix}_road_condition"] = safe_mode(lens_group["road_condition"])

            if "road_dust_level" in lens_group.columns:
                out[f"{prefix}_road_dust_level"] = safe_mode(lens_group["road_dust_level"])

        # Categorical summary
        if "road_condition" in group.columns:
            out["road_condition_mode"] = safe_mode(group["road_condition"])

        if "road_dust_level" in group.columns:
            out["road_dust_level_mode"] = safe_mode(group["road_dust_level"])

        grouped_rows.append(out)

    out_df = pd.DataFrame(grouped_rows)
    out_df = out_df.sort_values("sample_id").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved sample-level visual features to: {OUTPUT_PATH}")
    print(f"Output shape: {out_df.shape}")

    print("\nPreview:")
    preview_cols = [
        "sample_id",
        "timestamp",
        "matched_run_id",
        "num_lens_frames_used",
        "lenses_used",
        "total_vehicles_sum",
        "traffic_load_score_sum",
        "road_dust_score_mean",
        "road_condition_mode",
    ]
    preview_cols = [c for c in preview_cols if c in out_df.columns]
    print(out_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()