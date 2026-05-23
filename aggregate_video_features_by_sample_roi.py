from pathlib import Path

import pandas as pd


SAMPLE_FRAME_MAPPING_PATH = Path("outputs/features/sample_frame_mapping.csv")
ROI_FEATURES_PATH = Path("outputs/features/processed_frame_visual_features_roi.csv")
OUTPUT_PATH = Path("outputs/features/sample_visual_features_roi.csv")


VEHICLE_NUMERIC_COLS = [
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
]


ROAD_NUMERIC_COLS = [
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
    series = series[series.astype(str).str.len() > 0]

    if series.empty:
        return ""

    return series.mode().iloc[0]


def add_numeric_aggs(out, group, cols, prefix):
    for col in cols:
        if col not in group.columns:
            continue

        numeric_series = pd.to_numeric(group[col], errors="coerce")

        out[f"{prefix}{col}_sum"] = numeric_series.sum()
        out[f"{prefix}{col}_mean"] = numeric_series.mean()
        out[f"{prefix}{col}_max"] = numeric_series.max()
        out[f"{prefix}{col}_min"] = numeric_series.min()


def main():
    if not SAMPLE_FRAME_MAPPING_PATH.exists():
        print(f"Missing file: {SAMPLE_FRAME_MAPPING_PATH}")
        print("Run extract_frames_from_alignment.py first.")
        return

    if not ROI_FEATURES_PATH.exists():
        print(f"Missing file: {ROI_FEATURES_PATH}")
        print("Run run_processed_frame_feature_extraction_with_road_roi.py first.")
        return

    mapping_df = pd.read_csv(SAMPLE_FRAME_MAPPING_PATH)
    features_df = pd.read_csv(ROI_FEATURES_PATH)

    success_df = features_df[features_df["feature_status"] == "success"].copy()

    print("\nInput files:")
    print(f"Sample-frame mapping rows: {len(mapping_df)}")
    print(f"ROI feature rows: {len(features_df)}")
    print(f"Successful ROI feature rows: {len(success_df)}")

    merged = mapping_df.merge(
        success_df,
        left_on=["frame_key", "lens_id"],
        right_on=["source_frame_key", "lens_id"],
        how="inner",
        suffixes=("_map", "_feat"),
    )

    print(f"\nMerged mapping + ROI features rows: {len(merged)}")

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

        # 1. Vehicle features from all selected lenses
        add_numeric_aggs(
            out=out,
            group=group,
            cols=VEHICLE_NUMERIC_COLS,
            prefix=""
        )

        # 2. Road/dust features only from lenses where road ROI is available
        if "road_roi_available" in group.columns:
            road_group = group[group["road_roi_available"] == True].copy()
        else:
            road_group = pd.DataFrame()

        out["num_road_roi_lens_frames_used"] = len(road_group)

        if not road_group.empty:
            out["road_roi_lenses_used"] = ",".join(
                map(str, sorted(road_group["lens_id"].unique()))
            )

            add_numeric_aggs(
                out=out,
                group=road_group,
                cols=ROAD_NUMERIC_COLS,
                prefix="road_roi_"
            )

            out["road_condition_mode"] = safe_mode(road_group["road_condition"])
            out["road_dust_level_mode"] = safe_mode(road_group["road_dust_level"])
        else:
            out["road_roi_lenses_used"] = ""
            out["road_condition_mode"] = ""
            out["road_dust_level_mode"] = ""

        # 3. Lens-specific vehicle features for all lenses
        for lens_id in sorted(group["lens_id"].unique()):
            lens_group = group[group["lens_id"] == lens_id]
            lens_prefix = f"lens{lens_id}_"

            for col in [
                "total_vehicles",
                "car_count",
                "motorcycle_count",
                "bus_count",
                "truck_count",
                "traffic_load_score",
                "vehicle_box_area_ratio",
                "average_vehicle_confidence",
            ]:
                if col in lens_group.columns:
                    out[f"{lens_prefix}{col}"] = pd.to_numeric(
                        lens_group[col],
                        errors="coerce"
                    ).mean()

        # 4. Lens-specific road ROI features only for ROI lenses
        for lens_id in sorted(road_group["lens_id"].unique()) if not road_group.empty else []:
            lens_road_group = road_group[road_group["lens_id"] == lens_id]
            lens_prefix = f"lens{lens_id}_road_roi_"

            for col in [
                "road_dust_score",
                "visual_dust_score",
                "clip_road_dust_score",
                "haze_score",
                "brightness_mean",
                "contrast_std",
                "brown_pixel_ratio",
                "edge_density",
                "road_condition_confidence",
            ]:
                if col in lens_road_group.columns:
                    out[f"{lens_prefix}{col}"] = pd.to_numeric(
                        lens_road_group[col],
                        errors="coerce"
                    ).mean()

            out[f"lens{lens_id}_road_condition"] = safe_mode(lens_road_group["road_condition"])
            out[f"lens{lens_id}_road_dust_level"] = safe_mode(lens_road_group["road_dust_level"])

        grouped_rows.append(out)

    out_df = pd.DataFrame(grouped_rows)
    out_df = out_df.sort_values("sample_id").reset_index(drop=True)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved ROI-aware sample-level visual features to: {OUTPUT_PATH}")
    print(f"Output shape: {out_df.shape}")

    print("\nPreview:")
    preview_cols = [
        "sample_id",
        "timestamp",
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
    preview_cols = [c for c in preview_cols if c in out_df.columns]
    print(out_df[preview_cols].head(10).to_string(index=False))


if __name__ == "__main__":
    main()