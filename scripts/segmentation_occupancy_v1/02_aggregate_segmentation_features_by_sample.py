from pathlib import Path

import pandas as pd


MAPPING_CSV = Path("outputs/features/sample_frame_mapping.csv")
SEG_FEATURES_CSV = Path("outputs/features/segmentation_occupancy_features_v1.csv")

OUTPUT_CSV = Path("outputs/features/sample_segmentation_occupancy_features_v1.csv")


AGG_COLS = [
    "seg_total_vehicle_instances",
    "seg_vehicle_mask_area",
    "seg_vehicle_occupancy_ratio",
    "seg_average_vehicle_confidence",

    "seg_car_count",
    "seg_motorcycle_count",
    "seg_bus_count",
    "seg_truck_count",
    "seg_bicycle_count",

    "seg_car_mask_area",
    "seg_motorcycle_mask_area",
    "seg_bus_mask_area",
    "seg_truck_mask_area",
    "seg_bicycle_mask_area",

    "seg_car_occupancy_ratio",
    "seg_heavy_vehicle_mask_area",
    "seg_two_wheeler_mask_area",
    "seg_heavy_vehicle_occupancy_ratio",
    "seg_two_wheeler_occupancy_ratio",

    "seg_car_area_share",
    "seg_heavy_vehicle_area_share",
    "seg_two_wheeler_area_share",
]


def main():
    if not MAPPING_CSV.exists():
        print(f"Missing mapping file: {MAPPING_CSV}")
        return

    if not SEG_FEATURES_CSV.exists():
        print(f"Missing segmentation feature file: {SEG_FEATURES_CSV}")
        return

    mapping = pd.read_csv(MAPPING_CSV)
    seg = pd.read_csv(SEG_FEATURES_CSV)

    print("\nInput files:")
    print(f"Sample-frame mapping rows: {len(mapping)}")
    print(f"Segmentation feature rows: {len(seg)}")

    seg = seg[seg["feature_status"] == "success"].copy()

    print(f"Successful segmentation rows: {len(seg)}")

    # Match mapping frame_key to segmentation processed_frame_key/source_frame_key
    if "source_frame_key" in seg.columns:
        merge_left_key = "frame_key"
        merge_right_key = "source_frame_key"
    else:
        merge_left_key = "frame_key"
        merge_right_key = "processed_frame_key"

    merged = mapping.merge(
        seg,
        left_on=merge_left_key,
        right_on=merge_right_key,
        how="inner",
        suffixes=("_map", "_seg"),
    )

    print(f"\nMerged rows: {len(merged)}")

    if merged.empty:
        print("No rows matched.")
        print("Check frame_key, source_frame_key, processed_frame_key columns.")
        return

    available_agg_cols = [
        c for c in AGG_COLS
        if c in merged.columns
    ]

    grouped = (
        merged
        .groupby(["sample_id", "timestamp", "matched_run_id_map"], as_index=False)
        .agg(
            num_seg_lens_frames_used=("lens_id_map", "nunique"),
            seg_lenses_used=(
                "lens_id_map",
                lambda x: ",".join(map(str, sorted(set(x))))
            ),
            **{
                f"{col}_mean": (col, "mean")
                for col in available_agg_cols
            },
            **{
                f"{col}_sum": (col, "sum")
                for col in available_agg_cols
            },
            **{
                f"{col}_max": (col, "max")
                for col in available_agg_cols
            },
        )
    )

    grouped = grouped.rename(
        columns={
            "matched_run_id_map": "matched_run_id"
        }
    )

    # Lens-specific features
    lens_rows = []

    for (sample_id, timestamp), group in merged.groupby(["sample_id", "timestamp"]):
        row = {
            "sample_id": sample_id,
            "timestamp": timestamp,
        }

        for _, r in group.iterrows():
            lens_id = int(r["lens_id_map"])
            prefix = f"lens{lens_id}_"

            for col in available_agg_cols:
                row[prefix + col] = r[col]

        lens_rows.append(row)

    lens_df = pd.DataFrame(lens_rows)

    final = grouped.merge(
        lens_df,
        on=["sample_id", "timestamp"],
        how="left",
    )

    final.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Output shape: {final.shape}")

    print("\nPreview:")
    preview_cols = [
        "sample_id",
        "timestamp",
        "matched_run_id",
        "num_seg_lens_frames_used",
        "seg_lenses_used",
        "seg_vehicle_occupancy_ratio_mean",
        "seg_vehicle_occupancy_ratio_sum",
        "seg_total_vehicle_instances_sum",
        "seg_heavy_vehicle_occupancy_ratio_mean",
        "seg_two_wheeler_occupancy_ratio_mean",
    ]

    preview_cols = [
        c for c in preview_cols
        if c in final.columns
    ]

    print(final[preview_cols].head(15).to_string(index=False))


if __name__ == "__main__":
    main()