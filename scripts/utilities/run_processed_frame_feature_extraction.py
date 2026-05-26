import argparse
from pathlib import Path

import pandas as pd

from src.pipeline import ImageFeaturePipeline


PROCESSED_FRAME_MANIFEST = Path("outputs/features/processed_frame_manifest.csv")
OUTPUT_DIR = Path("outputs/features")
OUTPUT_PATH = OUTPUT_DIR / "processed_frame_visual_features.csv"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lenses",
        nargs="+",
        type=int,
        default=None,
        help="Optional lens filter. Example: --lenses 1 4 6"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing. Example: --limit 5"
    )

    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Disable saving annotated YOLO images."
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=5,
        help="Save output CSV after every N processed frames."
    )

    args = parser.parse_args()

    if not PROCESSED_FRAME_MANIFEST.exists():
        print(f"Processed frame manifest not found: {PROCESSED_FRAME_MANIFEST}")
        print("Run apply_lens_preprocessing.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(PROCESSED_FRAME_MANIFEST)

    required_cols = {
        "processed_frame_key",
        "source_frame_key",
        "matched_run_id",
        "video_offset_sec",
        "lens_id",
        "processed_frame_path",
        "preprocess_status",
    }

    missing_cols = required_cols - set(manifest_df.columns)

    if missing_cols:
        print("Processed frame manifest is missing required columns:")
        print(missing_cols)
        return

    frame_df = manifest_df[manifest_df["preprocess_status"] == "success"].copy()

    if args.lenses is not None:
        frame_df = frame_df[frame_df["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        frame_df = frame_df.head(args.limit).copy()

    print("\nProcessed frame feature extraction setup:")
    print(f"Total processed frames available: {len(manifest_df[manifest_df['preprocess_status'] == 'success'])}")
    print(f"Frames selected now: {len(frame_df)}")
    print(f"Lens filter: {args.lenses}")
    print(f"Save annotated images: {not args.no_annotated}")

    output_rows = []
    completed_keys = set()

    if OUTPUT_PATH.exists():
        old_df = pd.read_csv(OUTPUT_PATH)

        if "processed_frame_key" in old_df.columns:
            completed_keys = set(old_df["processed_frame_key"].astype(str))
            output_rows = old_df.to_dict(orient="records")

        print(f"\nLoaded existing output file: {OUTPUT_PATH}")
        print(f"Already processed frames: {len(completed_keys)}")

    pipeline = ImageFeaturePipeline()

    processed_since_save = 0

    for _, row in frame_df.iterrows():
        processed_frame_key = str(row["processed_frame_key"])

        if processed_frame_key in completed_keys:
            print(f"Skipping already processed frame: {processed_frame_key}")
            continue

        frame_path = Path(str(row["processed_frame_path"]))

        print(f"\nProcessing processed frame: {processed_frame_key}")
        print(f"Frame path: {frame_path}")

        if not frame_path.exists():
            print("  Frame file not found. Skipping.")

            output_rows.append({
                "processed_frame_key": processed_frame_key,
                "source_frame_key": row["source_frame_key"],
                "matched_run_id": row["matched_run_id"],
                "video_offset_sec": row["video_offset_sec"],
                "lens_id": row["lens_id"],
                "processed_frame_path": str(frame_path),
                "feature_status": "failed",
                "feature_error": "processed_frame_not_found",
            })

            continue

        try:
            features = pipeline.process_image(
                image_path=frame_path,
                save_annotated=not args.no_annotated
            )

            features.update({
                "processed_frame_key": processed_frame_key,
                "source_frame_key": row["source_frame_key"],
                "matched_run_id": row["matched_run_id"],
                "video_offset_sec": row["video_offset_sec"],
                "lens_id": row["lens_id"],
                "processed_frame_path": str(frame_path),

                "crop_x1": row.get("crop_x1", None),
                "crop_y1": row.get("crop_y1", None),
                "crop_x2": row.get("crop_x2", None),
                "crop_y2": row.get("crop_y2", None),
                "rotation_name": row.get("rotation_name", None),

                "feature_status": "success",
                "feature_error": "",
            })

            output_rows.append(features)
            completed_keys.add(processed_frame_key)

            print("  Success.")
            print(f"  total_vehicles: {features.get('total_vehicles')}")
            print(f"  traffic_load_score: {features.get('traffic_load_score')}")
            print(f"  road_condition: {features.get('road_condition')}")
            print(f"  road_dust_score: {features.get('road_dust_score')}")

        except Exception as e:
            print(f"  Error: {e}")

            output_rows.append({
                "processed_frame_key": processed_frame_key,
                "source_frame_key": row["source_frame_key"],
                "matched_run_id": row["matched_run_id"],
                "video_offset_sec": row["video_offset_sec"],
                "lens_id": row["lens_id"],
                "processed_frame_path": str(frame_path),
                "feature_status": "failed",
                "feature_error": str(e),
            })

        processed_since_save += 1

        if processed_since_save >= args.checkpoint_every:
            out_df = pd.DataFrame(output_rows)
            out_df.to_csv(OUTPUT_PATH, index=False)
            print(f"\nCheckpoint saved: {OUTPUT_PATH}")
            processed_since_save = 0

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved processed frame visual features to: {OUTPUT_PATH}")

    if not out_df.empty and "feature_status" in out_df.columns:
        print("\nFeature extraction status counts:")
        print(out_df["feature_status"].value_counts())

    if not out_df.empty and "lens_id" in out_df.columns:
        print("\nRows by lens:")
        print(out_df["lens_id"].value_counts())


if __name__ == "__main__":
    main()