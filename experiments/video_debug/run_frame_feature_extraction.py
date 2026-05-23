import argparse
from pathlib import Path

import pandas as pd

from src.pipeline import ImageFeaturePipeline


UNIQUE_FRAME_MANIFEST = Path("outputs/features/unique_extracted_frames.csv")
OUTPUT_DIR = Path("outputs/features")
OUTPUT_PATH = OUTPUT_DIR / "frame_visual_features.csv"


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
        help="Save output after every N processed frames."
    )

    args = parser.parse_args()

    if not UNIQUE_FRAME_MANIFEST.exists():
        print(f"Frame manifest not found: {UNIQUE_FRAME_MANIFEST}")
        print("Run extract_frames_from_alignment.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(UNIQUE_FRAME_MANIFEST)

    required_cols = {
        "frame_key",
        "matched_run_id",
        "video_offset_sec",
        "lens_id",
        "frame_path",
        "extract_status",
    }

    missing_cols = required_cols - set(manifest_df.columns)

    if missing_cols:
        print("Manifest is missing required columns:")
        print(missing_cols)
        return

    frame_df = manifest_df[manifest_df["extract_status"] == "success"].copy()

    if args.lenses is not None:
        frame_df = frame_df[frame_df["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        frame_df = frame_df.head(args.limit).copy()

    print("\nFrame feature extraction setup:")
    print(f"Total successful extracted frames available: {len(manifest_df[manifest_df['extract_status'] == 'success'])}")
    print(f"Frames selected now: {len(frame_df)}")
    print(f"Lens filter: {args.lenses}")
    print(f"Save annotated images: {not args.no_annotated}")

    # Resume if output already exists
    output_rows = []
    completed_frame_keys = set()

    if OUTPUT_PATH.exists():
        old_df = pd.read_csv(OUTPUT_PATH)

        if "frame_key" in old_df.columns:
            completed_frame_keys = set(old_df["frame_key"].astype(str))
            output_rows = old_df.to_dict(orient="records")

        print(f"\nLoaded existing frame features: {OUTPUT_PATH}")
        print(f"Already processed frames: {len(completed_frame_keys)}")

    pipeline = ImageFeaturePipeline()

    processed_since_save = 0

    for idx, row in frame_df.iterrows():
        frame_key = str(row["frame_key"])

        if frame_key in completed_frame_keys:
            print(f"Skipping already processed frame: {frame_key}")
            continue

        frame_path = Path(str(row["frame_path"]))

        print(f"\nProcessing frame: {frame_key}")
        print(f"Frame path: {frame_path}")

        if not frame_path.exists():
            print("  Frame file not found. Skipping.")
            continue

        try:
            features = pipeline.process_image(
                image_path=frame_path,
                save_annotated=not args.no_annotated
            )

            # Add video/frame metadata
            features.update({
                "frame_key": frame_key,
                "matched_run_id": row["matched_run_id"],
                "video_offset_sec": row["video_offset_sec"],
                "lens_id": row["lens_id"],
                "frame_path": str(frame_path),
                "feature_status": "success",
                "feature_error": "",
            })

            output_rows.append(features)
            completed_frame_keys.add(frame_key)

            print("  Success.")
            print(f"  total_vehicles: {features.get('total_vehicles')}")
            print(f"  traffic_load_score: {features.get('traffic_load_score')}")
            print(f"  road_condition: {features.get('road_condition')}")
            print(f"  road_dust_score: {features.get('road_dust_score')}")

        except Exception as e:
            print(f"  Error: {e}")

            output_rows.append({
                "frame_key": frame_key,
                "matched_run_id": row["matched_run_id"],
                "video_offset_sec": row["video_offset_sec"],
                "lens_id": row["lens_id"],
                "frame_path": str(frame_path),
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
    print(f"Saved frame visual features to: {OUTPUT_PATH}")

    if not out_df.empty and "feature_status" in out_df.columns:
        print("\nFeature extraction status counts:")
        print(out_df["feature_status"].value_counts())

    if not out_df.empty and "lens_id" in out_df.columns:
        print("\nRows by lens:")
        print(out_df["lens_id"].value_counts())


if __name__ == "__main__":
    main()