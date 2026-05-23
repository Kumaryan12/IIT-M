from pathlib import Path
import argparse

import cv2
import pandas as pd


INPUT_MANIFEST = Path("outputs/features/unique_extracted_frames.csv")
OUTPUT_ROOT = Path("outputs/cropped_frames")
OUTPUT_MANIFEST = Path("outputs/features/cropped_frame_manifest.csv")


# Crop format:
# lens_id: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
#
# Selected by visual inspection:
# Lens 1 -> middle_road_zone
# Lens 4 -> right_road_zone
# Lens 6 -> center_wide
LENS_CROP_CONFIG = {
    1: (0.25, 0.15, 0.70, 0.95),
    4: (0.40, 0.15, 0.90, 0.95),
    6: (0.15, 0.10, 0.85, 0.95),
}


def crop_image(image, crop_ratios):
    h, w = image.shape[:2]

    x1r, y1r, x2r, y2r = crop_ratios

    x1 = int(x1r * w)
    y1 = int(y1r * h)
    x2 = int(x2r * w)
    y2 = int(y2r * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    cropped = image[y1:y2, x1:x2]

    return cropped, x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lenses",
        nargs="+",
        type=int,
        default=[1, 4, 6],
        help="Lens IDs to crop. Example: --lenses 1 4 6"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional test limit. Example: --limit 10"
    )

    args = parser.parse_args()

    if not INPUT_MANIFEST.exists():
        print(f"Input manifest not found: {INPUT_MANIFEST}")
        print("Run extract_frames_from_alignment.py first.")
        return

    df = pd.read_csv(INPUT_MANIFEST)

    if "extract_status" in df.columns:
        df = df[df["extract_status"] == "success"].copy()

    df = df[df["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    if df.empty:
        print("No frames found for selected lenses.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []

    print("\nCropping setup:")
    print(f"Frames selected: {len(df)}")
    print(f"Lenses selected: {args.lenses}")
    print(f"Output folder: {OUTPUT_ROOT}")

    for _, row in df.iterrows():
        frame_key = str(row["frame_key"])
        lens_id = int(row["lens_id"])
        frame_path = Path(str(row["frame_path"]))

        if lens_id not in LENS_CROP_CONFIG:
            print(f"Skipping lens {lens_id}: no crop config defined.")
            continue

        if not frame_path.exists():
            print(f"Missing frame: {frame_path}")
            continue

        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read image: {frame_path}")
            continue

        crop_ratios = LENS_CROP_CONFIG[lens_id]

        cropped, x1, y1, x2, y2 = crop_image(image, crop_ratios)

        cropped_frame_key = f"{frame_key}_crop_lens{lens_id}"

        output_path = (
            OUTPUT_ROOT
            / str(row["matched_run_id"])
            / f"lens{lens_id}"
            / f"{cropped_frame_key}.jpg"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        cv2.imwrite(str(output_path), cropped)

        print(f"\nCropped: {frame_key}")
        print(f"  Lens: {lens_id}")
        print(f"  Source: {frame_path}")
        print(f"  Output: {output_path}")
        print(f"  Crop pixels: x1={x1}, y1={y1}, x2={x2}, y2={y2}")

        output_rows.append({
            "cropped_frame_key": cropped_frame_key,
            "source_frame_key": frame_key,
            "matched_run_id": row["matched_run_id"],
            "video_offset_sec": row["video_offset_sec"],
            "lens_id": lens_id,
            "source_frame_path": str(frame_path),
            "cropped_frame_path": str(output_path),
            "crop_x1": x1,
            "crop_y1": y1,
            "crop_x2": x2,
            "crop_y2": y2,
            "crop_x1_ratio": crop_ratios[0],
            "crop_y1_ratio": crop_ratios[1],
            "crop_x2_ratio": crop_ratios[2],
            "crop_y2_ratio": crop_ratios[3],
            "crop_status": "success",
        })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_MANIFEST, index=False)

    print("\nDone.")
    print(f"Cropped frame manifest saved to: {OUTPUT_MANIFEST}")

    if not out_df.empty:
        print("\nCropped rows by lens:")
        print(out_df["lens_id"].value_counts())


if __name__ == "__main__":
    main()