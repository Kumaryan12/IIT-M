from pathlib import Path
import argparse

import cv2
import pandas as pd


INPUT_MANIFEST = Path("outputs/features/unique_extracted_frames.csv")
OUTPUT_ROOT = Path("outputs/processed_frames")
OUTPUT_MANIFEST = Path("outputs/features/processed_frame_manifest.csv")


# Moderate crop: keep useful road scene, don't overcrop.
LENS_CROP_CONFIG = {
    1: (0.25, 0.10, 0.70, 0.88),
    4: (0.40, 0.10, 0.90, 0.88),
    6: (0.15, 0.10, 0.85, 0.88),
}


# Rotation chosen from manual preview inspection.
LENS_ROTATION_CONFIG = {
    1: "rot90_counterclockwise",
    4: "rot90_counterclockwise",
    6: "rot90_counterclockwise",
}


# Platform mask after crop + rotation.
# Format: lens_id: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
# This hides the fixed blue roof/camera platform area without cropping away
# useful road/vehicle context above it.
PLATFORM_MASK_CONFIG = {
    1: (0.00, 0.82, 1.00, 1.00),  # keep same
    4: (0.00, 0.95, 1.00, 1.00),  # less aggressive
    6: (0.00, 0.79, 1.00, 1.00),  # more aggressive
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


def rotate_image(image, rotation_name):
    if rotation_name == "rot0":
        return image

    if rotation_name == "rot90_clockwise":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if rotation_name == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)

    if rotation_name == "rot90_counterclockwise":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(f"Unknown rotation name: {rotation_name}")


def apply_platform_mask(image, mask_ratios):
    """
    Applies a black rectangular mask on the processed frame.

    This does not crop the image. It only hides the fixed camera roof/platform
    so YOLO does not detect it as a vehicle.
    """

    h, w = image.shape[:2]

    x1r, y1r, x2r, y2r = mask_ratios

    x1 = int(x1r * w)
    y1 = int(y1r * h)
    x2 = int(x2r * w)
    y2 = int(y2r * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    masked = image.copy()
    masked[y1:y2, x1:x2] = (0, 0, 0)

    return masked, x1, y1, x2, y2


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lenses",
        nargs="+",
        type=int,
        default=[1, 4, 6],
        help="Lens IDs to preprocess. Example: --lenses 1 4 6"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional test limit. Example: --limit 10"
    )

    parser.add_argument(
        "--no-mask",
        action="store_true",
        help="Disable platform mask for debugging."
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
        print("No extracted frames found for selected lenses.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)

    output_rows = []

    print("\nLens preprocessing setup:")
    print(f"Frames selected: {len(df)}")
    print(f"Lenses selected: {args.lenses}")
    print(f"Apply platform mask: {not args.no_mask}")
    print(f"Output folder: {OUTPUT_ROOT}")

    for _, row in df.iterrows():
        frame_key = str(row["frame_key"])
        lens_id = int(row["lens_id"])
        frame_path = Path(str(row["frame_path"]))

        if lens_id not in LENS_CROP_CONFIG:
            print(f"Skipping lens {lens_id}: no crop config.")
            continue

        if lens_id not in LENS_ROTATION_CONFIG:
            print(f"Skipping lens {lens_id}: no rotation config.")
            continue

        if not frame_path.exists():
            print(f"Missing frame: {frame_path}")
            continue

        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read frame: {frame_path}")
            continue

        crop_ratios = LENS_CROP_CONFIG[lens_id]
        rotation_name = LENS_ROTATION_CONFIG[lens_id]

        cropped, crop_x1, crop_y1, crop_x2, crop_y2 = crop_image(
            image,
            crop_ratios
        )

        rotated = rotate_image(cropped, rotation_name)

        mask_applied = False
        mask_x1 = mask_y1 = mask_x2 = mask_y2 = None
        mask_ratios = (None, None, None, None)

        if not args.no_mask and lens_id in PLATFORM_MASK_CONFIG:
            mask_ratios = PLATFORM_MASK_CONFIG[lens_id]
            processed, mask_x1, mask_y1, mask_x2, mask_y2 = apply_platform_mask(
                rotated,
                mask_ratios
            )
            mask_applied = True
        else:
            processed = rotated

        processed_frame_key = f"{frame_key}_processed_lens{lens_id}"

        output_path = (
            OUTPUT_ROOT
            / str(row["matched_run_id"])
            / f"lens{lens_id}"
            / f"{processed_frame_key}.jpg"
        )

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Writes only to outputs/processed_frames, never to raw videos.
        cv2.imwrite(str(output_path), processed)

        print(f"\nProcessed: {frame_key}")
        print(f"  Lens: {lens_id}")
        print(f"  Source extracted frame: {frame_path}")
        print(f"  Output processed frame: {output_path}")
        print(f"  Crop pixels: x1={crop_x1}, y1={crop_y1}, x2={crop_x2}, y2={crop_y2}")
        print(f"  Rotation: {rotation_name}")
        print(f"  Mask applied: {mask_applied}")
        if mask_applied:
            print(f"  Mask pixels: x1={mask_x1}, y1={mask_y1}, x2={mask_x2}, y2={mask_y2}")

        output_rows.append({
            "processed_frame_key": processed_frame_key,
            "source_frame_key": frame_key,
            "matched_run_id": row["matched_run_id"],
            "video_offset_sec": row["video_offset_sec"],
            "lens_id": lens_id,

            "source_frame_path": str(frame_path),
            "processed_frame_path": str(output_path),

            "crop_x1": crop_x1,
            "crop_y1": crop_y1,
            "crop_x2": crop_x2,
            "crop_y2": crop_y2,

            "crop_x1_ratio": crop_ratios[0],
            "crop_y1_ratio": crop_ratios[1],
            "crop_x2_ratio": crop_ratios[2],
            "crop_y2_ratio": crop_ratios[3],

            "rotation_name": rotation_name,

            "mask_applied": mask_applied,
            "mask_x1": mask_x1,
            "mask_y1": mask_y1,
            "mask_x2": mask_x2,
            "mask_y2": mask_y2,
            "mask_x1_ratio": mask_ratios[0],
            "mask_y1_ratio": mask_ratios[1],
            "mask_x2_ratio": mask_ratios[2],
            "mask_y2_ratio": mask_ratios[3],

            "preprocess_status": "success",
        })

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_MANIFEST, index=False)

    print("\nDone.")
    print(f"Processed frame manifest saved to: {OUTPUT_MANIFEST}")

    if not out_df.empty:
        print("\nProcessed rows by lens:")
        print(out_df["lens_id"].value_counts())


if __name__ == "__main__":
    main()