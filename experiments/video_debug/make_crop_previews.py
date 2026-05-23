from pathlib import Path
import argparse

import cv2
import pandas as pd


FRAME_MANIFEST = Path("outputs/features/unique_extracted_frames.csv")
OUTPUT_DIR = Path("outputs/crop_previews")


# Crop format:
# crop_name: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
#
# These are generic candidates.
# We will inspect outputs and choose the best crop per lens later.
CROP_CANDIDATES = {
    "full_reference": (0.00, 0.00, 1.00, 1.00),

    # Remove extreme fisheye border and some sky/rig
    "center_wide": (0.15, 0.10, 0.85, 0.95),

    # More road-focused candidates
    "left_road_zone": (0.15, 0.15, 0.60, 0.95),
    "middle_road_zone": (0.25, 0.15, 0.70, 0.95),
    "right_road_zone": (0.40, 0.15, 0.90, 0.95),

    # Lower region often contains road/vehicles
    "lower_wide": (0.10, 0.40, 0.90, 1.00),

    # Avoid sky-heavy upper area
    "no_sky_middle": (0.20, 0.25, 0.75, 0.95),
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

    return image[y1:y2, x1:x2]


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lenses",
        nargs="+",
        type=int,
        default=[1, 4, 6],
        help="Lens IDs to preview. Example: --lenses 1 4 6"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of frames to preview."
    )

    args = parser.parse_args()

    if not FRAME_MANIFEST.exists():
        print(f"Frame manifest not found: {FRAME_MANIFEST}")
        print("Run extract_frames_from_alignment.py first.")
        return

    df = pd.read_csv(FRAME_MANIFEST)

    if "extract_status" in df.columns:
        df = df[df["extract_status"] == "success"].copy()

    df = df[df["lens_id"].isin(args.lenses)].copy()
    df = df.head(args.limit)

    if df.empty:
        print("No matching extracted frames found.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    preview_rows = []

    for _, row in df.iterrows():
        frame_key = str(row["frame_key"])
        lens_id = int(row["lens_id"])
        frame_path = Path(str(row["frame_path"]))

        if not frame_path.exists():
            print(f"Missing frame: {frame_path}")
            continue

        image = cv2.imread(str(frame_path))

        if image is None:
            print(f"Could not read image: {frame_path}")
            continue

        print(f"\nCreating crop previews for: {frame_key}")

        for crop_name, ratios in CROP_CANDIDATES.items():
            cropped = crop_image(image, ratios)

            output_path = (
                OUTPUT_DIR
                / f"lens{lens_id}"
                / f"{frame_key}_{crop_name}.jpg"
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), cropped)

            preview_rows.append({
                "frame_key": frame_key,
                "lens_id": lens_id,
                "crop_name": crop_name,
                "crop_ratios": ratios,
                "source_frame_path": str(frame_path),
                "crop_preview_path": str(output_path),
                "crop_height": cropped.shape[0],
                "crop_width": cropped.shape[1],
            })

            print(f"  Saved: {output_path}")

    preview_df = pd.DataFrame(preview_rows)
    preview_csv = OUTPUT_DIR / "crop_preview_manifest.csv"
    preview_df.to_csv(preview_csv, index=False)

    print("\nDone.")
    print(f"Crop previews saved in: {OUTPUT_DIR}")
    print(f"Manifest saved to: {preview_csv}")


if __name__ == "__main__":
    main()