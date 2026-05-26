from pathlib import Path
import argparse

import cv2
import pandas as pd


PROCESSED_MANIFEST = Path("outputs/features/processed_frame_manifest.csv")
OUTPUT_ROOT = Path("outputs/road_roi_previews")


# ROI format:
# roi_name: (x1_ratio, y1_ratio, x2_ratio, y2_ratio)
#
# These are candidate road/ground regions on already processed frames.
# We are trying to avoid sky, trees, buildings, and masked camera rig area.
ROAD_ROI_CANDIDATES = {
    "lower_middle": (0.10, 0.45, 0.90, 0.82),
    "lower_left": (0.00, 0.45, 0.65, 0.82),
    "lower_right": (0.35, 0.45, 1.00, 0.82),

    "road_band_wide": (0.00, 0.35, 1.00, 0.78),
    "road_band_middle": (0.10, 0.35, 0.90, 0.78),

    "bottom_above_mask": (0.05, 0.58, 0.95, 0.82),
    "center_road_strip": (0.20, 0.45, 0.80, 0.78),

    # More aggressive options if sky/building/trees dominate
    "low_narrow_middle": (0.20, 0.55, 0.80, 0.80),
    "low_narrow_left": (0.00, 0.55, 0.60, 0.80),
    "low_narrow_right": (0.40, 0.55, 1.00, 0.80),
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

    return image[y1:y2, x1:x2], x1, y1, x2, y2


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
        default=12,
        help="Number of processed frames to preview."
    )

    args = parser.parse_args()

    if not PROCESSED_MANIFEST.exists():
        print(f"Processed frame manifest not found: {PROCESSED_MANIFEST}")
        print("Run apply_lens_preprocessing.py first.")
        return

    df = pd.read_csv(PROCESSED_MANIFEST)

    if "preprocess_status" in df.columns:
        df = df[df["preprocess_status"] == "success"].copy()

    df = df[df["lens_id"].isin(args.lenses)].copy()
    df = df.head(args.limit)

    if df.empty:
        print("No processed frames found for selected lenses.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    preview_rows = []

    print("\nRoad ROI preview setup:")
    print(f"Processed frames selected: {len(df)}")
    print(f"Lenses selected: {args.lenses}")
    print(f"Output folder: {OUTPUT_ROOT}")

    for _, row in df.iterrows():
        processed_frame_key = str(row["processed_frame_key"])
        lens_id = int(row["lens_id"])
        processed_frame_path = Path(str(row["processed_frame_path"]))

        if not processed_frame_path.exists():
            print(f"Missing processed frame: {processed_frame_path}")
            continue

        image = cv2.imread(str(processed_frame_path))

        if image is None:
            print(f"Could not read image: {processed_frame_path}")
            continue

        print(f"\nGenerating road ROI previews for: {processed_frame_key}")

        for roi_name, roi_ratios in ROAD_ROI_CANDIDATES.items():
            roi_image, x1, y1, x2, y2 = crop_image(image, roi_ratios)

            output_path = (
                OUTPUT_ROOT
                / f"lens{lens_id}"
                / f"{processed_frame_key}_{roi_name}.jpg"
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), roi_image)

            preview_rows.append({
                "processed_frame_key": processed_frame_key,
                "lens_id": lens_id,
                "roi_name": roi_name,
                "source_processed_frame_path": str(processed_frame_path),
                "road_roi_preview_path": str(output_path),
                "roi_x1": x1,
                "roi_y1": y1,
                "roi_x2": x2,
                "roi_y2": y2,
                "roi_x1_ratio": roi_ratios[0],
                "roi_y1_ratio": roi_ratios[1],
                "roi_x2_ratio": roi_ratios[2],
                "roi_y2_ratio": roi_ratios[3],
                "roi_width": roi_image.shape[1],
                "roi_height": roi_image.shape[0],
            })

            print(f"  Saved: {output_path}")

    preview_df = pd.DataFrame(preview_rows)
    preview_manifest_path = OUTPUT_ROOT / "road_roi_preview_manifest.csv"
    preview_df.to_csv(preview_manifest_path, index=False)

    print("\nDone.")
    print(f"Road ROI previews saved to: {OUTPUT_ROOT}")
    print(f"Preview manifest saved to: {preview_manifest_path}")
    print("\nOpen previews with:")
    print(f"open {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()