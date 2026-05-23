from pathlib import Path
import argparse

import cv2
import pandas as pd


CROPPED_MANIFEST = Path("outputs/features/cropped_frame_manifest.csv")
OUTPUT_DIR = Path("outputs/rotation_previews")


def rotate_image(image, rotation_name):
    if rotation_name == "rot0":
        return image

    if rotation_name == "rot90_clockwise":
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

    if rotation_name == "rot180":
        return cv2.rotate(image, cv2.ROTATE_180)

    if rotation_name == "rot90_counterclockwise":
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    raise ValueError(f"Unknown rotation: {rotation_name}")


ROTATIONS = [
    "rot0",
    "rot90_clockwise",
    "rot180",
    "rot90_counterclockwise",
]


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
        default=9,
        help="Number of cropped frames to preview."
    )

    args = parser.parse_args()

    if not CROPPED_MANIFEST.exists():
        print(f"Cropped manifest not found: {CROPPED_MANIFEST}")
        print("Run apply_lens_crops.py first.")
        return

    df = pd.read_csv(CROPPED_MANIFEST)

    if "crop_status" in df.columns:
        df = df[df["crop_status"] == "success"].copy()

    df = df[df["lens_id"].isin(args.lenses)].copy()
    df = df.head(args.limit)

    if df.empty:
        print("No cropped frames found for selected lenses.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    preview_rows = []

    for _, row in df.iterrows():
        cropped_frame_key = str(row["cropped_frame_key"])
        lens_id = int(row["lens_id"])
        cropped_frame_path = Path(str(row["cropped_frame_path"]))

        if not cropped_frame_path.exists():
            print(f"Missing cropped frame: {cropped_frame_path}")
            continue

        image = cv2.imread(str(cropped_frame_path))

        if image is None:
            print(f"Could not read cropped frame: {cropped_frame_path}")
            continue

        print(f"\nCreating rotation previews for: {cropped_frame_key}")

        for rotation_name in ROTATIONS:
            rotated = rotate_image(image, rotation_name)

            output_path = (
                OUTPUT_DIR
                / f"lens{lens_id}"
                / f"{cropped_frame_key}_{rotation_name}.jpg"
            )

            output_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(output_path), rotated)

            preview_rows.append({
                "cropped_frame_key": cropped_frame_key,
                "lens_id": lens_id,
                "rotation_name": rotation_name,
                "source_cropped_frame_path": str(cropped_frame_path),
                "rotation_preview_path": str(output_path),
                "height": rotated.shape[0],
                "width": rotated.shape[1],
            })

            print(f"  Saved: {output_path}")

    preview_df = pd.DataFrame(preview_rows)
    preview_csv = OUTPUT_DIR / "rotation_preview_manifest.csv"
    preview_df.to_csv(preview_csv, index=False)

    print("\nDone.")
    print(f"Rotation previews saved in: {OUTPUT_DIR}")
    print(f"Manifest saved to: {preview_csv}")


if __name__ == "__main__":
    main()