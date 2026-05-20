import pandas as pd
from pathlib import Path

from src.pipeline import ImageFeaturePipeline
from src.config import IMAGE_DIR, FEATURES_DIR


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def main():
    # Get all image files from data/images/
    image_paths = [
        path for path in IMAGE_DIR.iterdir()
        if path.suffix.lower() in VALID_EXTENSIONS
    ]

    image_paths = sorted(image_paths)

    if not image_paths:
        print(f"No images found in: {IMAGE_DIR}")
        print("Please add images inside data/images/")
        return

    print(f"Found {len(image_paths)} images.")
    print("Starting feature extraction...\n")

    pipeline = ImageFeaturePipeline()

    all_features = []

    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Processing: {image_path.name}")

        try:
            features = pipeline.process_image(image_path)
            all_features.append(features)

            print(
                f"  Vehicles: {features['total_vehicles']} | "
                f"Cars: {features['car_count']} | "
                f"Bikes: {features['motorcycle_count']} | "
                f"Buses: {features['bus_count']} | "
                f"Trucks: {features['truck_count']} | "
                f"Road: {features['road_condition']} | "
                f"Dust score: {features['road_dust_score']}"
            )

        except Exception as e:
            print(f"  Error processing {image_path.name}: {e}")

    if not all_features:
        print("\nNo images were processed successfully.")
        return

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = FEATURES_DIR / "image_features.csv"

    df = pd.DataFrame(all_features)
    df.to_csv(output_csv, index=False)

    print("\nDone.")
    print(f"Saved feature CSV to: {output_csv}")
    print(f"Processed {len(all_features)} images successfully.")


if __name__ == "__main__":
    main()