import argparse
import pandas as pd
from pathlib import Path

from src.pipeline import ImageFeaturePipeline
from src.config import FEATURES_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    pipeline = ImageFeaturePipeline()
    features = pipeline.process_image(args.image)

    print("\nExtracted Image Features:")
    for key, value in features.items():
        print(f"{key}: {value}")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)

    output_csv = FEATURES_DIR / "single_image_features.csv"
    pd.DataFrame([features]).to_csv(output_csv, index=False)

    print(f"\nSaved features to: {output_csv}")


if __name__ == "__main__":
    main()