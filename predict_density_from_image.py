import argparse
import pandas as pd
import joblib

from src.pipeline import ImageFeaturePipeline
from src.config import OUTPUT_DIR


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Path to input image")
    args = parser.parse_args()

    model_path = OUTPUT_DIR / "models" / "density_model_placeholder.joblib"

    if not model_path.exists():
        print(f"Model not found: {model_path}")
        print("Run this first:")
        print("python train_density_model.py")
        return

    model = joblib.load(model_path)

    pipeline = ImageFeaturePipeline()
    features = pipeline.process_image(args.image)

    df = pd.DataFrame([features])

    predicted_density = model.predict(df)[0]

    print("\nExtracted Features:")
    for key, value in features.items():
        print(f"{key}: {value}")

    print("\nPredicted Density:")
    print(f"density_placeholder_prediction: {predicted_density:.4f}")


if __name__ == "__main__":
    main()