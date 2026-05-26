import pandas as pd
from src.config import FEATURES_DIR
from src.density_estimator import estimate_density_placeholder


def main():
    input_csv = FEATURES_DIR / "image_features.csv"
    output_csv = FEATURES_DIR / "image_features_with_density_placeholder.csv"

    if not input_csv.exists():
        print(f"Input CSV not found: {input_csv}")
        print("Run this first:")
        print("python run_folder.py")
        return

    df = pd.read_csv(input_csv)

    print("Columns available in CSV:")
    print(list(df.columns))

    df["density_placeholder"] = df.apply(estimate_density_placeholder, axis=1)

    df.to_csv(output_csv, index=False)

    print("\nDone.")
    print(f"Saved density placeholder CSV to: {output_csv}")


if __name__ == "__main__":
    main()