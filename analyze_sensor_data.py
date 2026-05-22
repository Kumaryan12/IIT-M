import pandas as pd
from pathlib import Path


INPUT_PATH = Path("outputs/features/clean_sensor_data.csv")


def main():
    if not INPUT_PATH.exists():
        print(f"File not found: {INPUT_PATH}")
        print("Run this first:")
        print("python clean_sensor_data.py")
        return

    df = pd.read_csv(INPUT_PATH)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nMissing values:")
    print(df.isna().sum())

    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns

    print("\nNumeric summary:")
    print(df[numeric_cols].describe())

    print("\nCoordinate range:")
    print("Latitude:", df["latitude"].min(), "to", df["latitude"].max())
    print("Longitude:", df["longitude"].min(), "to", df["longitude"].max())

    if "timestamp" in df.columns:
        print("\nTime range:")
        print(df["timestamp"].min(), "to", df["timestamp"].max())

    pm_cols = [
        "pm1_mass",
        "pm2_5_mass",
        "pm4_mass",
        "pm10_mass",
        "npm1_count",
        "npm2_5_count",
        "npm4_count",
        "npm10_count",
    ]

    available_pm_cols = [col for col in pm_cols if col in df.columns]

    print("\nPM / particle columns summary:")
    print(df[available_pm_cols].describe())


if __name__ == "__main__":
    main()