import pandas as pd
from pathlib import Path


RAW_DATA_PATH = Path("data/sensor/MC1S_window_115430_124150.csv")
OUTPUT_DIR = Path("outputs/features")
OUTPUT_PATH = OUTPUT_DIR / "clean_sensor_data.csv"


def main():
    if not RAW_DATA_PATH.exists():
        print(f"Input file not found: {RAW_DATA_PATH}")
        print("Create this folder and place the CSV there:")
        print("data/sensor/MC1S_window_115430_124150.csv")
        return

    df = pd.read_csv(RAW_DATA_PATH)

    print("Original shape:", df.shape)
    print("Original columns:")
    print(list(df.columns))

    # Rename columns into clean project-friendly names
    rename_map = {
        "timestamp": "timestamp",
        "value.lat": "latitude",
        "value.long": "longitude",

        "temp": "temperature_c",
        "rh": "relative_humidity",

        "value.sPM1": "pm1_mass",
        "value.sPM2": "pm2_5_mass",
        "value.sPM4": "pm4_mass",
        "value.sPM10": "pm10_mass",

        "value.sNPM1": "npm1_count",
        "nPM2": "npm2_5_count",
        "value.sNPM4": "npm4_count",
        "value.sNPM10": "npm10_count",

        "value.co_ppb": "co_ppb",
        "value.no2_ppb": "no2_ppb",
        "value.so2_ppb": "so2_ppb",
        "value.o3_ppb_compensated": "o3_ppb_compensated",
    }

    df = df.rename(columns=rename_map)

    # Parse timestamp
    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        format="%d-%m-%Y %H:%M",
        errors="coerce"
    )

    # Create sample_id
    df.insert(0, "sample_id", range(1, len(df) + 1))

    # Reorder columns
    ordered_columns = [
        "sample_id",
        "timestamp",
        "latitude",
        "longitude",

        "temperature_c",
        "relative_humidity",

        "pm1_mass",
        "pm2_5_mass",
        "pm4_mass",
        "pm10_mass",

        "npm1_count",
        "npm2_5_count",
        "npm4_count",
        "npm10_count",

        "co_ppb",
        "no2_ppb",
        "so2_ppb",
        "o3_ppb_compensated",
    ]

    existing_ordered_columns = [
        col for col in ordered_columns if col in df.columns
    ]

    df = df[existing_ordered_columns]

    # Remove rows without coordinates
    before_rows = len(df)
    df = df.dropna(subset=["latitude", "longitude"])
    after_rows = len(df)

    print(f"Removed rows without lat-long: {before_rows - after_rows}")

    # Basic sanity filtering for coordinates
    df = df[
        (df["latitude"].between(-90, 90)) &
        (df["longitude"].between(-180, 180))
    ]

    # Sort by timestamp if available
    df = df.sort_values(by=["timestamp", "sample_id"]).reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nCleaned shape:", df.shape)
    print("Cleaned columns:")
    print(list(df.columns))

    print(f"\nSaved cleaned sensor data to: {OUTPUT_PATH}")

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()