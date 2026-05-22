import time
import argparse
from pathlib import Path

import pandas as pd

from src.osm_features import extract_osm_features


INPUT_PATH = Path("outputs/features/clean_sensor_data.csv")
OUTPUT_DIR = Path("outputs/features")

CHECKPOINT_PATH = OUTPUT_DIR / "osm_features_checkpoint.csv"
FINAL_OUTPUT_PATH = OUTPUT_DIR / "clean_sensor_data_with_osm.csv"


def create_location_key(latitude, longitude, decimals=4):
    """
    Rounds GPS coordinates so nearby points reuse the same OSM features.

    decimals=4 gives roughly 10-11 meter precision.
    """
    lat_round = round(float(latitude), decimals)
    lon_round = round(float(longitude), decimals)
    return f"{lat_round}_{lon_round}", lat_round, lon_round


def load_existing_checkpoint():
    """
    Loads already extracted OSM features if checkpoint exists.
    """
    if CHECKPOINT_PATH.exists():
        checkpoint_df = pd.read_csv(CHECKPOINT_PATH)
        print(f"Loaded checkpoint: {CHECKPOINT_PATH}")
        print(f"Already completed locations: {len(checkpoint_df)}")
        return checkpoint_df

    return pd.DataFrame()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing only. Example: --limit 5"
    )
    parser.add_argument(
        "--round-decimals",
        type=int,
        default=4,
        help="Coordinate rounding precision. Default: 4"
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=1.0,
        help="Sleep time between OSM queries in seconds. Default: 1.0"
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=3,
        help="Save checkpoint after every N successful locations. Default: 3"
    )

    args = parser.parse_args()

    if not INPUT_PATH.exists():
        print(f"Input file not found: {INPUT_PATH}")
        print("Run first:")
        print("python clean_sensor_data.py")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(INPUT_PATH)

    required_cols = {"latitude", "longitude"}

    if not required_cols.issubset(df.columns):
        print("CSV must contain latitude and longitude columns.")
        print("Available columns:")
        print(list(df.columns))
        return

    print("Input dataset shape:", df.shape)

    # Create rounded location keys
    location_keys = []
    rounded_lats = []
    rounded_lons = []

    for _, row in df.iterrows():
        key, lat_round, lon_round = create_location_key(
            row["latitude"],
            row["longitude"],
            decimals=args.round_decimals
        )

        location_keys.append(key)
        rounded_lats.append(lat_round)
        rounded_lons.append(lon_round)

    df["osm_location_key"] = location_keys
    df["osm_latitude_rounded"] = rounded_lats
    df["osm_longitude_rounded"] = rounded_lons

    unique_locations = (
        df[["osm_location_key", "osm_latitude_rounded", "osm_longitude_rounded"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    if args.limit is not None:
        unique_locations = unique_locations.head(args.limit)

    print(f"Unique rounded locations to process: {len(unique_locations)}")

    # Load checkpoint if available
    checkpoint_df = load_existing_checkpoint()

    completed_keys = set()
    if not checkpoint_df.empty and "osm_location_key" in checkpoint_df.columns:
        completed_keys = set(checkpoint_df["osm_location_key"].astype(str))

    all_osm_rows = []

    if not checkpoint_df.empty:
        all_osm_rows = checkpoint_df.to_dict(orient="records")

    processed_since_last_save = 0

    for idx, row in unique_locations.iterrows():
        location_key = str(row["osm_location_key"])
        lat = float(row["osm_latitude_rounded"])
        lon = float(row["osm_longitude_rounded"])

        if location_key in completed_keys:
            print(f"[{idx + 1}/{len(unique_locations)}] Skipping completed: {location_key}")
            continue

        print(f"\n[{idx + 1}/{len(unique_locations)}] Extracting OSM for {location_key}")
        print(f"Latitude: {lat}, Longitude: {lon}")

        try:
            osm_features = extract_osm_features(
                latitude=lat,
                longitude=lon,
                radii=(250,)
            )

            osm_features["osm_location_key"] = location_key
            osm_features["osm_latitude_rounded"] = lat
            osm_features["osm_longitude_rounded"] = lon
            osm_features["osm_status"] = "success"
            osm_features["osm_error"] = ""

            all_osm_rows.append(osm_features)
            completed_keys.add(location_key)
            processed_since_last_save += 1

            print("  Success.")

        except Exception as e:
            print(f"  Error: {e}")

            # Save failed row too, so we know it was attempted.
            osm_features = {
                "osm_location_key": location_key,
                "osm_latitude_rounded": lat,
                "osm_longitude_rounded": lon,
                "osm_status": "failed",
                "osm_error": str(e),
            }

            all_osm_rows.append(osm_features)
            completed_keys.add(location_key)
            processed_since_last_save += 1

        # Save checkpoint periodically
        if processed_since_last_save >= args.checkpoint_every:
            checkpoint_out = pd.DataFrame(all_osm_rows)
            checkpoint_out.to_csv(CHECKPOINT_PATH, index=False)
            print(f"  Checkpoint saved: {CHECKPOINT_PATH}")
            processed_since_last_save = 0

        # Avoid hammering Overpass API
        time.sleep(args.sleep)

    # Final checkpoint save
    osm_unique_df = pd.DataFrame(all_osm_rows)
    osm_unique_df.to_csv(CHECKPOINT_PATH, index=False)
    print(f"\nFinal checkpoint saved: {CHECKPOINT_PATH}")

    # Merge OSM features back to all original 281 rows
    final_df = df.merge(
        osm_unique_df,
        on=["osm_location_key", "osm_latitude_rounded", "osm_longitude_rounded"],
        how="left"
    )

    final_df.to_csv(FINAL_OUTPUT_PATH, index=False)

    print("\nDone.")
    print("Original sensor rows:", len(df))
    print("Unique OSM rows:", len(osm_unique_df))
    print("Final dataset shape:", final_df.shape)
    print(f"Saved final OSM-enriched dataset to: {FINAL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()