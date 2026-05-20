import pandas as pd
from src.config import FEATURES_DIR


def main():
    input_csv = FEATURES_DIR / "image_features_with_density_placeholder.csv"

    if not input_csv.exists():
        print(f"File not found: {input_csv}")
        print("Run:")
        print("python run_folder.py")
        print("python add_density_column.py")
        return

    df = pd.read_csv(input_csv)

    print("\nDataset Shape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

    print("\nVehicle Summary:")
    vehicle_cols = [
        "total_vehicles",
        "car_count",
        "motorcycle_count",
        "bus_count",
        "truck_count",
        "bicycle_count",
        "heavy_vehicle_count",
        "traffic_load_score"
    ]

    available_vehicle_cols = [col for col in vehicle_cols if col in df.columns]
    print(df[available_vehicle_cols].describe())

    print("\nRoad Condition Counts:")
    if "road_condition" in df.columns:
        print(df["road_condition"].value_counts())

    print("\nRoad Dust Level Counts:")
    if "road_dust_level" in df.columns:
        print(df["road_dust_level"].value_counts())

    print("\nDensity Placeholder Summary:")
    if "density_placeholder" in df.columns:
        print(df["density_placeholder"].describe())

    print("\nTop rows:")
    display_cols = [
        "image_name",
        "total_vehicles",
        "car_count",
        "bus_count",
        "truck_count",
        "road_condition",
        "road_dust_score",
        "traffic_load_score",
        "dust_traffic_interaction_score",
        "density_placeholder"
    ]

    available_display_cols = [col for col in display_cols if col in df.columns]
    print(df[available_display_cols].head(10))


if __name__ == "__main__":
    main()