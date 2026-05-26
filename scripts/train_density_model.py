import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import FEATURES_DIR, OUTPUT_DIR


def main():
    input_csv = FEATURES_DIR / "image_features_with_density_placeholder.csv"

    if not input_csv.exists():
        print(f"Input file not found: {input_csv}")
        print("Run these first:")
        print("python run_folder.py")
        print("python add_density_column.py")
        return

    df = pd.read_csv(input_csv)

    target_column = "density_placeholder"

    if target_column not in df.columns:
        print(f"Target column '{target_column}' not found.")
        print("Available columns:")
        print(list(df.columns))
        return

    feature_columns = [
    "total_vehicles",
    "car_count",
    "motorcycle_count",
    "bus_count",
    "truck_count",
    "bicycle_count",
    "heavy_vehicle_count",
    "two_wheeler_count",
    "motor_vehicle_count",
    "non_motor_vehicle_count",
    "heavy_vehicle_ratio",
    "two_wheeler_ratio",
    "motor_vehicle_ratio",
    "car_ratio",
    "truck_ratio",
    "bus_ratio",
    "traffic_load_score",
    "road_dust_score",
    "dust_traffic_interaction_score",
    "heavy_vehicle_dust_score",

    "clip_road_dust_score",
    "brightness_mean",
    "contrast_std",
    "brown_pixel_ratio",
    "edge_density",
    "haze_score",
    "visual_dust_score",

    "road_condition",
    "road_condition_full_label",
    "road_dust_level",
    "traffic_level",
    "vehicle_box_area_ratio",
    "average_vehicle_confidence",
    "small_vehicle_count",
    "medium_vehicle_count",
    "large_vehicle_count"
]

    available_feature_columns = [
        col for col in feature_columns if col in df.columns
    ]

    X = df[available_feature_columns]
    y = df[target_column]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        max_depth=None
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    if len(df) < 5:
        print("Warning: Very few samples. Training will run, but metrics are not meaningful.")

    if len(df) >= 5:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        print("\nModel Evaluation:")
        print(f"MAE: {mae:.4f}")
        print(f"MSE: {mse:.4f}")
        print(f"R2 Score: {r2:.4f}")

    else:
        pipeline.fit(X, y)
        print("Model trained on all available data.")

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "density_model_placeholder.joblib"
    joblib.dump(pipeline, model_path)

    print("\nDone.")
    print(f"Saved trained model to: {model_path}")


if __name__ == "__main__":
    main()