import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.config import FEATURES_DIR, OUTPUT_DIR


def evaluate_model(name, model, X_train, X_test, y_train, y_test, preprocessor):
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, y_pred)

    return {
        "model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2": r2,
        "pipeline": pipeline
    }


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
        "traffic_level"
    ]

    available_feature_columns = [
        col for col in feature_columns if col in df.columns
    ]

    X = df[available_feature_columns]
    y = df[target_column]

    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object"]).columns.tolist()

    linear_preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    tree_preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    models = [
        ("Linear Regression", LinearRegression(), linear_preprocessor),
        ("Ridge Regression", Ridge(alpha=1.0), linear_preprocessor),
        ("Lasso Regression", Lasso(alpha=0.001, max_iter=10000), linear_preprocessor),
        (
            "Random Forest",
            RandomForestRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=None
            ),
            tree_preprocessor
        ),
        (
            "Extra Trees",
            ExtraTreesRegressor(
                n_estimators=300,
                random_state=42,
                max_depth=None
            ),
            tree_preprocessor
        )
    ]

    results = []
    best_model = None
    best_r2 = -999

    for name, model, preprocessor in models:
        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test,
            preprocessor
        )

        results.append({
            "model": result["model"],
            "MAE": result["MAE"],
            "MSE": result["MSE"],
            "RMSE": result["RMSE"],
            "R2": result["R2"]
        })

        if result["R2"] > best_r2:
            best_r2 = result["R2"]
            best_model = result["pipeline"]

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values(by="R2", ascending=False)

    print("\nModel Comparison:")
    print(results_df)

    output_csv = FEATURES_DIR / "model_comparison_results.csv"
    results_df.to_csv(output_csv, index=False)

    model_dir = OUTPUT_DIR / "models"
    model_dir.mkdir(parents=True, exist_ok=True)

    best_model_path = model_dir / "best_density_model_placeholder.joblib"
    joblib.dump(best_model, best_model_path)

    print("\nDone.")
    print(f"Saved model comparison to: {output_csv}")
    print(f"Saved best model to: {best_model_path}")


if __name__ == "__main__":
    main()