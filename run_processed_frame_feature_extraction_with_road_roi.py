import argparse
from pathlib import Path

import cv2
import pandas as pd

from src.vehicle_detector import VehicleDetector
from src.road_classifier import RoadClassifier
from src.image_features import extract_basic_image_features
from src.config import ANNOTATED_DIR


PROCESSED_FRAME_MANIFEST = Path("outputs/features/processed_frame_manifest.csv")

OUTPUT_DIR = Path("outputs/features")
OUTPUT_PATH = OUTPUT_DIR / "processed_frame_visual_features_roi.csv"

ROAD_ROI_ROOT = Path("outputs/road_roi_frames")


# Selected from manual road ROI preview inspection:
# Lens 1 -> low_narrow_right
# Lens 6 -> lower_left
# Lens 4 -> no reliable road ROI, vehicle-only
ROAD_ROI_CONFIG = {
    1: (0.40, 0.55, 1.00, 0.80),  # low_narrow_right
    6: (0.00, 0.45, 0.65, 0.82),  # lower_left
}


def crop_image(image, crop_ratios):
    h, w = image.shape[:2]

    x1r, y1r, x2r, y2r = crop_ratios

    x1 = int(x1r * w)
    y1 = int(y1r * h)
    x2 = int(x2r * w)
    y2 = int(y2r * h)

    x1 = max(0, min(x1, w - 1))
    y1 = max(0, min(y1, h - 1))
    x2 = max(x1 + 1, min(x2, w))
    y2 = max(y1 + 1, min(y2, h))

    cropped = image[y1:y2, x1:x2]
    return cropped, x1, y1, x2, y2


def road_dust_level_from_score(score: float) -> str:
    if score <= 0.30:
        return "low"
    if score <= 0.65:
        return "moderate"
    return "high"


def process_one_frame(
    row,
    vehicle_detector: VehicleDetector,
    road_classifier: RoadClassifier,
    save_annotated: bool,
) -> dict:
    processed_frame_key = str(row["processed_frame_key"])
    lens_id = int(row["lens_id"])
    frame_path = Path(str(row["processed_frame_path"]))

    if not frame_path.exists():
        return {
            "processed_frame_key": processed_frame_key,
            "source_frame_key": row.get("source_frame_key", ""),
            "matched_run_id": row.get("matched_run_id", ""),
            "video_offset_sec": row.get("video_offset_sec", None),
            "lens_id": lens_id,
            "processed_frame_path": str(frame_path),
            "feature_status": "failed",
            "feature_error": "processed_frame_not_found",
        }

    image = cv2.imread(str(frame_path))

    if image is None:
        return {
            "processed_frame_key": processed_frame_key,
            "source_frame_key": row.get("source_frame_key", ""),
            "matched_run_id": row.get("matched_run_id", ""),
            "video_offset_sec": row.get("video_offset_sec", None),
            "lens_id": lens_id,
            "processed_frame_path": str(frame_path),
            "feature_status": "failed",
            "feature_error": "could_not_read_processed_frame",
        }

    # 1. Vehicle detection on full processed frame
    annotated_path = None
    if save_annotated:
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        annotated_path = ANNOTATED_DIR / f"annotated_{processed_frame_key}.jpg"

    vehicle_features = vehicle_detector.detect(
        image_path=str(frame_path),
        save_annotated_path=str(annotated_path) if annotated_path else None,
    )

    # Basic vehicle feature engineering
    total_vehicles = vehicle_features["total_vehicles"]
    car_count = vehicle_features["car_count"]
    motorcycle_count = vehicle_features["motorcycle_count"]
    bus_count = vehicle_features["bus_count"]
    truck_count = vehicle_features["truck_count"]
    bicycle_count = vehicle_features["bicycle_count"]

    heavy_vehicle_count = bus_count + truck_count
    two_wheeler_count = motorcycle_count + bicycle_count
    motor_vehicle_count = car_count + motorcycle_count + bus_count + truck_count
    non_motor_vehicle_count = bicycle_count

    if total_vehicles > 0:
        heavy_vehicle_ratio = heavy_vehicle_count / total_vehicles
        two_wheeler_ratio = two_wheeler_count / total_vehicles
        motor_vehicle_ratio = motor_vehicle_count / total_vehicles
        car_ratio = car_count / total_vehicles
        truck_ratio = truck_count / total_vehicles
        bus_ratio = bus_count / total_vehicles
    else:
        heavy_vehicle_ratio = 0.0
        two_wheeler_ratio = 0.0
        motor_vehicle_ratio = 0.0
        car_ratio = 0.0
        truck_ratio = 0.0
        bus_ratio = 0.0

    traffic_load_score = (
        1.0 * car_count
        + 0.6 * motorcycle_count
        + 3.0 * bus_count
        + 3.5 * truck_count
        + 0.1 * bicycle_count
    )

    if total_vehicles == 0:
        traffic_level = "no_traffic"
    elif total_vehicles <= 5:
        traffic_level = "low"
    elif total_vehicles <= 15:
        traffic_level = "moderate"
    else:
        traffic_level = "high"

    # 2. Road ROI features only for lenses with valid ROI
    road_roi_available = lens_id in ROAD_ROI_CONFIG

    road_roi_path = ""
    road_condition = ""
    road_condition_full_label = ""
    road_condition_confidence = 0.0
    clip_road_dust_score = 0.0

    brightness_mean = 0.0
    contrast_std = 0.0
    brown_pixel_ratio = 0.0
    edge_density = 0.0
    haze_score = 0.0
    visual_dust_score = 0.0

    road_dust_score = 0.0
    road_dust_level = "not_available"

    roi_x1 = roi_y1 = roi_x2 = roi_y2 = None
    roi_ratios = (None, None, None, None)

    if road_roi_available:
        roi_ratios = ROAD_ROI_CONFIG[lens_id]
        roi_image, roi_x1, roi_y1, roi_x2, roi_y2 = crop_image(image, roi_ratios)

        road_roi_path = (
            ROAD_ROI_ROOT
            / str(row["matched_run_id"])
            / f"lens{lens_id}"
            / f"{processed_frame_key}_road_roi.jpg"
        )

        road_roi_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(road_roi_path), roi_image)

        # CLIP road classifier on ROI, not whole frame
        road_features = road_classifier.classify(str(road_roi_path))

        road_condition = road_features["road_condition"]
        road_condition_full_label = road_features["road_condition_full_label"]
        road_condition_confidence = road_features["road_condition_confidence"]
        clip_road_dust_score = road_features["clip_road_dust_score"]

        # OpenCV features on ROI, not whole frame
        basic_image_features = extract_basic_image_features(road_roi_path)

        brightness_mean = basic_image_features["brightness_mean"]
        contrast_std = basic_image_features["contrast_std"]
        brown_pixel_ratio = basic_image_features["brown_pixel_ratio"]
        edge_density = basic_image_features["edge_density"]
        haze_score = basic_image_features["haze_score"]
        visual_dust_score = basic_image_features["visual_dust_score"]

        # Combined road dust score from ROI
        road_dust_score = (
            0.70 * clip_road_dust_score
            + 0.30 * visual_dust_score
        )

        road_dust_level = road_dust_level_from_score(road_dust_score)

    dust_traffic_interaction_score = traffic_load_score * road_dust_score
    heavy_vehicle_dust_score = heavy_vehicle_count * road_dust_score

    features = {
        # Metadata
        "processed_frame_key": processed_frame_key,
        "source_frame_key": row.get("source_frame_key", ""),
        "matched_run_id": row.get("matched_run_id", ""),
        "video_offset_sec": row.get("video_offset_sec", None),
        "lens_id": lens_id,
        "processed_frame_path": str(frame_path),

        # Vehicle features from full processed frame
        "total_vehicles": total_vehicles,
        "car_count": car_count,
        "motorcycle_count": motorcycle_count,
        "bus_count": bus_count,
        "truck_count": truck_count,
        "bicycle_count": bicycle_count,

        "vehicle_box_area_ratio": vehicle_features["vehicle_box_area_ratio"],
        "average_vehicle_confidence": vehicle_features["average_vehicle_confidence"],
        "small_vehicle_count": vehicle_features["small_vehicle_count"],
        "medium_vehicle_count": vehicle_features["medium_vehicle_count"],
        "large_vehicle_count": vehicle_features["large_vehicle_count"],

        "raw_vehicle_detection_count": vehicle_features.get("raw_vehicle_detection_count", total_vehicles),
        "platform_filtered_count": vehicle_features.get("platform_filtered_count", 0),

        "full_source_detection_count": vehicle_features.get("full_source_detection_count", total_vehicles),
        "tile_source_detection_count": vehicle_features.get("tile_source_detection_count", 0),
        "raw_detection_count_before_merge": vehicle_features.get("raw_detection_count_before_merge", total_vehicles),
        "final_detection_count_after_merge": vehicle_features.get("final_detection_count_after_merge", total_vehicles),

        # Engineered traffic features
        "heavy_vehicle_count": heavy_vehicle_count,
        "two_wheeler_count": two_wheeler_count,
        "motor_vehicle_count": motor_vehicle_count,
        "non_motor_vehicle_count": non_motor_vehicle_count,

        "heavy_vehicle_ratio": heavy_vehicle_ratio,
        "two_wheeler_ratio": two_wheeler_ratio,
        "motor_vehicle_ratio": motor_vehicle_ratio,
        "car_ratio": car_ratio,
        "truck_ratio": truck_ratio,
        "bus_ratio": bus_ratio,

        "traffic_load_score": traffic_load_score,
        "traffic_level": traffic_level,

        # Road ROI information
        "road_roi_available": road_roi_available,
        "road_roi_path": str(road_roi_path) if road_roi_available else "",
        "road_roi_x1": roi_x1,
        "road_roi_y1": roi_y1,
        "road_roi_x2": roi_x2,
        "road_roi_y2": roi_y2,
        "road_roi_x1_ratio": roi_ratios[0],
        "road_roi_y1_ratio": roi_ratios[1],
        "road_roi_x2_ratio": roi_ratios[2],
        "road_roi_y2_ratio": roi_ratios[3],

        # Road/visual features from ROI only
        "road_condition": road_condition,
        "road_condition_full_label": road_condition_full_label,
        "road_condition_confidence": road_condition_confidence,
        "clip_road_dust_score": clip_road_dust_score,

        "brightness_mean": brightness_mean,
        "contrast_std": contrast_std,
        "brown_pixel_ratio": brown_pixel_ratio,
        "edge_density": edge_density,
        "haze_score": haze_score,
        "visual_dust_score": visual_dust_score,

        "road_dust_score": road_dust_score,
        "road_dust_level": road_dust_level,

        "dust_traffic_interaction_score": dust_traffic_interaction_score,
        "heavy_vehicle_dust_score": heavy_vehicle_dust_score,

        "annotated_image_path": str(annotated_path) if annotated_path else "",

        "feature_status": "success",
        "feature_error": "",
    }

    return features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lenses",
        nargs="+",
        type=int,
        default=[1, 4, 6],
        help="Lens IDs to process. Example: --lenses 1 4 6"
    )

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional limit for testing. Example: --limit 12"
    )

    parser.add_argument(
        "--no-annotated",
        action="store_true",
        help="Disable saving annotated YOLO images."
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save output after every N processed frames."
    )

    args = parser.parse_args()

    if not PROCESSED_FRAME_MANIFEST.exists():
        print(f"Processed frame manifest not found: {PROCESSED_FRAME_MANIFEST}")
        print("Run apply_lens_preprocessing.py first.")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ROAD_ROI_ROOT.mkdir(parents=True, exist_ok=True)

    manifest_df = pd.read_csv(PROCESSED_FRAME_MANIFEST)

    required_cols = {
        "processed_frame_key",
        "source_frame_key",
        "matched_run_id",
        "video_offset_sec",
        "lens_id",
        "processed_frame_path",
        "preprocess_status",
    }

    missing_cols = required_cols - set(manifest_df.columns)

    if missing_cols:
        print("Processed frame manifest is missing columns:")
        print(missing_cols)
        return

    frame_df = manifest_df[manifest_df["preprocess_status"] == "success"].copy()
    frame_df = frame_df[frame_df["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        frame_df = frame_df.head(args.limit).copy()

    print("\nROI-based processed frame feature extraction setup:")
    print(f"Total processed frames available: {len(manifest_df[manifest_df['preprocess_status'] == 'success'])}")
    print(f"Frames selected now: {len(frame_df)}")
    print(f"Lenses selected: {args.lenses}")
    print(f"Road ROI lenses: {sorted(ROAD_ROI_CONFIG.keys())}")
    print(f"Save annotated images: {not args.no_annotated}")

    output_rows = []
    completed_keys = set()

    if OUTPUT_PATH.exists():
        old_df = pd.read_csv(OUTPUT_PATH)

        if "processed_frame_key" in old_df.columns:
            completed_keys = set(old_df["processed_frame_key"].astype(str))
            output_rows = old_df.to_dict(orient="records")

        print(f"\nLoaded existing ROI output file: {OUTPUT_PATH}")
        print(f"Already processed frames: {len(completed_keys)}")

    vehicle_detector = VehicleDetector()
    road_classifier = RoadClassifier()

    processed_since_save = 0

    for _, row in frame_df.iterrows():
        processed_frame_key = str(row["processed_frame_key"])

        if processed_frame_key in completed_keys:
            print(f"Skipping already processed frame: {processed_frame_key}")
            continue

        print(f"\nProcessing frame: {processed_frame_key}")

        try:
            features = process_one_frame(
                row=row,
                vehicle_detector=vehicle_detector,
                road_classifier=road_classifier,
                save_annotated=not args.no_annotated,
            )

            output_rows.append(features)
            completed_keys.add(processed_frame_key)

            print("  Status:", features.get("feature_status"))
            print("  Lens:", features.get("lens_id"))
            print("  Vehicles:", features.get("total_vehicles"))
            print("  Traffic score:", features.get("traffic_load_score"))
            print("  Road ROI available:", features.get("road_roi_available"))
            print("  Road condition:", features.get("road_condition"))
            print("  Road dust score:", features.get("road_dust_score"))

        except Exception as e:
            print(f"  Error: {e}")

            output_rows.append({
                "processed_frame_key": processed_frame_key,
                "source_frame_key": row.get("source_frame_key", ""),
                "matched_run_id": row.get("matched_run_id", ""),
                "video_offset_sec": row.get("video_offset_sec", None),
                "lens_id": row.get("lens_id", None),
                "processed_frame_path": row.get("processed_frame_path", ""),
                "feature_status": "failed",
                "feature_error": str(e),
            })

        processed_since_save += 1

        if processed_since_save >= args.checkpoint_every:
            pd.DataFrame(output_rows).to_csv(OUTPUT_PATH, index=False)
            print(f"\nCheckpoint saved: {OUTPUT_PATH}")
            processed_since_save = 0

    out_df = pd.DataFrame(output_rows)
    out_df.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved ROI-based processed frame visual features to: {OUTPUT_PATH}")

    if not out_df.empty and "feature_status" in out_df.columns:
        print("\nFeature status counts:")
        print(out_df["feature_status"].value_counts())

    if not out_df.empty and "lens_id" in out_df.columns:
        print("\nRows by lens:")
        print(out_df["lens_id"].value_counts())

    if not out_df.empty and "road_roi_available" in out_df.columns:
        print("\nRoad ROI availability:")
        print(out_df["road_roi_available"].value_counts(dropna=False))


if __name__ == "__main__":
    main()