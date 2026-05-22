from pathlib import Path
from typing import Union

from src.vehicle_detector import VehicleDetector
from src.road_classifier import RoadClassifier
from src.image_features import extract_basic_image_features
from src.config import ANNOTATED_DIR


class ImageFeaturePipeline:
    def __init__(self):
        self.vehicle_detector = VehicleDetector()
        self.road_classifier = RoadClassifier()

    def process_image(self, image_path: Union[str, Path], save_annotated: bool = True) -> dict:
        image_path = Path(image_path)

        annotated_path = None
        if save_annotated:
            ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
            annotated_path = ANNOTATED_DIR / f"annotated_{image_path.name}"

        # 1. Vehicle detection
        vehicle_features = self.vehicle_detector.detect(
            image_path=str(image_path),
            save_annotated_path=str(annotated_path) if annotated_path else None
        )

        # 2. Road classification
        road_features = self.road_classifier.classify(str(image_path))

        # 3. Basic visual image features
        basic_image_features = extract_basic_image_features(image_path)

        # -----------------------------
        # Vehicle count features
        # -----------------------------
        total_vehicles = vehicle_features["total_vehicles"]
        car_count = vehicle_features["car_count"]
        motorcycle_count = vehicle_features["motorcycle_count"]
        bus_count = vehicle_features["bus_count"]
        truck_count = vehicle_features["truck_count"]
        bicycle_count = vehicle_features["bicycle_count"]

        # -----------------------------
        # Detection quality and occupancy features
        # -----------------------------
        vehicle_box_area_ratio = vehicle_features["vehicle_box_area_ratio"]
        average_vehicle_confidence = vehicle_features["average_vehicle_confidence"]
        small_vehicle_count = vehicle_features["small_vehicle_count"]
        medium_vehicle_count = vehicle_features["medium_vehicle_count"]
        large_vehicle_count = vehicle_features["large_vehicle_count"]

        # -----------------------------
        # Sliced detection debug features
        # -----------------------------
        full_source_detection_count = vehicle_features["full_source_detection_count"]
        tile_source_detection_count = vehicle_features["tile_source_detection_count"]
        raw_detection_count_before_merge = vehicle_features["raw_detection_count_before_merge"]
        final_detection_count_after_merge = vehicle_features["final_detection_count_after_merge"]

        # -----------------------------
        # Road features from CLIP
        # -----------------------------
        road_condition = road_features["road_condition"]
        road_condition_full_label = road_features["road_condition_full_label"]
        road_condition_confidence = road_features["road_condition_confidence"]
        clip_road_dust_score = road_features["clip_road_dust_score"]

        # -----------------------------
        # Visual dust features from OpenCV
        # -----------------------------
        brightness_mean = basic_image_features["brightness_mean"]
        contrast_std = basic_image_features["contrast_std"]
        brown_pixel_ratio = basic_image_features["brown_pixel_ratio"]
        edge_density = basic_image_features["edge_density"]
        haze_score = basic_image_features["haze_score"]
        visual_dust_score = basic_image_features["visual_dust_score"]

        # -----------------------------
        # Final combined dust score
        # -----------------------------
        # CLIP gives semantic road understanding.
        # OpenCV gives visual color/texture/haze evidence.
        road_dust_score = (
            0.70 * clip_road_dust_score
            + 0.30 * visual_dust_score
        )

        if road_dust_score <= 0.30:
            road_dust_level = "low"
        elif road_dust_score <= 0.65:
            road_dust_level = "moderate"
        else:
            road_dust_level = "high"

        # -----------------------------
        # Engineered vehicle features
        # -----------------------------
        heavy_vehicle_count = bus_count + truck_count
        two_wheeler_count = motorcycle_count + bicycle_count
        motor_vehicle_count = car_count + motorcycle_count + bus_count + truck_count
        non_motor_vehicle_count = bicycle_count

        # -----------------------------
        # Vehicle ratios
        # -----------------------------
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

        # -----------------------------
        # Weighted traffic score
        # -----------------------------
        traffic_load_score = (
            1.0 * car_count
            + 0.6 * motorcycle_count
            + 3.0 * bus_count
            + 3.5 * truck_count
            + 0.1 * bicycle_count
        )

        # -----------------------------
        # Interaction features
        # -----------------------------
        dust_traffic_interaction_score = traffic_load_score * road_dust_score
        heavy_vehicle_dust_score = heavy_vehicle_count * road_dust_score

        # -----------------------------
        # Traffic level category
        # -----------------------------
        if total_vehicles == 0:
            traffic_level = "no_traffic"
        elif total_vehicles <= 5:
            traffic_level = "low"
        elif total_vehicles <= 15:
            traffic_level = "moderate"
        else:
            traffic_level = "high"

        features = {
            # Image information
            "image_name": image_path.name,
            "image_path": str(image_path),

            # Basic vehicle features
            "total_vehicles": total_vehicles,
            "car_count": car_count,
            "motorcycle_count": motorcycle_count,
            "bus_count": bus_count,
            "truck_count": truck_count,
            "bicycle_count": bicycle_count,

            # Detection quality and occupancy features
            "vehicle_box_area_ratio": vehicle_box_area_ratio,
            "average_vehicle_confidence": average_vehicle_confidence,
            "small_vehicle_count": small_vehicle_count,
            "medium_vehicle_count": medium_vehicle_count,
            "large_vehicle_count": large_vehicle_count,

            # Sliced detection debug features
            "full_source_detection_count": full_source_detection_count,
            "tile_source_detection_count": tile_source_detection_count,
            "raw_detection_count_before_merge": raw_detection_count_before_merge,
            "final_detection_count_after_merge": final_detection_count_after_merge,

            # Engineered vehicle features
            "heavy_vehicle_count": heavy_vehicle_count,
            "two_wheeler_count": two_wheeler_count,
            "motor_vehicle_count": motor_vehicle_count,
            "non_motor_vehicle_count": non_motor_vehicle_count,

            # Vehicle ratios
            "heavy_vehicle_ratio": heavy_vehicle_ratio,
            "two_wheeler_ratio": two_wheeler_ratio,
            "motor_vehicle_ratio": motor_vehicle_ratio,
            "car_ratio": car_ratio,
            "truck_ratio": truck_ratio,
            "bus_ratio": bus_ratio,

            # Traffic intensity features
            "traffic_load_score": traffic_load_score,
            "traffic_level": traffic_level,

            # Road classifier features
            "road_condition": road_condition,
            "road_condition_full_label": road_condition_full_label,
            "road_condition_confidence": road_condition_confidence,
            "clip_road_dust_score": clip_road_dust_score,

            # OpenCV visual image features
            "brightness_mean": brightness_mean,
            "contrast_std": contrast_std,
            "brown_pixel_ratio": brown_pixel_ratio,
            "edge_density": edge_density,
            "haze_score": haze_score,
            "visual_dust_score": visual_dust_score,

            # Final combined road dust features
            "road_dust_score": road_dust_score,
            "road_dust_level": road_dust_level,

            # Interaction features
            "dust_traffic_interaction_score": dust_traffic_interaction_score,
            "heavy_vehicle_dust_score": heavy_vehicle_dust_score,

            # Output path
            "annotated_image_path": str(annotated_path) if annotated_path else None
        }

        return features