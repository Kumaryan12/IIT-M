from typing import Optional
import cv2
from ultralytics import YOLO
from src.config import YOLO_MODEL_NAME, VEHICLE_CLASSES, VEHICLE_CONF_THRESHOLD


class VehicleDetector:
    def __init__(self, model_name: str = YOLO_MODEL_NAME):
        self.model = YOLO(model_name)

    def detect(self, image_path: str, save_annotated_path: Optional[str] = None) -> dict:
        results = self.model(
    image_path,
    conf=VEHICLE_CONF_THRESHOLD,
    device="mps"
)
        result = results[0]

        counts = {
            "car_count": 0,
            "motorcycle_count": 0,
            "bus_count": 0,
            "truck_count": 0,
            "bicycle_count": 0
        }

        detections = []

        image = cv2.imread(image_path)

        if result.boxes is not None:
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                class_name = result.names[cls_id]

                if class_name not in VEHICLE_CLASSES:
                    continue

                key = f"{class_name}_count"
                counts[key] += 1

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                detections.append({
                    "class_name": class_name,
                    "confidence": conf,
                    "bbox_xyxy": [x1, y1, x2, y2]
                })

                if image is not None and save_annotated_path is not None:
                    label = f"{class_name} {conf:.2f}"

                    cv2.rectangle(
                        image,
                        (x1, y1),
                        (x2, y2),
                        (255, 255, 255),
                        2
                    )

                    cv2.putText(
                        image,
                        label,
                        (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        2
                    )

        total_vehicles = sum(counts.values())

        if image is not None and save_annotated_path is not None:
            cv2.imwrite(save_annotated_path, image)

        return {
            "total_vehicles": total_vehicles,
            **counts,
            "vehicle_detections": detections
        }