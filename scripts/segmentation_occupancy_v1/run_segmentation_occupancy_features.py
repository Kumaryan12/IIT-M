from pathlib import Path
import argparse

import cv2
import pandas as pd
from ultralytics import YOLO


PROCESSED_FRAME_MANIFEST = Path("outputs/features/processed_frame_manifest.csv")

OUTPUT_PATH = Path("outputs/features/segmentation_occupancy_features_v1.csv")
ANNOTATED_DIR = Path("outputs/segmentation_annotated_v1")

SEG_MODEL = "yolo11n-seg.pt"

VEHICLE_CLASSES = {
    "car",
    "motorcycle",
    "bus",
    "truck",
    "bicycle",
}


def process_frame(model, row, save_annotated=True):
    frame_path = Path(row["processed_frame_path"])
    frame_key = row["processed_frame_key"]
    lens_id = int(row["lens_id"])

    image = cv2.imread(str(frame_path))
    if image is None:
        return {
            "processed_frame_key": frame_key,
            "lens_id": lens_id,
            "feature_status": "failed",
            "feature_error": "could_not_read_image",
        }

    h, w = image.shape[:2]
    image_area = h * w

    results = model(str(frame_path), conf=0.25, verbose=False)
    result = results[0]

    total_vehicle_mask_area = 0
    total_vehicle_boxes = 0

    class_mask_area = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
        "bicycle": 0,
    }

    class_count = {
        "car": 0,
        "motorcycle": 0,
        "bus": 0,
        "truck": 0,
        "bicycle": 0,
    }

    if result.masks is not None and result.boxes is not None:
        masks = result.masks.data.cpu().numpy()
        boxes = result.boxes

        for i, box in enumerate(boxes):
            cls_id = int(box.cls[0])
            class_name = result.names[cls_id]

            if class_name not in VEHICLE_CLASSES:
                continue

            mask = masks[i]
            mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
            mask_binary = mask_resized > 0.5

            area = int(mask_binary.sum())

            total_vehicle_mask_area += area
            total_vehicle_boxes += 1

            class_mask_area[class_name] += area
            class_count[class_name] += 1

    vehicle_occupancy_ratio = total_vehicle_mask_area / image_area

    heavy_vehicle_mask_area = (
        class_mask_area["bus"] + class_mask_area["truck"]
    )

    two_wheeler_mask_area = (
        class_mask_area["motorcycle"] + class_mask_area["bicycle"]
    )

    heavy_vehicle_occupancy_ratio = heavy_vehicle_mask_area / image_area
    two_wheeler_occupancy_ratio = two_wheeler_mask_area / image_area

    if save_annotated:
        ANNOTATED_DIR.mkdir(parents=True, exist_ok=True)
        annotated = result.plot()
        annotated_path = ANNOTATED_DIR / f"{frame_key}_seg.jpg"
        cv2.imwrite(str(annotated_path), annotated)
    else:
        annotated_path = ""

    return {
        "processed_frame_key": frame_key,
        "source_frame_key": row.get("source_frame_key", ""),
        "matched_run_id": row.get("matched_run_id", ""),
        "video_offset_sec": row.get("video_offset_sec", None),
        "lens_id": lens_id,
        "processed_frame_path": str(frame_path),

        "seg_total_vehicle_instances": total_vehicle_boxes,
        "seg_vehicle_mask_area": total_vehicle_mask_area,
        "seg_vehicle_occupancy_ratio": vehicle_occupancy_ratio,

        "seg_car_count": class_count["car"],
        "seg_motorcycle_count": class_count["motorcycle"],
        "seg_bus_count": class_count["bus"],
        "seg_truck_count": class_count["truck"],
        "seg_bicycle_count": class_count["bicycle"],

        "seg_car_mask_area": class_mask_area["car"],
        "seg_motorcycle_mask_area": class_mask_area["motorcycle"],
        "seg_bus_mask_area": class_mask_area["bus"],
        "seg_truck_mask_area": class_mask_area["truck"],
        "seg_bicycle_mask_area": class_mask_area["bicycle"],

        "seg_heavy_vehicle_mask_area": heavy_vehicle_mask_area,
        "seg_two_wheeler_mask_area": two_wheeler_mask_area,
        "seg_heavy_vehicle_occupancy_ratio": heavy_vehicle_occupancy_ratio,
        "seg_two_wheeler_occupancy_ratio": two_wheeler_occupancy_ratio,

        "seg_annotated_path": str(annotated_path),
        "feature_status": "success",
        "feature_error": "",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 4, 6])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    parser.add_argument("--no-annotated", action="store_true")
    args = parser.parse_args()

    if not PROCESSED_FRAME_MANIFEST.exists():
        print(f"Missing file: {PROCESSED_FRAME_MANIFEST}")
        return

    manifest = pd.read_csv(PROCESSED_FRAME_MANIFEST)
    manifest = manifest[manifest["preprocess_status"] == "success"].copy()
    manifest = manifest[manifest["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        manifest = manifest.head(args.limit).copy()

    print("\nSegmentation occupancy extraction")
    print(f"Frames selected: {len(manifest)}")
    print(f"Lenses: {args.lenses}")
    print(f"Model: {SEG_MODEL}")

    existing_rows = []
    done_keys = set()

    if OUTPUT_PATH.exists():
        old = pd.read_csv(OUTPUT_PATH)
        existing_rows = old.to_dict(orient="records")
        if "processed_frame_key" in old.columns:
            done_keys = set(old["processed_frame_key"].astype(str))
        print(f"Loaded existing output: {len(done_keys)} frames")

    model = YOLO(SEG_MODEL)

    rows = existing_rows
    since_save = 0

    for _, row in manifest.iterrows():
        key = str(row["processed_frame_key"])

        if key in done_keys:
            print(f"Skipping: {key}")
            continue

        print(f"Processing: {key}")

        try:
            features = process_frame(
                model=model,
                row=row,
                save_annotated=not args.no_annotated,
            )
        except Exception as e:
            features = {
                "processed_frame_key": key,
                "lens_id": row.get("lens_id", None),
                "feature_status": "failed",
                "feature_error": str(e),
            }

        rows.append(features)
        done_keys.add(key)
        since_save += 1

        if since_save >= args.checkpoint_every:
            pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
            print(f"Checkpoint saved: {OUTPUT_PATH}")
            since_save = 0

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_PATH, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_PATH}")
    print(out["feature_status"].value_counts(dropna=False))

    if "seg_vehicle_occupancy_ratio" in out.columns:
        print("\nOccupancy summary:")
        print(out["seg_vehicle_occupancy_ratio"].describe())


if __name__ == "__main__":
    main()