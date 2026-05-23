import argparse
import json
from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs/features")
OUTPUT_PATH = OUTPUT_DIR / "video_run_index.csv"


def load_capture_stats(capture_stats_path: Path) -> dict:
    """
    Reads capture_stats.json for one video run.

    This function only reads metadata.
    It does not modify any video file.
    """

    with open(capture_stats_path, "r") as f:
        meta = json.load(f)

    run_folder = capture_stats_path.parent
    run_id = meta.get("run_id", run_folder.name)

    start_utc = pd.to_datetime(meta["mic_start_time_utc"], utc=True)
    end_utc = pd.to_datetime(meta["mic_end_time_utc"], utc=True)

    start_ist = start_utc.tz_convert("Asia/Kolkata").tz_localize(None)
    end_ist = end_utc.tz_convert("Asia/Kolkata").tz_localize(None)

    duration_sec = float(
        meta.get(
            "total_duration_sec_estimate",
            (end_utc - start_utc).total_seconds()
        )
    )

    avg_fps = float(meta.get("avg_fps_over_lenses", 30.0))
    per_lens = meta.get("per_lens", {})

    row = {
        "run_id": run_id,
        "run_folder": str(run_folder),

        "video_start_utc": start_utc.isoformat(),
        "video_end_utc": end_utc.isoformat(),

        "video_start_ist": start_ist,
        "video_end_ist": end_ist,

        "duration_sec": duration_sec,
        "avg_fps_over_lenses": avg_fps,
        "container": meta.get("container", ""),
    }

    for lens_id in range(1, 7):
        lens_key = str(lens_id)

        video_path = run_folder / f"LENS{lens_id}" / f"video_lens{lens_id}.mp4"

        row[f"lens{lens_id}_local_path"] = str(video_path)
        row[f"lens{lens_id}_exists"] = video_path.exists()

        if lens_key in per_lens:
            row[f"lens{lens_id}_duration_sec"] = per_lens[lens_key].get("duration_sec", None)
            row[f"lens{lens_id}_avg_fps"] = per_lens[lens_key].get("avg_fps", None)
        else:
            row[f"lens{lens_id}_duration_sec"] = None
            row[f"lens{lens_id}_avg_fps"] = None

    return row


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--video-root",
        required=True,
        help="External hard-disk folder containing run_* folders"
    )

    args = parser.parse_args()

    video_root = Path(args.video_root)

    if not video_root.exists():
        print(f"Video root not found: {video_root}")
        return

    capture_files = sorted(video_root.glob("run_*/capture_stats.json"))

    if not capture_files:
        print(f"No capture_stats.json files found under: {video_root}")
        print("Expected structure:")
        print("video_root/run_xxxxx/capture_stats.json")
        return

    rows = []

    for capture_path in capture_files:
        print(f"Reading metadata: {capture_path}")

        try:
            row = load_capture_stats(capture_path)
            rows.append(row)

        except Exception as e:
            print(f"  Error reading {capture_path}: {e}")

    if not rows:
        print("No valid video runs found.")
        return

    df = pd.DataFrame(rows)
    df = df.sort_values("video_start_ist").reset_index(drop=True)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)

    print("\nVideo run index created successfully.")
    print(f"Total video runs indexed: {len(df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    display_cols = [
        "run_id",
        "video_start_ist",
        "video_end_ist",
        "duration_sec",
        "lens1_exists",
        "lens2_exists",
        "lens3_exists",
        "lens4_exists",
        "lens5_exists",
        "lens6_exists",
    ]

    print("\nPreview:")
    print(df[display_cols].to_string(index=False))


if __name__ == "__main__":
    main()