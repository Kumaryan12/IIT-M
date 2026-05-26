import json
from pathlib import Path

import pandas as pd


VIDEO_META_PATH = Path("data/video/run_20260223_110646_1346/capture_stats.json")
SENSOR_CSV_PATH = Path("outputs/features/clean_sensor_data.csv")


def load_video_window(meta_path: Path):
    with open(meta_path, "r") as f:
        meta = json.load(f)

    run_id = meta.get("run_id", "unknown_run")

    video_start_utc = pd.to_datetime(meta["mic_start_time_utc"], utc=True)
    video_end_utc = pd.to_datetime(meta["mic_end_time_utc"], utc=True)

    video_start_ist = video_start_utc.tz_convert("Asia/Kolkata")
    video_end_ist = video_end_utc.tz_convert("Asia/Kolkata")

    duration_sec = (video_end_utc - video_start_utc).total_seconds()

    return {
        "run_id": run_id,
        "video_start_utc": video_start_utc,
        "video_end_utc": video_end_utc,
        "video_start_ist": video_start_ist,
        "video_end_ist": video_end_ist,
        "duration_sec": duration_sec,
    }


def load_sensor_window(sensor_csv_path: Path):
    df = pd.read_csv(sensor_csv_path)

    if "timestamp" not in df.columns:
        raise ValueError("Sensor CSV must contain a 'timestamp' column.")

    # Cleaned CSV usually has ISO-like timestamp.
    # Raw CSV may have dd-mm-yyyy HH:MM.
    sensor_time = pd.to_datetime(df["timestamp"], errors="coerce", dayfirst=True)

    if sensor_time.isna().all():
        raise ValueError("Could not parse sensor timestamps.")

    sensor_start = sensor_time.min()
    sensor_end = sensor_time.max()

    return {
        "sensor_start": sensor_start,
        "sensor_end": sensor_end,
        "sensor_rows": len(df),
    }


def check_overlap(video_start, video_end, sensor_start, sensor_end):
    """
    All inputs should be timezone-compatible or timezone-naive consistently.
    Here we compare using naive IST times.
    """

    latest_start = max(video_start, sensor_start)
    earliest_end = min(video_end, sensor_end)

    overlap_sec = (earliest_end - latest_start).total_seconds()

    if overlap_sec > 0:
        return True, overlap_sec

    # If no overlap, calculate nearest gap.
    if sensor_start > video_end:
        gap_sec = (sensor_start - video_end).total_seconds()
    else:
        gap_sec = (video_start - sensor_end).total_seconds()

    return False, -gap_sec


def main():
    if not VIDEO_META_PATH.exists():
        print(f"Video metadata not found: {VIDEO_META_PATH}")
        print("Update VIDEO_META_PATH in the script.")
        return

    if not SENSOR_CSV_PATH.exists():
        print(f"Sensor CSV not found: {SENSOR_CSV_PATH}")
        print("Run clean_sensor_data.py first or update SENSOR_CSV_PATH.")
        return

    video = load_video_window(VIDEO_META_PATH)
    sensor = load_sensor_window(SENSOR_CSV_PATH)

    # Convert video IST timestamps to timezone-naive for comparison with sensor timestamps.
    video_start_ist_naive = video["video_start_ist"].tz_localize(None)
    video_end_ist_naive = video["video_end_ist"].tz_localize(None)

    sensor_start = sensor["sensor_start"]
    sensor_end = sensor["sensor_end"]

    # If sensor timestamps are timezone-aware, remove timezone for comparison.
    if getattr(sensor_start, "tzinfo", None) is not None:
        sensor_start = sensor_start.tz_localize(None)
        sensor_end = sensor_end.tz_localize(None)

    has_overlap, value_sec = check_overlap(
        video_start_ist_naive,
        video_end_ist_naive,
        sensor_start,
        sensor_end,
    )

    print("\nVideo Run:")
    print(f"run_id: {video['run_id']}")
    print(f"video_start_utc: {video['video_start_utc']}")
    print(f"video_end_utc:   {video['video_end_utc']}")
    print(f"video_start_ist: {video_start_ist_naive}")
    print(f"video_end_ist:   {video_end_ist_naive}")
    print(f"duration_sec:    {video['duration_sec']:.2f}")

    print("\nSensor Window:")
    print(f"sensor_start: {sensor_start}")
    print(f"sensor_end:   {sensor_end}")
    print(f"sensor_rows:  {sensor['sensor_rows']}")

    print("\nAlignment Result:")
    if has_overlap:
        print("STATUS: OVERLAP FOUND")
        print(f"overlap_sec: {value_sec:.2f}")
        print(f"overlap_min: {value_sec / 60:.2f}")
    else:
        print("STATUS: NO OVERLAP")
        print(f"nearest_gap_sec: {-value_sec:.2f}")
        print(f"nearest_gap_min: {-value_sec / 60:.2f}")

        if sensor_start > video_end_ist_naive:
            print("Reason: sensor data starts after video ends.")
        else:
            print("Reason: video starts after sensor data ends.")

    print("\nRecommendation:")
    if has_overlap:
        print("This video run can be aligned with this sensor CSV.")
    else:
        print("Do not extract final features from this video for this sensor CSV yet.")
        print("Find the matching video run, or confirm timestamp timezone/clock offset.")


if __name__ == "__main__":
    main()