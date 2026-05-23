from pathlib import Path

import pandas as pd


SENSOR_CSV_PATH = Path("outputs/features/clean_sensor_data.csv")
VIDEO_INDEX_PATH = Path("outputs/features/video_run_index.csv")

OUTPUT_DIR = Path("outputs/features")
OUTPUT_PATH = OUTPUT_DIR / "sensor_video_alignment.csv"


def main():
    if not SENSOR_CSV_PATH.exists():
        print(f"Sensor CSV not found: {SENSOR_CSV_PATH}")
        print("You need outputs/features/clean_sensor_data.csv before alignment.")
        return

    if not VIDEO_INDEX_PATH.exists():
        print(f"Video index not found: {VIDEO_INDEX_PATH}")
        print("Run build_video_run_index.py first.")
        return

    sensor_df = pd.read_csv(SENSOR_CSV_PATH)
    video_df = pd.read_csv(VIDEO_INDEX_PATH)

    if "timestamp" not in sensor_df.columns:
        print("Sensor CSV must contain a timestamp column.")
        print("Available columns:")
        print(list(sensor_df.columns))
        return

    sensor_df["timestamp"] = pd.to_datetime(
        sensor_df["timestamp"],
        errors="coerce"
    )

    video_df["video_start_ist"] = pd.to_datetime(
        video_df["video_start_ist"],
        errors="coerce"
    )

    video_df["video_end_ist"] = pd.to_datetime(
        video_df["video_end_ist"],
        errors="coerce"
    )

    aligned_rows = []

    for _, sensor_row in sensor_df.iterrows():
        sensor_time = sensor_row["timestamp"]
        out = sensor_row.to_dict()

        if pd.isna(sensor_time):
            out.update({
                "video_match_status": "invalid_sensor_timestamp",
                "matched_run_id": "",
                "matched_run_folder": "",
                "video_start_ist": "",
                "video_end_ist": "",
                "video_offset_sec": None,
                "video_time_gap_sec": None,
            })

            for lens_id in range(1, 7):
                out[f"lens{lens_id}_local_path"] = ""
                out[f"lens{lens_id}_exists"] = False

            aligned_rows.append(out)
            continue

        matching_runs = video_df[
            (video_df["video_start_ist"] <= sensor_time) &
            (video_df["video_end_ist"] >= sensor_time)
        ]

        if len(matching_runs) > 0:
            # If more than one run matches, choose the latest-starting one.
            match = matching_runs.sort_values("video_start_ist").iloc[-1]

            offset_sec = (sensor_time - match["video_start_ist"]).total_seconds()

            out.update({
                "video_match_status": "matched",
                "matched_run_id": match["run_id"],
                "matched_run_folder": match["run_folder"],
                "video_start_ist": match["video_start_ist"],
                "video_end_ist": match["video_end_ist"],
                "video_offset_sec": offset_sec,
                "video_time_gap_sec": 0.0,
                "nearest_run_id": "",
                "nearest_run_start_ist": "",
                "nearest_run_end_ist": "",
            })

            for lens_id in range(1, 7):
                out[f"lens{lens_id}_local_path"] = match.get(
                    f"lens{lens_id}_local_path",
                    ""
                )
                out[f"lens{lens_id}_exists"] = match.get(
                    f"lens{lens_id}_exists",
                    False
                )

        else:
            # No direct match. Find nearest video run for debugging.
            gaps = []

            for _, run in video_df.iterrows():
                if sensor_time < run["video_start_ist"]:
                    gap = (run["video_start_ist"] - sensor_time).total_seconds()
                elif sensor_time > run["video_end_ist"]:
                    gap = (sensor_time - run["video_end_ist"]).total_seconds()
                else:
                    gap = 0.0

                gaps.append(gap)

            nearest_idx = int(pd.Series(gaps).idxmin())
            nearest_run = video_df.iloc[nearest_idx]
            nearest_gap = float(gaps[nearest_idx])

            out.update({
                "video_match_status": "no_match",
                "matched_run_id": "",
                "matched_run_folder": "",
                "video_start_ist": "",
                "video_end_ist": "",
                "video_offset_sec": None,
                "video_time_gap_sec": nearest_gap,
                "nearest_run_id": nearest_run["run_id"],
                "nearest_run_start_ist": nearest_run["video_start_ist"],
                "nearest_run_end_ist": nearest_run["video_end_ist"],
            })

            for lens_id in range(1, 7):
                out[f"lens{lens_id}_local_path"] = ""
                out[f"lens{lens_id}_exists"] = False

        aligned_rows.append(out)

    aligned_df = pd.DataFrame(aligned_rows)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    aligned_df.to_csv(OUTPUT_PATH, index=False)

    print("\nSensor-video alignment completed.")
    print(f"Sensor rows: {len(sensor_df)}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nMatch status counts:")
    print(aligned_df["video_match_status"].value_counts())

    matched_df = aligned_df[aligned_df["video_match_status"] == "matched"]

    if len(matched_df) > 0:
        print("\nMatched run counts:")
        print(matched_df["matched_run_id"].value_counts())

        print("\nVideo offset summary in seconds:")
        print(matched_df["video_offset_sec"].describe())

    no_match_df = aligned_df[aligned_df["video_match_status"] == "no_match"]

    if len(no_match_df) > 0:
        print("\nNo-match rows summary:")
        print("No-match count:", len(no_match_df))

        print("\nNearest gap seconds:")
        print(no_match_df["video_time_gap_sec"].describe())

        print("\nFirst few no-match rows:")
        display_cols = [
            "sample_id",
            "timestamp",
            "video_match_status",
            "video_time_gap_sec",
            "nearest_run_id",
            "nearest_run_start_ist",
            "nearest_run_end_ist",
        ]

        display_cols = [c for c in display_cols if c in no_match_df.columns]
        print(no_match_df[display_cols].head(20).to_string(index=False))


if __name__ == "__main__":
    main()