import argparse
import subprocess
from pathlib import Path

import pandas as pd


ALIGNMENT_CSV = Path("outputs/features/sensor_video_alignment.csv")
OUTPUT_ROOT = Path("outputs/extracted_frames")
OUTPUT_DIR = Path("outputs/features")

UNIQUE_FRAME_MANIFEST = OUTPUT_DIR / "unique_extracted_frames.csv"
SAMPLE_FRAME_MAPPING = OUTPUT_DIR / "sample_frame_mapping.csv"


def check_ffmpeg_available() -> bool:
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        return result.returncode == 0
    except Exception:
        return False


def run_ffmpeg_extract(video_path: Path, offset_sec: float, output_path: Path) -> tuple[bool, str]:
    """
    Extract one frame from a video.

    Safety:
    - Reads the original video only.
    - Writes a new JPG to outputs/extracted_frames/.
    - Uses -n so existing output images are not overwritten.
    - Does not modify, move, delete, rename, or compress the source video.
    """

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.exists():
        return True, "already_exists"

    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel", "error",
        "-ss", str(offset_sec),
        "-i", str(video_path),
        "-frames:v", "1",
        "-q:v", "2",
        "-n",
        str(output_path),
    ]

    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    if result.returncode == 0 and output_path.exists():
        return True, "success"

    return False, result.stderr.strip()


def make_frame_key(run_id: str, offset_sec: float, lens_id: int) -> str:
    """
    Creates a stable key for one unique frame extraction job.

    Offset is rounded to milliseconds to avoid float formatting issues.
    """
    offset_ms = int(round(float(offset_sec) * 1000))
    return f"{run_id}_offsetms_{offset_ms}_lens{lens_id}"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--lenses",
        nargs="+",
        type=int,
        default=[1],
        help="Lens IDs to extract. Example: --lenses 1 2 3"
    )

    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        help="Optional number of matched sensor rows to test. Example: --limit-samples 3"
    )

    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=10,
        help="Save manifests after every N unique frame attempts."
    )

    args = parser.parse_args()

    if not check_ffmpeg_available():
        print("ffmpeg is not available.")
        print("Install ffmpeg first, then rerun this script.")
        return

    if not ALIGNMENT_CSV.exists():
        print(f"Alignment CSV not found: {ALIGNMENT_CSV}")
        print("Run align_sensor_with_video.py first.")
        return

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    alignment_df = pd.read_csv(ALIGNMENT_CSV)

    if "video_match_status" not in alignment_df.columns:
        print("sensor_video_alignment.csv does not contain video_match_status.")
        return

    matched_df = alignment_df[alignment_df["video_match_status"] == "matched"].copy()

    if args.limit_samples is not None:
        matched_df = matched_df.head(args.limit_samples)

    print("\nFrame extraction setup:")
    print(f"Total matched rows available: {len(alignment_df[alignment_df['video_match_status'] == 'matched'])}")
    print(f"Matched rows selected now: {len(matched_df)}")
    print(f"Lenses selected: {args.lenses}")
    print(f"Output folder: {OUTPUT_ROOT}")

    # Build sample-to-frame mapping and unique extraction jobs
    mapping_rows = []
    unique_jobs = {}

    for _, row in matched_df.iterrows():
        sample_id = int(row["sample_id"])
        timestamp = str(row["timestamp"])
        run_id = str(row["matched_run_id"])
        offset_sec = float(row["video_offset_sec"])

        for lens_id in args.lenses:
            lens_path_col = f"lens{lens_id}_local_path"

            if lens_path_col not in row:
                continue

            video_path = Path(str(row[lens_path_col]))
            frame_key = make_frame_key(run_id, offset_sec, lens_id)

            output_path = (
                OUTPUT_ROOT
                / run_id
                / f"lens{lens_id}"
                / f"{frame_key}.jpg"
            )

            mapping_rows.append({
                "sample_id": sample_id,
                "timestamp": timestamp,
                "matched_run_id": run_id,
                "video_offset_sec": offset_sec,
                "lens_id": lens_id,
                "frame_key": frame_key,
                "video_path": str(video_path),
                "frame_path": str(output_path),
            })

            if frame_key not in unique_jobs:
                unique_jobs[frame_key] = {
                    "frame_key": frame_key,
                    "matched_run_id": run_id,
                    "video_offset_sec": offset_sec,
                    "lens_id": lens_id,
                    "video_path": str(video_path),
                    "frame_path": str(output_path),
                }

    unique_jobs_list = list(unique_jobs.values())

    print(f"\nSample-frame mapping rows: {len(mapping_rows)}")
    print(f"Unique frame extraction jobs: {len(unique_jobs_list)}")

    unique_results = []
    attempted_since_save = 0

    # If old unique manifest exists, resume from it
    completed_frame_keys = set()

    if UNIQUE_FRAME_MANIFEST.exists():
        old_manifest = pd.read_csv(UNIQUE_FRAME_MANIFEST)

        if "frame_key" in old_manifest.columns:
            completed_frame_keys = set(
                old_manifest[
                    old_manifest["extract_status"].isin(["success", "already_exists"])
                ]["frame_key"].astype(str)
            )

        unique_results = old_manifest.to_dict(orient="records")

        print(f"\nLoaded existing unique frame manifest: {UNIQUE_FRAME_MANIFEST}")
        print(f"Completed frame keys loaded: {len(completed_frame_keys)}")

    for idx, job in enumerate(unique_jobs_list, start=1):
        frame_key = job["frame_key"]

        if frame_key in completed_frame_keys:
            print(f"[{idx}/{len(unique_jobs_list)}] Skipping already extracted: {frame_key}")
            continue

        video_path = Path(job["video_path"])
        output_path = Path(job["frame_path"])
        offset_sec = float(job["video_offset_sec"])

        print(f"\n[{idx}/{len(unique_jobs_list)}] Extracting frame")
        print(f"Frame key: {frame_key}")
        print(f"Video: {video_path}")
        print(f"Offset: {offset_sec:.3f} sec")
        print(f"Output: {output_path}")

        if not video_path.exists():
            success = False
            status = "video_not_found"
            error_message = f"Video file not found: {video_path}"
            print(f"  Failed: {error_message}")

        else:
            success, status = run_ffmpeg_extract(
                video_path=video_path,
                offset_sec=offset_sec,
                output_path=output_path,
            )
            error_message = "" if success else status

            if success:
                print(f"  Success: {status}")
            else:
                print(f"  Failed: {status}")

        extract_status = "success" if success else "failed"

        result_row = {
            **job,
            "extract_status": extract_status,
            "extract_message": status,
            "error_message": error_message,
        }

        unique_results.append(result_row)

        attempted_since_save += 1

        if attempted_since_save >= args.checkpoint_every:
            pd.DataFrame(unique_results).to_csv(UNIQUE_FRAME_MANIFEST, index=False)
            pd.DataFrame(mapping_rows).to_csv(SAMPLE_FRAME_MAPPING, index=False)
            print(f"\nCheckpoint saved:")
            print(f"  {UNIQUE_FRAME_MANIFEST}")
            print(f"  {SAMPLE_FRAME_MAPPING}")
            attempted_since_save = 0

    unique_df = pd.DataFrame(unique_results)
    mapping_df = pd.DataFrame(mapping_rows)

    unique_df.to_csv(UNIQUE_FRAME_MANIFEST, index=False)
    mapping_df.to_csv(SAMPLE_FRAME_MAPPING, index=False)

    print("\nDone.")
    print(f"Unique frame manifest saved to: {UNIQUE_FRAME_MANIFEST}")
    print(f"Sample-frame mapping saved to: {SAMPLE_FRAME_MAPPING}")

    if not unique_df.empty:
        print("\nUnique extraction status counts:")
        print(unique_df["extract_status"].value_counts())

    print("\nReminder:")
    print("Original video files were only read. Extracted JPG frames were written to outputs/extracted_frames/.")


if __name__ == "__main__":
    main()