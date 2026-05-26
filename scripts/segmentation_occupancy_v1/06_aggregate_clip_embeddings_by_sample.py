from pathlib import Path

import pandas as pd


MAPPING_CSV = Path("outputs/features/sample_frame_mapping.csv")
CLIP_CSV = Path("outputs/features/clip_frame_embeddings_v1.csv")
OUTPUT_CSV = Path("outputs/features/sample_clip_embeddings_v1.csv")


def main():
    if not MAPPING_CSV.exists():
        print(f"Missing mapping file: {MAPPING_CSV}")
        return

    if not CLIP_CSV.exists():
        print(f"Missing CLIP embedding file: {CLIP_CSV}")
        return

    mapping = pd.read_csv(MAPPING_CSV)
    clip = pd.read_csv(CLIP_CSV)

    print("\nInput files:")
    print(f"Mapping rows: {mapping.shape}")
    print(f"CLIP rows: {clip.shape}")

    clip = clip[clip["clip_status"] == "success"].copy()

    emb_cols = [
        c for c in clip.columns
        if c.startswith("clip_") and c[5:].isdigit()
    ]

    print(f"Embedding columns: {len(emb_cols)}")

    merged = mapping.merge(
        clip,
        left_on="frame_key",
        right_on="source_frame_key",
        how="inner",
        suffixes=("_map", "_clip"),
    )

    print(f"Merged rows: {merged.shape}")

    if merged.empty:
        print("No rows matched.")
        return

    agg_dict = {
        "lens_id_map": "nunique",
    }

    for col in emb_cols:
        agg_dict[col] = "mean"

    grouped = (
        merged
        .groupby(["sample_id", "timestamp", "matched_run_id_map"], as_index=False)
        .agg(agg_dict)
    )

    grouped = grouped.rename(
        columns={
            "matched_run_id_map": "matched_run_id",
            "lens_id_map": "num_clip_lens_frames_used",
        }
    )

    grouped["clip_lenses_used"] = (
        merged
        .groupby(["sample_id", "timestamp"])["lens_id_map"]
        .apply(lambda x: ",".join(map(str, sorted(set(x)))))
        .values
    )

    # Rename embeddings as sample-level CLIP means
    rename_map = {
        col: f"sample_clip_mean_{col.split('_')[1]}"
        for col in emb_cols
    }

    grouped = grouped.rename(columns=rename_map)

    grouped.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(f"Output shape: {grouped.shape}")

    print("\nPreview:")
    print(
        grouped[
            [
                "sample_id",
                "timestamp",
                "matched_run_id",
                "num_clip_lens_frames_used",
                "clip_lenses_used",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()