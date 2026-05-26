from pathlib import Path

import pandas as pd


MANIFEST_CSV = Path("outputs/features/image_density_manifest_v1.csv")
EMBEDDINGS_CSV = Path("outputs/features/image_clip_embeddings_v1.csv")

OUTPUT_CSV = Path("outputs/features/sample_clip_image_embeddings_v1.csv")

TARGET_COL = "effective_density_kg_m3"


def main():
    if not MANIFEST_CSV.exists():
        print(f"Missing manifest: {MANIFEST_CSV}")
        return

    if not EMBEDDINGS_CSV.exists():
        print(f"Missing embeddings: {EMBEDDINGS_CSV}")
        return

    manifest = pd.read_csv(MANIFEST_CSV)
    embeddings = pd.read_csv(EMBEDDINGS_CSV)

    print("\nLoaded:")
    print(f"Manifest: {manifest.shape}")
    print(f"Embeddings: {embeddings.shape}")

    embeddings = embeddings[
        embeddings["embedding_status"] == "success"
    ].copy()

    emb_cols = [
        c for c in embeddings.columns
        if c.startswith("clip_") and c[5:].isdigit()
    ]

    print(f"\nEmbedding columns: {len(emb_cols)}")

    emb_keep = ["frame_key"] + emb_cols

    merged = manifest.merge(
        embeddings[emb_keep],
        on="frame_key",
        how="inner",
    )

    print("\nMerged manifest + embeddings:")
    print(merged.shape)

    if merged.empty:
        print("No rows matched. Check frame_key.")
        return

    grouped = (
        merged
        .groupby(["sample_id", "timestamp"], as_index=False)
        .agg(
            matched_run_id=("matched_run_id", "first"),
            effective_density_kg_m3=(TARGET_COL, "first"),
            num_lens_images_used=("lens_id", "nunique"),
            lenses_used=(
                "lens_id",
                lambda x: ",".join(map(str, sorted(set(x))))
            ),
            **{
                f"image_clip_mean_{i}": (f"clip_{i}", "mean")
                for i in range(len(emb_cols))
            }
        )
    )

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
                "effective_density_kg_m3",
                "num_lens_images_used",
                "lenses_used",
            ]
        ].head(10).to_string(index=False)
    )


if __name__ == "__main__":
    main()