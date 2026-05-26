from pathlib import Path
import pandas as pd


MANIFEST_CSV = Path("outputs/features/image_density_manifest_v1.csv")
EMBEDDINGS_CSV = Path("outputs/features/image_dinov2_embeddings_v1.csv")
OUTPUT_CSV = Path("outputs/features/sample_dinov2_image_embeddings_v1.csv")

TARGET_COL = "effective_density_kg_m3"


def main():
    manifest = pd.read_csv(MANIFEST_CSV)
    embeddings = pd.read_csv(EMBEDDINGS_CSV)

    print("\nLoaded:")
    print("Manifest:", manifest.shape)
    print("DINO embeddings:", embeddings.shape)

    embeddings = embeddings[embeddings["embedding_status"] == "success"].copy()

    emb_cols = [
        c for c in embeddings.columns
        if c.startswith("dino_") and c[5:].isdigit()
    ]

    print(f"DINO embedding columns: {len(emb_cols)}")

    merged = manifest.merge(
        embeddings[["frame_key"] + emb_cols],
        on="frame_key",
        how="inner",
    )

    print("Merged:", merged.shape)

    grouped = (
        merged
        .groupby(["sample_id", "timestamp"], as_index=False)
        .agg(
            matched_run_id=("matched_run_id", "first"),
            effective_density_kg_m3=(TARGET_COL, "first"),
            num_lens_images_used=("lens_id", "nunique"),
            lenses_used=("lens_id", lambda x: ",".join(map(str, sorted(set(x))))),
            **{
                f"image_dino_mean_{i}": (f"dino_{i}", "mean")
                for i in range(len(emb_cols))
            }
        )
    )

    grouped.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")
    print("Shape:", grouped.shape)


if __name__ == "__main__":
    main()