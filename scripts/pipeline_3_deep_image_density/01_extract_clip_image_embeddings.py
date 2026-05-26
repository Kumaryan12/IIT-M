from pathlib import Path
import argparse

import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


INPUT_CSV = Path("outputs/features/image_density_manifest_v1.csv")
OUTPUT_CSV = Path("outputs/features/image_clip_embeddings_v1.csv")

MODEL_NAME = "openai/clip-vit-base-patch32"


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_image(path):
    return Image.open(path).convert("RGB")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    df = pd.read_csv(INPUT_CSV)

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("\nImage-only CLIP embedding extraction")
    print(f"Rows: {len(df)}")
    print(f"Model: {MODEL_NAME}")

    device = get_device()
    print(f"Device: {device}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    rows = []
    done_keys = set()

    if OUTPUT_CSV.exists():
        old = pd.read_csv(OUTPUT_CSV)
        rows = old.to_dict(orient="records")
        done_keys = set(old["frame_key"].astype(str))
        print(f"Loaded existing rows: {len(done_keys)}")
        original_rows = len(df)

        df = df.drop_duplicates(subset=["frame_key"]).copy()

        print(f"Original manifest rows: {original_rows}")
        print(f"Unique frame rows: {len(df)}")

    with torch.no_grad():
        for _, row in df.iterrows():
            frame_key = str(row["frame_key"])

            if frame_key in done_keys:
                print(f"Skipping: {frame_key}")
                continue

            print(f"Processing: {frame_key}")

            try:
                image = load_image(row["frame_path"])

                inputs = processor(
                    images=image,
                    return_tensors="pt",
                )

                inputs = {
                    k: v.to(device)
                    for k, v in inputs.items()
                }

                emb = model.get_image_features(**inputs)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                emb = emb[0].detach().cpu().numpy()

                out = {
                    "sample_id": row["sample_id"],
                    "timestamp": row["timestamp"],
                    "matched_run_id": row["matched_run_id"],
                    "video_offset_sec": row["video_offset_sec"],
                    "lens_id": row["lens_id"],
                    "frame_key": frame_key,
                    "frame_path": row["frame_path"],
                    "effective_density_kg_m3": row["effective_density_kg_m3"],
                    "embedding_status": "success",
                    "embedding_error": "",
                }

                for i, val in enumerate(emb):
                    out[f"clip_{i}"] = float(val)

            except Exception as e:
                out = {
                    "sample_id": row.get("sample_id", None),
                    "timestamp": row.get("timestamp", None),
                    "matched_run_id": row.get("matched_run_id", None),
                    "video_offset_sec": row.get("video_offset_sec", None),
                    "lens_id": row.get("lens_id", None),
                    "frame_key": frame_key,
                    "frame_path": row.get("frame_path", ""),
                    "effective_density_kg_m3": row.get("effective_density_kg_m3", None),
                    "embedding_status": "failed",
                    "embedding_error": str(e),
                }

            rows.append(out)
            done_keys.add(frame_key)

            if len(rows) % args.checkpoint_every == 0:
                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
                print(f"Checkpoint saved: {OUTPUT_CSV}")

    out_df = pd.DataFrame(rows)
    out_df.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(out_df["embedding_status"].value_counts(dropna=False))

    emb_cols = [
        c for c in out_df.columns
        if c.startswith("clip_") and c[5:].isdigit()
    ]

    print(f"Embedding dimensions: {len(emb_cols)}")


if __name__ == "__main__":
    main()