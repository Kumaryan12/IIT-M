from pathlib import Path
import argparse

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


MANIFEST_CSV = Path("outputs/features/processed_frame_manifest.csv")
OUTPUT_CSV = Path("outputs/features/clip_frame_embeddings_v1.csv")

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
    parser.add_argument("--lenses", nargs="+", type=int, default=[1, 4, 6])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint-every", type=int, default=25)
    args = parser.parse_args()

    if not MANIFEST_CSV.exists():
        print(f"Missing manifest: {MANIFEST_CSV}")
        return

    df = pd.read_csv(MANIFEST_CSV)
    df = df[df["preprocess_status"] == "success"].copy()
    df = df[df["lens_id"].isin(args.lenses)].copy()

    if args.limit is not None:
        df = df.head(args.limit).copy()

    print("\nCLIP embedding extraction")
    print(f"Frames selected: {len(df)}")
    print(f"Lenses: {args.lenses}")
    print(f"Model: {MODEL_NAME}")

    device = get_device()
    print(f"Device: {device}")

    model = CLIPModel.from_pretrained(MODEL_NAME).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval()

    existing_rows = []
    done_keys = set()

    if OUTPUT_CSV.exists():
        old = pd.read_csv(OUTPUT_CSV)
        existing_rows = old.to_dict(orient="records")
        if "processed_frame_key" in old.columns:
            done_keys = set(old["processed_frame_key"].astype(str))
        print(f"Loaded existing embeddings: {len(done_keys)}")

    rows = existing_rows
    since_save = 0

    with torch.no_grad():
        for _, row in df.iterrows():
            key = str(row["processed_frame_key"])

            if key in done_keys:
                print(f"Skipping: {key}")
                continue

            image_path = Path(row["processed_frame_path"])
            print(f"Processing: {key}")

            try:
                image = load_image(image_path)

                inputs = processor(
                    images=image,
                    return_tensors="pt",
                )

                inputs = {
                    k: v.to(device)
                    for k, v in inputs.items()
                }

                image_features = model.get_image_features(**inputs)

                image_features = image_features / image_features.norm(
                    dim=-1,
                    keepdim=True,
                )

                emb = image_features[0].detach().cpu().numpy()

                out_row = {
                    "processed_frame_key": key,
                    "source_frame_key": row.get("source_frame_key", ""),
                    "matched_run_id": row.get("matched_run_id", ""),
                    "video_offset_sec": row.get("video_offset_sec", None),
                    "lens_id": int(row["lens_id"]),
                    "processed_frame_path": str(image_path),
                    "clip_status": "success",
                    "clip_error": "",
                }

                for i, val in enumerate(emb):
                    out_row[f"clip_{i}"] = float(val)

            except Exception as e:
                out_row = {
                    "processed_frame_key": key,
                    "source_frame_key": row.get("source_frame_key", ""),
                    "lens_id": row.get("lens_id", None),
                    "processed_frame_path": str(image_path),
                    "clip_status": "failed",
                    "clip_error": str(e),
                }

            rows.append(out_row)
            done_keys.add(key)
            since_save += 1

            if since_save >= args.checkpoint_every:
                pd.DataFrame(rows).to_csv(OUTPUT_CSV, index=False)
                print(f"Checkpoint saved: {OUTPUT_CSV}")
                since_save = 0

    out = pd.DataFrame(rows)
    out.to_csv(OUTPUT_CSV, index=False)

    print("\nDone.")
    print(f"Saved to: {OUTPUT_CSV}")
    print(out["clip_status"].value_counts(dropna=False))

    emb_cols = [c for c in out.columns if c.startswith("clip_") and c[5:].isdigit()]
    print(f"Embedding dimensions: {len(emb_cols)}")


if __name__ == "__main__":
    main()