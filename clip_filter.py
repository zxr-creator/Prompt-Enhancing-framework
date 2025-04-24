import os
import csv
import shutil
import tempfile
import requests
from pathlib import Path
from PIL import Image
from io import BytesIO
import pandas as pd
import torch
from transformers import CLIPProcessor, CLIPModel
from sklearn.model_selection import train_test_split
from torch.nn.functional import cosine_similarity
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
import gc  # Add this to your imports

def clip_filter(
    input_csv: str,
    output_dir: str = "output",
    train_ratio: float = 0.8,
    clip_model: str = "openai/clip-vit-base-patch32",
    num_workers: int | None = None,
):
    """
    Splits a dataset into train/test by CLIP similarity between image and prompt.
    * Drops rows whose prompt token count > 77 (CLIP's max context length).
    * Downloads images to a temporary folder (auto-cleaned), processes in parallel,
      and outputs train/test CSVs with an added 'similarity' column.
    * Logs progress every 1000 images.
    """
    # 1. Load data -------------------------------------------------------------
    print(f"[INFO] Start loading data")
    df = pd.read_csv(input_csv)

    # 2. Prepare CLIP ----------------------------------------------------------
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = CLIPProcessor.from_pretrained(clip_model, use_fast=False)
    model = CLIPModel.from_pretrained(clip_model).to(device).eval()
    tokenizer = processor.tokenizer

    # 3. Filter prompts longer than 77 tokens ---------------------------------
    print(f"[INFO] Start filtering prompts longer than 77 tokens")
    df["token_len"] = df["prompt"].astype(str).apply(
        lambda p: len(tokenizer(p)["input_ids"])
    )
    orig_rows = len(df)
    df = df[df["token_len"] <= 77].reset_index(drop=True).drop(columns="token_len")
    skipped = orig_rows - len(df)
    if skipped:
        print(f"[INFO] Skipped {skipped} rows (prompt length > 77 tokens)")

    if df.empty:
        print("[WARN] All rows were filtered out — nothing to process.")
        return

    # 4. Temp folder for images -----------------------------------------------
    print(f"[INFO] Start filtering prompts longer than 77 tokens")
    temp_dir = tempfile.mkdtemp(prefix="clip_filter_")

    # 5. Progress tracking -----------------------------------------------------
    processed_count = 0
    count_lock = threading.Lock()

    # 6. Helper to process one row --------------------------------------------
    def process_row(idx, row):
    nonlocal processed_count
    url = row["url"]
    try:
        # Download image
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img_path = Path(temp_dir) / f"{idx}.png"
        img.save(img_path)

        # CLIP encode
        inputs = processor(
            text=row["prompt"], images=img, return_tensors="pt", padding=True
        ).to(device)

        with torch.no_grad():
            img_emb = model.get_image_features(**inputs)
            txt_emb = model.get_text_features(**inputs)
            sim = cosine_similarity(img_emb, txt_emb).item()

        # Clean up image file
        img_path.unlink(missing_ok=True)

        # Explicitly delete to reduce memory footprint
        del img, inputs, img_emb, txt_emb
        gc.collect()
        torch.cuda.empty_cache()

        # Thread-safe log
        with count_lock:
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"[INFO] Processed {processed_count} images")

        return sim, row.to_dict()
    except Exception:
        return None

    # 7. Parallel processing ---------------------------------------------------
    print(f"[INFO] Start Parallel processing")
    results = []
    max_workers = num_workers or (os.cpu_count() or 4)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(process_row, idx, row): idx
            for idx, row in df.iterrows()
        }
        for future in as_completed(futures):
            res = future.result()
            if res is not None:
                results.append(res)

    # 8. Cleanup temp folder ---------------------------------------------------
    print(f"[INFO] Start Cleanup temp folder")
    shutil.rmtree(temp_dir, ignore_errors=True)

    # 9. Build DataFrame -------------------------------------------------------
    print(f"[INFO] Start Building DataFrame")
    if not results:
        print("[WARN] No valid images processed.")
        return

    sims, rows = zip(*results)
    result_df = pd.DataFrame(rows)
    result_df["similarity"] = sims

    # 10. Sort and split -------------------------------------------------------
    print(f"[INFO] Start sorting and spliting")
    result_df = result_df.sort_values("similarity", ascending=False).reset_index(
        drop=True
    )
    split_idx = int(len(result_df) * train_ratio)
    train_df = result_df.iloc[:split_idx]
    test_df = result_df.iloc[split_idx:]

    # 11. Write CSVs -----------------------------------------------------------
    print(f"[INFO] Start writing CSVs")
    for name, split_df in [("train", train_df), ("test", test_df)]:
        folder = Path(output_dir) / name
        folder.mkdir(parents=True, exist_ok=True)
        csv_path = folder / f"{name}.csv"
        split_df.to_csv(csv_path, index=False)

    print(
        f"Train ({len(train_df)}) and Test ({len(test_df)}) CSVs written under '{output_dir}'."
    )


# Example call
clip_filter(
    "./datasets/900k-diffusion-prompts-dataset/diffusion_prompts.csv",
    output_dir="./datasets/900k-diffusion-prompts-dataset/finetune",
    train_ratio=0.8,
    num_workers=64,
)
