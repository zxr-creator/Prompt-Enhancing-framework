"""Optimised CLIP train/test splitter — batch inference + disk‑backed images
============================================================================

*   **Thread‑safe**: CLIP model lives only in the main thread; worker threads
    are strictly I/O‑bound.
*   **Batch inference**: images & prompts are processed in configurable batches
    on the GPU.
*   **Disk‑backed images**: each image is saved to a temporary PNG on disk to
    keep peak RAM usage minimal. After its batch is processed the file is
    deleted immediately.  
    (If you later want full in‑memory mode, swap `download_and_save()` to return
    PIL images instead of paths.)

Usage example
-------------
```bash
python clip_filter_batch_io.py \
    --input_csv data.csv \
    --output_dir out \
    --batch_size 128 \
    --num_workers 32
```
"""

from __future__ import annotations
import argparse
import os
import shutil
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd
import requests
import torch
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import CLIPModel, CLIPProcessor

# -----------------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------------

def download_and_save(idx: int, row: pd.Series, tmp_dir: Path) -> Tuple[Path, str, dict[str, Any]] | None:  # noqa: E501
    """Download image to `tmp_dir/idx.png`; return (path, prompt, row_dict) or None."""
    url = row["url"]
    img_path = tmp_dir / f"{idx}.png"
    try:
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        img.save(img_path)
        return img_path, row["prompt"], row.to_dict()
    except Exception:
        # on failure ensure no half file remains
        try:
            img_path.unlink(missing_ok=True)
        except Exception:
            pass
        return None


@lru_cache(maxsize=20_000)
def truncate_prompt(prompt: str, tokenizer, eos_id: int) -> str:
    ids = tokenizer(prompt)["input_ids"]
    if len(ids) <= 77:
        return prompt
    ids = ids[:77]
    if ids[-1] != eos_id:
        ids[-1] = eos_id
    return tokenizer.decode(ids, skip_special_tokens=True).strip()


# -----------------------------------------------------------------------------
# Core routine
# -----------------------------------------------------------------------------

def clip_filter(
    input_csv: str,
    output_dir: str = "output",
    train_ratio: float = 0.8,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    num_workers: int | None = None,
    batch_size: int = 64,
):
    print("[INFO] Loading CSV …")
    df = pd.read_csv(input_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor: CLIPProcessor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=False)
    model: CLIPModel = CLIPModel.from_pretrained(clip_model_name).to(device).eval()
    tokenizer = processor.tokenizer
    eos_id = tokenizer.eos_token_id

    print("[INFO] Truncating prompts >77 tokens …")
    df["prompt"] = df["prompt"].astype(str).apply(lambda p: truncate_prompt(p, tokenizer, eos_id))

    # ── Temporary dir for images ────────────────────────────────────────────
    tmp_dir = Path(tempfile.mkdtemp(prefix="clip_filter_"))
    print(f"[INFO] Temp dir: {tmp_dir}")

    # Thread pool for downloads
    max_workers = num_workers or (os.cpu_count() or 4)
    print(f"[INFO] Download threads: {max_workers}")

    processed = 0
    processed_lock = threading.Lock()
    results: List[Tuple[float, dict[str, Any]]] = []

    def infer_batch(img_paths: List[Path], prompts: List[str]):
        # open images lazily
        imgs = [Image.open(p).convert("RGB") for p in img_paths]
        with torch.no_grad():
            img_inputs = processor(images=imgs, return_tensors="pt").to(device)
            txt_inputs = processor(text=prompts, return_tensors="pt", padding=True).to(device)
            img_emb = model.get_image_features(**img_inputs)
            txt_emb = model.get_text_features(**txt_inputs)
            sims = cosine_similarity(img_emb, txt_emb, dim=1).cpu()
        # close & delete images to free RAM/disk
        for p, im in zip(img_paths, imgs):
            im.close()
            try:
                p.unlink()
            except Exception:
                pass
        return sims

    # batching queues
    batch_paths: List[Path] = []
    batch_prompts: List[str] = []
    batch_rows: List[dict[str, Any]] = []

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(download_and_save, idx, row, tmp_dir): idx for idx, row in df.iterrows()}

        for fut in as_completed(futures):
            sample = fut.result()
            if sample is None:
                continue
            path, prompt, row_dict = sample
            batch_paths.append(path)
            batch_prompts.append(prompt)
            batch_rows.append(row_dict)

            if len(batch_paths) >= batch_size:
                sims = infer_batch(batch_paths, batch_prompts)
                results.extend([(s.item(), r) for s, r in zip(sims, batch_rows)])
                batch_paths.clear(); batch_prompts.clear(); batch_rows.clear()

                with processed_lock:
                    processed += batch_size
                    if processed % 1000 == 0:
                        print(f"[INFO] Processed {processed} samples …")

    # flush remainder
    if batch_paths:
        sims = infer_batch(batch_paths, batch_prompts)
        results.extend([(s.item(), r) for s, r in zip(sims, batch_rows)])

    # clean temp dir
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not results:
        print("[WARN] No valid images processed.")
        return

    sims_arr, rows_arr = zip(*results)
    result_df = pd.DataFrame(rows_arr)
    result_df["similarity"] = sims_arr
    result_df = result_df.sort_values("similarity", ascending=False).reset_index(drop=True)

    split_idx = int(len(result_df) * train_ratio)
    train_df = result_df.iloc[:split_idx]
    test_df = result_df.iloc[split_idx:]

    for name, split_df in (("train", train_df), ("test", test_df)):
        folder = Path(output_dir) / name
        folder.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(folder / f"{name}.csv", index=False)

    print(f"[DONE] Train({len(train_df)}) & Test({len(test_df)}) written to '{output_dir}'")


# -----------------------------------------------------------------------------
# CLI entry
# -----------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    from io import BytesIO

    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--clip_model", default="openai/clip-vit-base-patch32")
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    clip_filter(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        clip_model_name=args.clip_model,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
    )
