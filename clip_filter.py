#!/usr/bin/env python3
"""Optimised CLIP train/test splitter — streaming inference + disk-backed images
===============================================================================

*   **Thread-safe**: CLIP model lives only in the main thread; worker threads
    are strictly I/O-bound.
*   **Streaming inference**: images are processed as they become available.
*   **Dynamic image management**: each image is saved to a temporary PNG on disk and
    deleted immediately after processing to minimize disk usage.
*   **Download throttling**: prevents too many images from accumulating by
    pausing downloads when a threshold is reached, adjusted to inference speed.
*   **Optimizations**: parallel downloading, parallel image loading, mixed precision inference.
"""

from __future__ import annotations
import argparse
import os
import shutil
import tempfile
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, List, Tuple, Dict, Optional
import gc

import pandas as pd
import requests
import requests
import torch
from PIL import Image
from PIL import Image
from torch.nn.functional import cosine_similarity
from transformers import CLIPModel, CLIPProcessor
from torch.amp import autocast
import contextlib
from io import BytesIO

class ThrottledDownloader:
    """Manages downloads with throttling to prevent excessive disk usage."""
    
    def __init__(
        self, 
        df: pd.DataFrame, 
        tmp_dir: Path, 
        max_pending: int = 150,
        max_workers: int = None
    ):
        self.df = df
        self.tmp_dir = tmp_dir
        self.max_pending = max_pending
        self.max_workers = max_workers or (os.cpu_count() or 4)
        
        # Synchronization
        self.queue = queue.Queue()
        self.pending_lock = threading.Lock()
        self.pending_cond = threading.Condition(self.pending_lock)
        self.all_scheduled = threading.Event()
        self.download_complete = threading.Event()
        
        # Track counts
        self.pending_count = 0
        self.processed_count = 0
        self.failed_count = 0
        self.completed_workers = 0
        self.download_threads = []
    
    def download_and_save(self, idx: int, row: pd.Series) -> Tuple[Path, str, dict[str, Any]] | None:
        """Download image to `tmp_dir/idx.png`; return (path, prompt, row_dict) or None."""
        url = row["url"]
        img_path = self.tmp_dir / f"{idx}.png"
        try:
            resp = requests.get(url, timeout=2.5)
            resp.raise_for_status()
            img = Image.open(BytesIO(resp.content)).convert("RGB")
            img.save(img_path)
            return img_path, row["prompt"], row.to_dict()
        except Exception as e:
            #print(f"[ERROR] Failed to download {url}: {e}")
            img_path.unlink(missing_ok=True)
            return None
    
    def download_worker(self, start_idx: int, end_idx: int):
        """Thread worker to download images for a chunk of the dataframe."""
        for idx in range(start_idx, end_idx):
            row = self.df.iloc[idx]
            
            with self.pending_cond:
                while self.pending_count >= self.max_pending:
                    self.pending_cond.wait()
                self.pending_count += 1
            
            result = self.download_and_save(idx, row)
            if result:
                self.queue.put(result)
            else:
                with self.pending_lock:
                    self.pending_count -= 1
                    self.failed_count += 1
                    self.pending_cond.notify_all()
        
        with self.pending_lock:
            self.completed_workers += 1
            if self.completed_workers == len(self.download_threads):
                self.all_scheduled.set()
    
    def mark_processed(self):
        """Mark an image as processed, reducing the pending count."""
        with self.pending_lock:
            self.pending_count -= 1
            self.processed_count += 1
            self.pending_cond.notify_all()
    
    def start(self):
        """Start multiple download threads, each handling a chunk of the dataframe."""
        num_rows = len(self.df)
        chunk_size = (num_rows + self.max_workers - 1) // self.max_workers  # Ceiling division
        self.download_threads = []
        for i in range(self.max_workers):
            start_idx = i * chunk_size
            end_idx = min(start_idx + chunk_size, num_rows)
            if start_idx >= end_idx:
                break
            t = threading.Thread(target=self.download_worker, args=(start_idx, end_idx))
            t.daemon = True
            t.start()
            self.download_threads.append(t)
    
    def get_next_batch(self, batch_size: int) -> List[Tuple[Path, str, dict[str, Any]]]:
        """Get the next batch of images, up to batch_size."""
        batch = []
        timeout = 1.0  # Initial timeout
        
        while len(batch) < batch_size:
            try:
                if self.all_scheduled.is_set() and self.queue.empty():
                    if len(batch) > 0:
                        break
                    else:
                        self.download_complete.set()
                        break
                
                item = self.queue.get(timeout=timeout)
                batch.append(item)
                self.queue.task_done()
                timeout = 0.1
                
            except queue.Empty:
                if len(batch) > 0:
                    break
                timeout = min(timeout * 1.5, 5.0)
        
        return batch
    
    def is_complete(self) -> bool:
        """Check if downloading and processing is complete."""
        return self.download_complete.is_set()
    
    def get_stats(self) -> Dict[str, int]:
        """Get download statistics."""
        with self.pending_lock:
            return {
                "processed": self.processed_count,
                "failed": self.failed_count,
                "pending": self.pending_count
            }

def clip_filter(
    input_csv: str,
    output_dir: str = "output",
    train_ratio: float = 0.8,
    clip_model_name: str = "openai/clip-vit-base-patch32",
    clip_model_name: str = "openai/clip-vit-base-patch32",
    num_workers: int | None = None,
    batch_size: int = 64,
    max_pending: int = 150,
):
    print("[INFO] Loading CSV...")
    df = pd.read_csv(input_csv)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Loading CLIP model on {device}...")
    processor = CLIPProcessor.from_pretrained(clip_model_name, use_fast=True)
    model = CLIPModel.from_pretrained(clip_model_name).to(device).eval()

    # Check if processor has necessary attributes
    if not hasattr(processor, 'image_processor') or not hasattr(processor, 'tokenizer'):
        raise AttributeError("The processor does not have 'image_processor' or 'tokenizer' attributes.")

    tmp_dir = Path(tempfile.mkdtemp(prefix="clip_filter_"))
    print(f"[INFO] Temp dir: {tmp_dir}")

    max_workers = num_workers or (os.cpu_count() or 4)
    print(f"[INFO] Download threads: {max_workers}")
    
    results: List[Tuple[float, dict[str, Any]]] = []
    
    # Initialize the downloader with throttling
    downloader = ThrottledDownloader(
        df=df, 
        tmp_dir=tmp_dir, 
        max_pending=max_pending,
        max_workers=max_workers
    )
    
    # Start downloading
    downloader.start()

    def infer_batch(batch_data: List[Tuple[Path, str, dict[str, Any]]]):
        """Process a batch of images with optimized inference."""
        if not batch_data:
            return []
        
        img_paths, prompts, row_dicts = zip(*batch_data)
        
        # Parallel image loading
        with ThreadPoolExecutor(max_workers=8) as executor:
            imgs = list(executor.map(lambda p: Image.open(p).convert("RGB"), img_paths))
        
        try:
            with torch.no_grad():
                with autocast('cuda') if device == "cuda" else contextlib.nullcontext():
                    # Manually process images
                    image_processor = processor.image_processor
                    img_inputs = image_processor(images=imgs, return_tensors="pt").to(device)
                    
                    # Manually process texts
                    tokenizer = processor.tokenizer
                    txt_inputs = tokenizer(
                        text=prompts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=77
                    ).to(device)
                    
                    img_emb = model.get_image_features(**img_inputs)
                    txt_emb = model.get_text_features(**txt_inputs)
                    sims = cosine_similarity(img_emb, txt_emb, dim=1).cpu()
                
            batch_results = [(s.item(), r) for s, r in zip(sims, row_dicts)]
        except Exception as e:
            print(f"[ERROR] Inference failed: {e}")
            batch_results = []
        
        # Clean up
        for p, im in zip(img_paths, imgs):
            im.close()
            p.unlink(missing_ok=True)
            downloader.mark_processed()
        
        return batch_results

    print("[INFO] Starting streaming inference...")
    
    # Main processing loop with periodic memory cleanup
    batch_counter = 0
    while not downloader.is_complete():
        batch_data = downloader.get_next_batch(batch_size)
        if batch_data:
            batch_results = infer_batch(batch_data)
            results.extend(batch_results)
            
            batch_counter += 1
            if batch_counter % 10 == 0 and device == "cuda":
                torch.cuda.empty_cache()
            
            stats = downloader.get_stats()
            print(f"[INFO] Processed: {stats['processed']}, Failed: {stats['failed']}, Pending: {stats['pending']}")
            
            if device != "cuda":
                gc.collect()

    # Clean up temp directory
    print("[INFO] Cleaning up temporary files...")
    shutil.rmtree(tmp_dir, ignore_errors=True)

    if not results:
        print("[WARN] No valid images processed.")
        return

    # Process results
    print(f"[INFO] Processing {len(results)} results...")
    sims_arr, rows_arr = zip(*results)
    result_df = pd.DataFrame(rows_arr)
    result_df["similarity"] = sims_arr
    result_df = result_df.sort_values("similarity", ascending=False).reset_index(drop=True)

    # Split into train and test sets
    split_idx = int(len(result_df) * train_ratio)
    train_df, test_df = result_df.iloc[:split_idx], result_df.iloc[split_idx:]

    # Write to output files
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for name, split_df in (("train", train_df), ("test", test_df)):
        folder = output_path / name
        folder.mkdir(parents=True, exist_ok=True)
        split_df.to_csv(folder / f"{name}.csv", index=False)
        split_df.to_csv(folder / f"{name}.csv", index=False)

    print(f"[DONE] Train({len(train_df)}) & Test({len(test_df)}) written to '{output_dir}'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", required=True)
    parser.add_argument("--output_dir", default="output")
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument(
        "--clip_model",
        dest="clip_model_name",
        default="openai/clip-vit-base-patch32"
    )
    parser.add_argument("--num_workers", type=int)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_pending", type=int, default=150,
                        help="Maximum number of pending downloads before throttling")
    args = parser.parse_args()

    clip_filter(
        input_csv=args.input_csv,
        output_dir=args.output_dir,
        train_ratio=args.train_ratio,
        clip_model_name=args.clip_model_name,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        max_pending=args.max_pending,
    )