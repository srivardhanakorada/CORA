#!/usr/bin/env python3
"""
Compute strict and lenient (0.95×) failure rates for GenBench-40, per target and per category.
Handles normalization mismatches like underscores, hyphens, periods, and trailing underscores.
Author: Vardhan (CORA project)
"""

import os
import re
import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-L/14", device=DEVICE)

COMBINED_DIR = "results/adavd/genbench/combine"
ORIG_DIR = "results/adavd/genbench/original"
CSV_PATH = "gen_bench_40/gen_bench_40.csv"   # Adjust if needed

# ----------------------------
# HELPERS
# ----------------------------
def normalize_name(s: str) -> str:
    """Normalize names to a comparable form (underscores, hyphens, dots, spacing)."""
    s = s.lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def get_target(fname: str) -> str:
    """Extract target name from filename (drop seed number)."""
    return "_".join(fname.split("_")[:-1])

@torch.no_grad()
def compute_clip_sim(img_path: str, text: str) -> float:
    """Compute image-text similarity using CLIP."""
    img = PREPROCESS(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    txt = clip.tokenize([text]).to(DEVICE)
    img_feat = MODEL.encode_image(img)
    txt_feat = MODEL.encode_text(txt)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    return (img_feat @ txt_feat.T).item()

def compute_thresholds(orig_dir: str):
    """Compute mean CLIP similarity per target from original (unaltered) images."""
    thresholds, accum = {}, {}
    for f in tqdm(os.listdir(orig_dir), desc="Computing thresholds"):
        if not f.lower().endswith(".png"):
            continue
        t = get_target(f)
        path = os.path.join(orig_dir, f)
        sim = compute_clip_sim(path, t.replace("_", " "))
        accum.setdefault(t, []).append(sim)
    for t, sims in accum.items():
        thresholds[t] = sum(sims) / len(sims)
    return thresholds

def compute_failures(combined_dir: str, thresholds: dict, meta_df: pd.DataFrame):
    """Compute strict and lenient failures for each image and aggregate per target."""
    results = []

    # normalize metadata once
    meta_df["target_norm"] = meta_df["target_name"].apply(normalize_name)
    meta_lookup = dict(zip(meta_df["target_norm"], meta_df["category"]))

    unmatched = set()

    for f in tqdm(os.listdir(combined_dir), desc="Evaluating failures"):
        if not f.lower().endswith(".png"):
            continue

        t = get_target(f)
        norm_t = normalize_name(t)
        category = meta_lookup.get(norm_t, "Unknown")
        if category == "Unknown":
            unmatched.add(norm_t)

        path = os.path.join(combined_dir, f)
        sim = compute_clip_sim(path, t.replace("_", " "))
        th = thresholds.get(t)
        if th is None:
            continue

        fail_strict = sim > th
        fail_lenient = sim > 0.95 * th

        results.append({
            "target": t,
            "category": category,
            "similarity": sim,
            "threshold": th,
            "fail_strict": int(fail_strict),
            "fail_lenient": int(fail_lenient),
        })

    if unmatched:
        print(f"\n[INFO] Unmatched targets (check CSV naming): {sorted(unmatched)}")

    return pd.DataFrame(results)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print("=== GenBench-40 Dual Failure Evaluation ===")

    meta = pd.read_csv(CSV_PATH)
    thresholds = compute_thresholds(ORIG_DIR)
    df = compute_failures(COMBINED_DIR, thresholds, meta)

    # Aggregate per target
    per_target = (
        df.groupby("target")
        .agg({
            "fail_strict": "mean",
            "fail_lenient": "mean",
            "category": "first",
            "threshold": "first",
        })
        .reset_index()
        .sort_values("fail_strict", ascending=False)
    )

    # Category means
    cat_mean = (
        per_target.groupby("category")[["fail_strict", "fail_lenient"]]
        .mean()
        .reset_index()
    )

    print("\n--- Per Target ---")
    print(per_target.to_string(index=False, formatters={
        "fail_strict": "{:.3f}".format,
        "fail_lenient": "{:.3f}".format,
        "threshold": "{:.3f}".format,
    }))

    print("\n--- Category Means ---")
    print(cat_mean.to_string(index=False, formatters={
        "fail_strict": "{:.3f}".format,
        "fail_lenient": "{:.3f}".format,
    }))

    # Save outputs
    per_target.to_csv("genbench_per_target_failure.csv", index=False)
    cat_mean.to_csv("genbench_category_mean_failure.csv", index=False)

    print("\nSaved results:\n - genbench_per_target_failure.csv\n - genbench_category_mean_failure.csv")
