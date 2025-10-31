#!/usr/bin/env python3
"""
Compute strict and lenient failure rates for GenBench-40 (AdaVD-style filenames).
Supports fuzzy matching between descriptive filename prompts and CSV prompts.
Author: Vardhan (CORA project)
"""

import os
import re
import torch
import clip
import pandas as pd
from PIL import Image
from tqdm import tqdm
from difflib import SequenceMatcher

# ----------------------------
# CONFIG
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL, PREPROCESS = clip.load("ViT-L/14", device=DEVICE)

COMBINED_DIR = "results/sld/genbench/combined"
ORIG_DIR = "results/sld/genbench/original"
CSV_PATH = "gen_bench_40/gen_bench_40.csv"   # your benchmark CSV

# ----------------------------
# HELPERS
# ----------------------------
def normalize_text(s: str) -> str:
    """Normalize text for consistent matching (lowercase, strip, unify spacing)."""
    s = s.lower()
    s = s.replace("_", " ")
    s = s.replace("-", " ")
    s = s.replace(".", "")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def extract_prompt_from_filename(fname: str) -> str:
    """Extract the textual prompt from filenames like 'prompt_text_0.png'."""
    base = os.path.splitext(fname)[0]
    if "_" in base:
        base = "_".join(base.split("_")[:-1])  # drop numeric seed
    return base

@torch.no_grad()
def compute_clip_sim(img_path: str, text: str) -> float:
    """Compute CLIP image-text similarity."""
    img = PREPROCESS(Image.open(img_path).convert("RGB")).unsqueeze(0).to(DEVICE)
    txt = clip.tokenize([text]).to(DEVICE)
    img_feat = MODEL.encode_image(img)
    txt_feat = MODEL.encode_text(txt)
    img_feat /= img_feat.norm(dim=-1, keepdim=True)
    txt_feat /= txt_feat.norm(dim=-1, keepdim=True)
    return (img_feat @ txt_feat.T).item()

def compute_thresholds(orig_dir: str, csv_df: pd.DataFrame):
    """Compute per-target thresholds from original (unaltered) generations."""
    thresholds, accum = {}, {}

    csv_df["prompt_norm"] = csv_df["prompt"].apply(normalize_text)
    prompt_to_target = dict(zip(csv_df["prompt_norm"], csv_df["target_name"]))

    for f in tqdm(os.listdir(orig_dir), desc="Computing thresholds"):
        if not f.lower().endswith(".png"):
            continue
        prompt = extract_prompt_from_filename(f)
        norm_prompt = normalize_text(prompt)
        target = prompt_to_target.get(norm_prompt)
        if target is None:
            # fuzzy match fallback
            best_match = None
            best_ratio = 0
            for p_norm, tname in prompt_to_target.items():
                ratio = SequenceMatcher(None, norm_prompt, p_norm).ratio()
                if ratio > best_ratio:
                    best_ratio, best_match = ratio, tname
            if best_ratio > 0.85:
                target = best_match
            else:
                continue
        path = os.path.join(orig_dir, f)
        sim = compute_clip_sim(path, target)
        accum.setdefault(target, []).append(sim)

    for t, sims in accum.items():
        thresholds[t] = sum(sims) / len(sims)
    return thresholds

def compute_failures(combined_dir: str, thresholds: dict, csv_df: pd.DataFrame):
    """Compute failure stats using fuzzy prompt-to-CSV matching."""
    results = []

    csv_df["prompt_norm"] = csv_df["prompt"].apply(normalize_text)
    meta_records = csv_df[["prompt_norm", "target_name", "category"]].to_dict("records")

    def best_match(prompt_norm):
        best_ratio, best_row = 0, None
        for rec in meta_records:
            ratio = SequenceMatcher(None, prompt_norm, rec["prompt_norm"]).ratio()
            if ratio > best_ratio:
                best_ratio, best_row = ratio, rec
        if best_ratio > 0.85:
            return best_row
        return None

    unmatched = set()

    for f in tqdm(os.listdir(combined_dir), desc="Evaluating failures"):
        if not f.lower().endswith(".png"):
            continue

        prompt = extract_prompt_from_filename(f)
        norm_prompt = normalize_text(prompt)

        meta = best_match(norm_prompt)
        if meta is None:
            unmatched.add(norm_prompt)
            continue

        target, category = meta["target_name"], meta["category"]
        th = thresholds.get(target)
        if th is None:
            continue

        img_path = os.path.join(combined_dir, f)
        sim = compute_clip_sim(img_path, target)

        results.append({
            "prompt": prompt,
            "target": target,
            "category": category,
            "similarity": sim,
            "threshold": th,
            "fail_strict": int(sim > th),
            "fail_lenient": int(sim > 0.95 * th),
        })

    if unmatched:
        print(f"\n[INFO] Unmatched prompts: {len(unmatched)} / {len(os.listdir(combined_dir))}")
        print("Example unmatched prompt:", next(iter(unmatched)))

    if not results:
        print("[ERROR] No matches found — check if your CSV 'prompt' field aligns with filenames.")
    return pd.DataFrame(results)

# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    print("=== GenBench-40 Failure Evaluation (AdaVD-style filenames) ===")

    csv_df = pd.read_csv(CSV_PATH)
    thresholds = compute_thresholds(ORIG_DIR, csv_df)
    df = compute_failures(COMBINED_DIR, thresholds, csv_df)

    if df.empty:
        print("[ERROR] Evaluation failed — no matched results found.")
        exit()

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

    per_target.to_csv("adavd_per_target_failure.csv", index=False)
    cat_mean.to_csv("adavd_category_mean_failure.csv", index=False)

    print("\nSaved results:\n - adavd_per_target_failure.csv\n - adavd_category_mean_failure.csv")
