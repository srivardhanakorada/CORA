#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
scale_cs_metrics_per_concept_idx.py

ESR / PSR / HM evaluation where image filenames follow:
    A_photo_of_<Concept_With_Underscores>_<per_concept_index>.png

Key points
- Each concept has images indexed 0..(n_per_concept-1) (default 0..9).
- Per-concept index is computed from CSV order (or you can supply --index-col).
- Overflow handling for index >= n_per_concept: wrap / skip / cap.

CSV expected columns: id, type, text, concept, seed  (id is not used for filenames)
Pools:
  - ESR (targets):     rows with type == "erase"      -> CS(img, "A photo of {concept}.")
  - PSR (non-targets): rows with type == "retention"  -> CS(img, "A photo of {concept}.")
Metrics:
  - ESR := 1 - mean_CS(erase)
  - PSR := mean_CS(retention)
  - HM  := harmonic mean(ESR, PSR) with ESR/PSR clamped to [0,1] ONLY for HM

Usage example:
  python scale_cs_metrics_per_concept_idx.py \
    --csv data/archive/celebrity_five.csv \
    --image-root archive_outputs/archive_two/outputs_cora_multi_full/five/erase \
    --model ViT-L/14 \
    --n-per-concept 10 \
    --index-overflow wrap \
    --out-csv esr_psr_report.csv

Dependencies:
  pip install torch torchvision pillow pandas tqdm git+https://github.com/openai/CLIP.git
"""

import argparse
import os
import re
import sys
from typing import Dict, List, Optional

import torch
import clip  # type: ignore
import pandas as pd
from PIL import Image
from tqdm import tqdm


# --------------------- Utilities ---------------------

def mean(xs: List[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")

def clamp01(x: float) -> float:
    if x != x:  # NaN
        return x
    return 0.0 if x < 0 else 1.0 if x > 1 else x

def harmonic_mean(a: float, b: float) -> float:
    if (a != a) or (b != b):
        return float("nan")
    a, b = clamp01(a), clamp01(b)
    s = a + b
    return (2*a*b/s) if s > 0 else float("nan")

ALNUM_UNDERSCORE = re.compile(r"[^A-Za-z0-9_]+")

def concept_to_slug(concept: str) -> str:
    """
    'Donald Trump' -> 'Donald_Trump'
    'Anne-Hathaway' -> 'AnneHathaway'
    (spaces -> '_', then drop anything not [A-Za-z0-9_])
    """
    slug = str(concept).strip().replace(" ", "_")
    slug = ALNUM_UNDERSCORE.sub("", slug)
    return slug

def build_path(image_root: str, concept: str, per_concept_idx: int) -> str:
    """
    Build path as: A_photo_of_<Concept_With_Underscores>_<idx>.png
    """
    filename = f"A_photo_of_{concept_to_slug(concept)}_{int(per_concept_idx)}.png"
    return os.path.join(image_root, filename)


# --------------------- CLIP helpers ---------------------

def load_clip(model_name: str, device: str):
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()
    return model, preprocess

@torch.no_grad()
def encode_text(model, device: str, prompt: str) -> torch.Tensor:
    toks = clip.tokenize([prompt]).to(device)
    t = model.encode_text(toks).float()
    t /= t.norm(dim=-1, keepdim=True)
    return t.squeeze(0)

@torch.no_grad()
def encode_image(model, preprocess, device: str, path: str) -> Optional[torch.Tensor]:
    try:
        im = Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Could not open image: {path} ({e})", file=sys.stderr)
        return None
    x = preprocess(im).unsqueeze(0).to(device)
    v = model.encode_image(x).float()
    v /= v.norm(dim=-1, keepdim=True)
    return v.squeeze(0)


# --------------------- Main ---------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="CSV with columns: id,type,text,concept,seed")
    ap.add_argument("--image-root", required=True, help="Root folder containing A_photo_of_<concept>_<idx>.png")
    ap.add_argument("--model", default="ViT-L/14", help="CLIP model (e.g., ViT-B/32, ViT-L/14)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--erase-token", default="erase", help="Value in 'type' column for ESR pool")
    ap.add_argument("--retention-token", default="retention", help="Value in 'type' column for PSR pool")

    # Per-concept indexing options
    ap.add_argument("--index-col", default="",
                    help="Existing per-concept index column to use (0-based). If empty, compute via cumcount().")
    ap.add_argument("--sort-by", default="",
                    help="Comma-separated columns to sort within each concept BEFORE cumcount (e.g., seed,id).")
    ap.add_argument("--ascending", default="",
                    help="Comma-separated 1/0 flags aligned with --sort-by (default all ascending).")

    # New: enforce fixed count per concept and handle overflow
    ap.add_argument("--n-per-concept", type=int, default=10,
                    help="Number of files per concept (default 10 for indices 0..9).")
    ap.add_argument("--index-overflow", choices=["wrap", "skip", "cap"], default="wrap",
                    help="How to handle per-concept index >= n-per-concept: wrap (mod n), skip row, or cap to n-1.")

    ap.add_argument("--out-csv", default="", help="Optional path to write per-concept means and overall metrics")
    args = ap.parse_args()

    # Load CSV
    try:
        df = pd.read_csv(args.csv)
    except Exception as e:
        print(f"[ERROR] Failed to read CSV '{args.csv}': {e}", file=sys.stderr)
        sys.exit(2)

    # Basic checks
    for col in ("type", "concept"):
        if col not in df.columns:
            print(f"[ERROR] CSV missing required column '{col}'.", file=sys.stderr)
            sys.exit(2)

    # Normalize key columns
    df["type"] = df["type"].astype("string").str.strip()
    df["_label"] = df["type"].str.lower().fillna("")
    df["concept"] = df["concept"].astype("string").str.strip().fillna("")

    # Determine per-concept index
    if args.index_col:
        if args.index_col not in df.columns:
            print(f"[ERROR] CSV missing --index-col '{args.index_col}'.", file=sys.stderr)
            sys.exit(2)
        df["_idx"] = pd.to_numeric(df[args.index_col], errors="coerce").fillna(-1).astype(int)
        if (df["_idx"] < 0).any():
            bad = df[df["_idx"] < 0]
            print(f"[WARN] {len(bad)} row(s) have invalid {args.index_col}; they will be skipped.", file=sys.stderr)
            df = df[df["_idx"] >= 0].copy()
    else:
        # Optional sorting within each concept before cumcount
        if args.sort_by:
            sort_keys = [s.strip() for s in args.sort_by.split(",") if s.strip()]
            if args.ascending:
                asc_flags = [a.strip() for a in args.ascending.split(",") if a.strip()]
                if len(asc_flags) != len(sort_keys):
                    print("[ERROR] --ascending must have same number of entries as --sort-by.", file=sys.stderr)
                    sys.exit(2)
                ascending = [True if a in ("1", "true", "True") else False for a in asc_flags]
            else:
                ascending = [True] * len(sort_keys)
            df = df.sort_values(by=["concept"] + sort_keys, ascending=[True] + ascending, kind="mergesort")
        else:
            # Stable order within concept as they appear in CSV
            df = df.sort_values(by=["concept"], kind="mergesort")

        # Compute 0-based per-concept index
        df["_idx"] = df.groupby("concept").cumcount()

    # Enforce 0..n_per_concept-1
    n = int(args.n_per_concept)
    if n <= 0:
        print("[ERROR] --n-per-concept must be > 0.", file=sys.stderr)
        sys.exit(2)

    if args.index_overflow == "wrap":
        df["_idx"] = df["_idx"] % n
    elif args.index_overflow == "cap":
        df["_idx"] = df["_idx"].clip(lower=0, upper=n-1)
    else:  # skip
        before = len(df)
        df = df[df["_idx"].between(0, n-1)]
        after = len(df)
        if before != after:
            print(f"[INFO] Skipped {before-after} row(s) due to index overflow beyond 0..{n-1}.", file=sys.stderr)

    # Build file paths with underscore pattern
    df["_path"] = df.apply(lambda r: build_path(args.image_root, r["concept"], int(r["_idx"])), axis=1)

    # Split pools
    erase_tok = str(args.erase_token).lower()
    retention_tok = str(args.retention_token).lower()
    df_erase = df[df["_label"] == erase_tok].copy()
    df_ret   = df[df["_label"] == retention_tok].copy()

    # Load CLIP
    device = args.device
    model, preprocess = load_clip(args.model, device)

    # Cache text embeddings per concept
    concept_cache: Dict[str, torch.Tensor] = {}
    @torch.no_grad()
    def get_text_emb(concept: str) -> torch.Tensor:
        if concept not in concept_cache:
            concept_cache[concept] = encode_text(model, device, f"A photo of {concept}.")
        return concept_cache[concept]

    # Score images
    erase_cs: List[float] = []
    ret_cs: List[float] = []
    per_concept_psr: Dict[str, List[float]] = {}

    # ESR: erase vs own concept
    for _, r in tqdm(df_erase.iterrows(), total=len(df_erase), desc="Scoring ERASE (ESR)"):
        img = encode_image(model, preprocess, device, r["_path"])
        if img is None: continue
        t = get_text_emb(r["concept"])
        erase_cs.append(float((img @ t).item()))

    # PSR: retention vs own concept
    for _, r in tqdm(df_ret.iterrows(), total=len(df_ret), desc="Scoring RETENTION (PSR)"):
        img = encode_image(model, preprocess, device, r["_path"])
        if img is None: continue
        c = r["concept"]
        t = get_text_emb(c)
        cs = float((img @ t).item())
        ret_cs.append(cs)
        per_concept_psr.setdefault(c, []).append(cs)

    # Metrics
    ESR = (1.0 - mean(erase_cs)) if erase_cs else float("nan")
    PSR = mean(ret_cs) if ret_cs else float("nan")
    HM  = harmonic_mean(ESR, PSR)

    # Report
    print("\n=== ESR / PSR with per-concept indices ===")
    print(f"Images root: {args.image_root}")
    print(f"ESR = 1 - mean_CS(erase): {ESR:.4f} over {len(erase_cs)} images")
    print(f"PSR = mean_CS(retention): {PSR:.4f} over {len(ret_cs)} images")
    print(f"HM(ESR, PSR):            {HM:.4f}")

    # Optional CSV out
    if args.out_csv:
        import csv as _csv
        os.makedirs(os.path.dirname(args.out_csv) or ".", exist_ok=True)
        with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
            w = _csv.writer(f)
            w.writerow(["group","concept","num_images","mean_CS","note"])
            w.writerow(["erase_pool","(mixed concepts)",len(erase_cs),mean(erase_cs),"ESR uses 1 - mean_CS(erase)"])
            for c in sorted(per_concept_psr.keys()):
                vals = per_concept_psr[c]
                w.writerow(["retention_pool",c,len(vals),mean(vals),"PSR = mean_CS(retention)"])
            w.writerow([])
            w.writerow(["ESR (1 - mean_CS_erase)","",len(erase_cs),ESR,""])
            w.writerow(["PSR (mean_CS_retention)","",len(ret_cs),PSR,""])
            w.writerow(["HM(ESR,PSR) [clamp in HM only]","","",HM,""])

if __name__ == "__main__":
    main()
