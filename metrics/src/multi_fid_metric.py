#!/usr/bin/env python3
"""
Compute FID_org for ONLY non-target categories using CleanFID,
matching concepts by the SAME strict filename scheme as ESR/PSR/HM.

Accepted filename stems (case-insensitive):
    "A photo of {concept}_{idx}"
    "A_photo_of_{concept}_{idx}"
where {idx} ∈ [0, 999], and {concept} may use spaces or underscores.

We take:
    --gen_folder : generated images tree
    --orig_folder: original images tree
    --non_targets: list of concept names to KEEP (normalize spaces/underscores; case-insensitive)

We:
  1) Parse every image filename in each tree using the strict stem parser above.
  2) Keep only images whose parsed concept ∈ non_targets (after normalization).
  3) Compute FID_org (CleanFID) between the two filtered subsets.

Usage:
    pip install clean-fid

Example:
    python fid_metric.py \
      --gen_folder results/cora/multi/one/erase/ \
      --orig_folder results/cora/multi/one/original/ \
      --non_targets "Adriana_Lima,Amber_Heard,Amy_Adams,Andrew_Garfield,Angeline_Jolie,Anjelica_Huston,Anna_Faris,Anna_Kendrick,Anne_Hathaway,Chris_Evans,Dwayne_Johnson,Elon_Musk,Emma_Stone,Hugh_Jackman,Melania_Trump,Tom_Cruise,Ryan_Reynolds,Tom_Hiddleston,Nicolas_Cage" \
      --out_json results/cora/multi/one/erase_fid_org_nontargets.json
"""

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from typing import List, Optional, Tuple, Set

from cleanfid import fid  # type: ignore

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}

# ---------- normalization & parsing (identical behavior to ESR script) ----------

def concept_norm(s: str) -> str:
    """underscores->spaces, trim, collapse spaces, lowercase"""
    return " ".join(s.replace("_", " ").strip().split()).lower()

# two accepted stem patterns:
STEM_SPACE = re.compile(r"^a photo of\s+(.+?)_([0-9]{1,3})$", re.IGNORECASE)
STEM_UNDER = re.compile(r"^a_photo_of_(.+?)_([0-9]{1,3})$", re.IGNORECASE)

def parse_stem(stem: str) -> Optional[Tuple[str, int]]:
    """
    Return (normalized_concept, idx) if stem matches either accepted pattern and 0 <= idx <= 999.
    Otherwise None.
    """
    m = STEM_SPACE.match(stem) or STEM_UNDER.match(stem)
    if not m:
        return None
    concept_raw, idx_s = m.group(1), m.group(2)
    try:
        idx = int(idx_s)
    except ValueError:
        return None
    if idx < 0 or idx > 999:
        return None
    c = concept_norm(concept_raw)
    return (c, idx) if c else None

# ---------- IO ----------

def read_concepts(spec: Optional[str]) -> List[str]:
    """Read JSON/TXT/CSV string; return normalized concepts."""
    if not spec:
        return []
    p = spec.strip()
    if not p:
        return []
    if os.path.isfile(p):
        if p.lower().endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                arr = json.load(f)
            items = [str(x) for x in arr]
        else:
            with open(p, "r", encoding="utf-8") as f:
                items = [line.strip() for line in f if line.strip()]
    else:
        items = [x.strip() for x in p.split(",") if x.strip()]
    return [concept_norm(x) for x in items]

def list_images(folder: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, name))
    return paths

# ---------- filtering by parsed concept ----------

def filter_folder_by_concept_stems(src_folder: str, keep_concepts: Set[str], dst_folder: str) -> int:
    """
    Copy all images from src_folder whose filename stem parses successfully and
    whose parsed concept ∈ keep_concepts (normalized). Returns count copied.
    """
    os.makedirs(dst_folder, exist_ok=True)
    count = 0
    for src in list_images(src_folder):
        stem = os.path.splitext(os.path.basename(src))[0]
        parsed = parse_stem(stem)
        if parsed is None:
            continue
        concept, _ = parsed
        if concept not in keep_concepts:
            continue
        # Copy flat; disambiguate on collisions
        dst = os.path.join(dst_folder, os.path.basename(src))
        if os.path.exists(dst):
            base, ext = os.path.splitext(os.path.basename(src))
            i = 1
            while True:
                cand = os.path.join(dst_folder, f"{base}_{i}{ext}")
                if not os.path.exists(cand):
                    dst = cand
                    break
                i += 1
        shutil.copy2(src, dst)
        count += 1
    return count

# ---------- main ----------

def main():
    ap = argparse.ArgumentParser(description="Compute FID_org for non-targets using strict filename parsing.")
    ap.add_argument("--gen_folder", required=True, help="Folder of generated images")
    ap.add_argument("--orig_folder", required=True, help="Folder of original images")
    ap.add_argument("--non_targets", required=True,
                    help="Non-target concepts (CSV, TXT path, or JSON path). Names may use spaces or underscores.")
    ap.add_argument("--mode", default="clean",
                    choices=["clean", "legacy_pytorch", "legacy_tensorflow"],
                    help="CleanFID mode (default: clean)")
    ap.add_argument("--out_json", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    # sanity
    if not os.path.isdir(args.gen_folder):
        print(f"[ERROR] --gen_folder not found: {args.gen_folder}", file=sys.stderr); sys.exit(2)
    if not os.path.isdir(args.orig_folder):
        print(f"[ERROR] --orig_folder not found: {args.orig_folder}", file=sys.stderr); sys.exit(2)

    non_targets_norm = read_concepts(args.non_targets)
    if not non_targets_norm:
        print("[ERROR] --non_targets is empty after parsing.", file=sys.stderr); sys.exit(2)
    keep_set = set(non_targets_norm)

    # temp dirs
    tmp_root = tempfile.mkdtemp(prefix="fid_nontargets_strict_")
    gen_filtered = os.path.join(tmp_root, "gen_filtered")
    orig_filtered = os.path.join(tmp_root, "orig_filtered")

    try:
        print(f"[Info] Non-target concepts (normalized): {sorted(keep_set)}")

        print("[Info] Filtering generated images by strict stems...")
        n_gen = filter_folder_by_concept_stems(args.gen_folder, keep_set, gen_filtered)
        print(f"[Info] Kept {n_gen} generated images.")

        print("[Info] Filtering original images by strict stems...")
        n_orig = filter_folder_by_concept_stems(args.orig_folder, keep_set, orig_filtered)
        print(f"[Info] Kept {n_orig} original images.")

        # helpful debug if nothing matched
        if n_gen == 0 or n_orig == 0:
            # show some stems we saw to help fix formatting
            sample_gen = [os.path.splitext(os.path.basename(p))[0] for p in list_images(args.gen_folder)[:10]]
            sample_orig = [os.path.splitext(os.path.basename(p))[0] for p in list_images(args.orig_folder)[:10]]
            print("[WARN] One filtered set is empty; FID is undefined.", file=sys.stderr)
            print(f"[Debug] Sample gen stems:  {sample_gen}", file=sys.stderr)
            print(f"[Debug] Sample orig stems: {sample_orig}", file=sys.stderr)
            result = {"FID_org_non_targets": float("nan"), "num_gen": n_gen, "num_orig": n_orig}
            if args.out_json:
                os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                print(f"[Info] Wrote JSON: {args.out_json}")
            return

        print("[Info] Computing FID_org (CleanFID) on filtered non-target subsets...")
        fid_val = float(fid.compute_fid(gen_filtered, orig_filtered, mode=args.mode))
        print(f"[Result] FID_org (non-targets, strict stems): {fid_val:.4f}")

        if args.out_json:
            os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
            with open(args.out_json, "w", encoding="utf-8") as f:
                json.dump({
                    "FID_org_non_targets": fid_val,
                    "num_gen": n_gen,
                    "num_orig": n_orig
                }, f, indent=2)
            print(f"[Info] Wrote JSON: {args.out_json}")

    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

if __name__ == "__main__":
    main()
