#!/usr/bin/env python3
"""
Compute FID_org for ONLY non-target categories using CleanFID.

We take two folders:
    --gen_folder   : generated images
    --orig_folder  : original images

We also take a list of non-target concepts. We FILTER both folders, keeping only
files whose *filename without extension* (case-insensitively) contains the
substring "a photo of {concept}" for ANY concept in the non-target list.
Then we compute FID between these two filtered subsets.

Conventions:
- Matching is case-insensitive, but space-sensitive (literal spaces), e.g.,
  filename "...A photo of Lemon..." matches the concept "lemon".
- Image extensions recognized: .png, .jpg, .jpeg, .webp, .bmp, .tiff, .gif
- CleanFID defaults: mode="clean", Inception-V3.

Usage:
    pip install clean-fid

    python fid_org_nontargets.py \
      --gen_folder results/cora/trump/erase \
      --orig_folder results/cora/trump/original \
      --non_targets "Lemon, Dog" \
      --out_json results/cora/trump/erase_fid_org_nontargets.json
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
from typing import List, Optional

from cleanfid import fid  # type: ignore

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}


def read_concepts(spec: Optional[str]) -> List[str]:
    """
    Read concepts from JSON (list of str), TXT (one per line), or comma-separated string.
    Returns lowercased, stripped items.
    """
    if not spec:
        return []
    p = spec.strip()
    if not p:
        return []
    if os.path.isfile(p):
        if p.lower().endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                arr = json.load(f)
            return [str(x).strip().lower() for x in arr if str(x).strip()]
        else:
            with open(p, "r", encoding="utf-8") as f:
                arr = [line.strip() for line in f if line.strip()]
            return [x.lower() for x in arr]
    # inline CSV
    return [x.strip().lower() for x in p.split(",") if x.strip()]


def list_images(folder: str) -> List[str]:
    paths = []
    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                paths.append(os.path.join(root, name))
    return paths


def belongs_to_any_nontarget(filename_noext_lower: str, nontargets_lower: List[str]) -> bool:
    """
    Return True iff the filename (sans extension, lowercased) contains the substring
    'a photo of {concept}' for ANY concept in nontargets_lower.
    """
    for c in nontargets_lower:
        if f"a photo of {c}" in filename_noext_lower:
            return True
    return False


def filter_folder(src_folder: str, nontargets_lower: List[str], dst_folder: str) -> int:
    """
    Copy (preserve filenames) all images from src_folder that match ANY non-target concept
    into dst_folder. Returns the number of files copied.
    """
    os.makedirs(dst_folder, exist_ok=True)
    count = 0
    for p in list_images(src_folder):
        noext = os.path.splitext(os.path.basename(p))[0]
        fn_lower = noext.lower()
        if belongs_to_any_nontarget(fn_lower, nontargets_lower):
            # Copy flat (no subdirs). If name collisions occur, append an index.
            dst_path = os.path.join(dst_folder, os.path.basename(p))
            if os.path.exists(dst_path):
                stem, ext = os.path.splitext(os.path.basename(p))
                i = 1
                while True:
                    candidate = os.path.join(dst_folder, f"{stem}_{i}{ext}")
                    if not os.path.exists(candidate):
                        dst_path = candidate
                        break
                    i += 1
            shutil.copy2(p, dst_path)
            count += 1
    return count


def main():
    ap = argparse.ArgumentParser(description="Compute FID_org restricted to non-target categories.")
    ap.add_argument("--gen_folder", required=True, help="Folder of generated images")
    ap.add_argument("--orig_folder", required=True, help="Folder of original images")
    ap.add_argument("--non_targets", required=True,
                    help="Non-target concepts (CSV, TXT, or JSON list).")
    ap.add_argument("--mode", default="clean",
                    choices=["clean", "legacy_pytorch", "legacy_tensorflow"],
                    help="CleanFID mode (default: clean)")
    ap.add_argument("--out_json", default=None, help="Optional JSON output path")
    args = ap.parse_args()

    # Basic checks
    if not os.path.isdir(args.gen_folder):
        print(f"[ERROR] --gen_folder not found: {args.gen_folder}", file=sys.stderr)
        sys.exit(2)
    if not os.path.isdir(args.orig_folder):
        print(f"[ERROR] --orig_folder not found: {args.orig_folder}", file=sys.stderr)
        sys.exit(2)

    non_targets = read_concepts(args.non_targets)
    if not non_targets:
        print("[ERROR] --non_targets is empty after parsing.", file=sys.stderr)
        sys.exit(2)

    # Create temporary filtered folders
    tmp_root = tempfile.mkdtemp(prefix="fid_nontargets_")
    gen_filtered = os.path.join(tmp_root, "gen_filtered")
    orig_filtered = os.path.join(tmp_root, "orig_filtered")

    try:
        n_lower = non_targets  # already lowercased by read_concepts

        print(f"[Info] Filtering generated folder for non-targets: {non_targets}")
        n_gen = filter_folder(args.gen_folder, n_lower, gen_filtered)
        print(f"[Info] Kept {n_gen} generated images for non-target evaluation.")

        print(f"[Info] Filtering original folder for non-targets: {non_targets}")
        n_orig = filter_folder(args.orig_folder, n_lower, orig_filtered)
        print(f"[Info] Kept {n_orig} original images for non-target evaluation.")

        if n_gen == 0 or n_orig == 0:
            print("[WARN] One of the filtered sets is empty; FID is undefined.", file=sys.stderr)
            result = {"FID_org_non_targets": float("nan"),
                      "num_gen": n_gen, "num_orig": n_orig}
            if args.out_json:
                os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
                with open(args.out_json, "w", encoding="utf-8") as f:
                    json.dump(result, f, indent=2)
                print(f"[Info] Wrote JSON: {args.out_json}")
            # still exit 0, but warn
            return

        print("[Info] Computing FID_org (CleanFID) on non-target subsets...")
        fid_val = float(fid.compute_fid(gen_filtered, orig_filtered, mode=args.mode))
        print(f"[Result] FID_org (non-targets): {fid_val:.4f}")

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
        # Clean up temp dirs
        shutil.rmtree(tmp_root, ignore_errors=True)


if __name__ == "__main__":
    main()
