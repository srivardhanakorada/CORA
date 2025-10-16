#!/usr/bin/env python3
"""
Compute two FIDs for a generated images folder using CleanFID:

1) FIR_org  : FID(gen_folder  ||  orig_folder)
2) FID_COCO : FID(gen_folder  ||  COCO-30k reference stats from a local .npz)

Requirements:
    pip install clean-fid

Notes:
- We keep CleanFID defaults (Inception-V3, mode="clean").
- You supply a local .npz (with keys 'mu' and 'sigma') for COCO-30k once; the
  script registers it under a name in CleanFID's stats cache and then computes FID.
"""

import argparse
import json
import os
import shutil
import sys
from typing import Dict, Any, Optional

from cleanfid import fid  # type: ignore


def _stats_cache_dir() -> str:
    """Return CleanFID's internal stats directory."""
    import cleanfid as _cleanfid  # type: ignore
    return os.path.join(os.path.dirname(_cleanfid.__file__), "stats")


def _install_custom_stats(npz_path: str, custom_name: str, mode: str = "clean") -> str:
    """
    Copy your local stats file into CleanFID's cache with the expected filename:
        <custom_name>_<mode>_custom_na.npz
    Returns the destination path.
    """
    if not os.path.isfile(npz_path):
        raise FileNotFoundError(f"Stats npz not found: {npz_path}")

    dst_dir = _stats_cache_dir()
    os.makedirs(dst_dir, exist_ok=True)

    base = f"{custom_name}_{mode}_custom_na".lower()
    dst = os.path.join(dst_dir, base + ".npz")
    shutil.copyfile(npz_path, dst)
    return dst


def compute_fir_org(gen_folder: str, orig_folder: str, mode: str = "clean") -> float:
    """FID between two folders (generated vs original)."""
    return float(fid.compute_fid(gen_folder, orig_folder, mode=mode))


def ensure_custom_stats_available(custom_name: str, npz_path: Optional[str], mode: str = "clean") -> None:
    """
    Ensure the custom stats referenced by 'custom_name' exist in CleanFID's cache.
    If missing and npz_path is provided, install them. Otherwise, raise.
    """
    if fid.test_stats_exists(custom_name, mode):
        return
    if npz_path:
        _install_custom_stats(npz_path, custom_name, mode=mode)
        if fid.test_stats_exists(custom_name, mode):
            return
    raise FileNotFoundError(
        f"CleanFID stats '{custom_name}' not found and no valid --coco_stats_npz provided.\n"
        f"Either pass --coco_stats_npz or drop a file named "
        f"'{custom_name}_{mode}_custom_na.npz' into: {_stats_cache_dir()}"
    )


def compute_fid_vs_custom(gen_folder: str, custom_name: str, mode: str = "clean") -> float:
    """FID between a folder and a preinstalled custom stats pack (e.g., COCO-30k)."""
    return float(fid.compute_fid(
        gen_folder,
        dataset_name=custom_name,
        dataset_split="custom",
        mode=mode,
    ))


def main():
    ap = argparse.ArgumentParser(description="Compute FIR_org and FID_COCO with CleanFID.")
    ap.add_argument("--gen_folder", required=True, help="Folder of generated images")
    ap.add_argument("--orig_folder", required=False, help="Folder of original images (for FIR_org)")
    ap.add_argument("--coco_stats_name", required=True,
                    help="Name to register/use for the COCO-30k stats (e.g., coco30k_val_clean)")
    ap.add_argument("--coco_stats_npz", required=False,
                    help="Path to local COCO-30k stats .npz (mu/sigma). "
                         "Provide once; then you can omit it and reuse the name.")
    ap.add_argument("--mode", default="clean",
                    choices=["clean", "legacy_pytorch", "legacy_tensorflow"],
                    help="CleanFID mode (default: clean)")
    ap.add_argument("--out_json", default=None, help="Optional JSON output path for the numbers")
    args = ap.parse_args()

    # Basic checks
    if not os.path.isdir(args.gen_folder):
        print(f"[ERROR] --gen_folder not found: {args.gen_folder}", file=sys.stderr)
        sys.exit(2)
    if args.orig_folder and not os.path.isdir(args.orig_folder):
        print(f"[ERROR] --orig_folder not found: {args.orig_folder}", file=sys.stderr)
        sys.exit(2)

    results: Dict[str, Any] = {}

    # 1) FIR_org (if orig_folder specified)
    if args.orig_folder:
        print(f"[Info] FIR_org: computing FID between\n  gen : {args.gen_folder}\n  orig: {args.orig_folder}")
        fir_org = compute_fir_org(args.gen_folder, args.orig_folder, mode=args.mode)
        results["FIR_org"] = fir_org
        print(f"[Result] FIR_org = {fir_org:.4f}")
    else:
        print("[Info] Skipping FIR_org (no --orig_folder).")

    # 2) FID_COCO (vs custom stats)
    print(f"[Info] Ensuring custom stats '{args.coco_stats_name}' are available...")
    ensure_custom_stats_available(args.coco_stats_name, args.coco_stats_npz, mode=args.mode)
    print(f"[Info] FID_COCO: computing FID between\n  gen : {args.gen_folder}\n  ref : {args.coco_stats_name} (custom stats)")
    fid_coco = compute_fid_vs_custom(args.gen_folder, args.coco_stats_name, mode=args.mode)
    results["FID_COCO"] = fid_coco
    print(f"[Result] FID_COCO = {fid_coco:.4f}")

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"[Info] Wrote JSON: {args.out_json}")


if __name__ == "__main__":
    main()
