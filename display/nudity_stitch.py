#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import sys
from math import ceil
from pathlib import Path
from typing import List, Tuple, Iterable

from PIL import Image

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def list_images(folder: Path) -> List[Path]:
    return sorted(
        [p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS and p.is_file()],
        key=lambda p: p.name
    )


def basename_noext(p: Path) -> str:
    return p.stem


def intersection_by_name(f1_files: List[Path], f2_files: List[Path]) -> Tuple[List[Path], List[Path], List[str]]:
    f1_map = {basename_noext(p): p for p in f1_files}
    f2_map = {basename_noext(p): p for p in f2_files}
    common = sorted(set(f1_map.keys()) & set(f2_map.keys()))
    return [f1_map[k] for k in common], [f2_map[k] for k in common], common


def load_and_fit(im_path: Path, size: Tuple[int, int], keep_aspect: bool, resample=Image.BICUBIC) -> Image.Image:
    img = Image.open(im_path).convert("RGB")
    target_w, target_h = size
    if keep_aspect:
        img.thumbnail((target_w, target_h), resample)
        # Paste centered on canvas
        canvas = Image.new("RGB", (target_w, target_h), (255, 255, 255))
        x = (target_w - img.width) // 2
        y = (target_h - img.height) // 2
        canvas.paste(img, (x, y))
        return canvas
    else:
        return img.resize((target_w, target_h), resample)


def make_grid(images: Iterable[Image.Image],
              cols: int,
              thumb_size: Tuple[int, int],
              pad: int,
              bg_rgb: Tuple[int, int, int]) -> Image.Image:
    ims = list(images)
    if len(ims) == 0:
        raise ValueError("No images to grid.")
    w, h = thumb_size
    n = len(ims)
    rows = ceil(n / cols)
    grid_w = cols * w + (cols + 1) * pad
    grid_h = rows * h + (rows + 1) * pad
    grid = Image.new("RGB", (grid_w, grid_h), bg_rgb)
    for idx, im in enumerate(ims):
        r = idx // cols
        c = idx % cols
        x = pad + c * (w + pad)
        y = pad + r * (h + pad)
        grid.paste(im, (x, y))
    return grid


def parse_color_hex(s: str) -> Tuple[int, int, int]:
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) == 3:
        s = "".join([ch * 2 for ch in s])
    if len(s) != 6:
        raise argparse.ArgumentTypeError("Color must be #RGB, #RRGGBB, RGB, or RRGGBB")
    r = int(s[0:2], 16)
    g = int(s[2:4], 16)
    b = int(s[4:6], 16)
    return (r, g, b)


def main():
    ap = argparse.ArgumentParser(
        description="Build two grids (one per folder) from images with the same base filenames."
    )
    ap.add_argument("--folder1", required=True, type=Path, help="Path to first folder")
    ap.add_argument("--folder2", required=True, type=Path, help="Path to second folder")
    ap.add_argument("--out1", type=Path, default=Path("grid_folder1.png"), help="Output image for folder1 grid")
    ap.add_argument("--out2", type=Path, default=Path("grid_folder2.png"), help="Output image for folder2 grid")
    ap.add_argument("--cols", type=int, default=4, help="Number of columns in grid")
    ap.add_argument("--thumb_w", type=int, default=384, help="Thumbnail width per cell")
    ap.add_argument("--thumb_h", type=int, default=384, help="Thumbnail height per cell")
    ap.add_argument("--pad", type=int, default=8, help="Padding (pixels) between cells and around edges")
    ap.add_argument("--bg", type=parse_color_hex, default="#ffffff", help="Background color (hex), e.g., #000000 or fff")
    ap.add_argument("--keep_aspect", action="store_true", help="Keep aspect ratio with letterboxing; otherwise stretch")
    args = ap.parse_args()

    f1 = args.folder1
    f2 = args.folder2
    if not f1.is_dir() or not f2.is_dir():
        print("Both --folder1 and --folder2 must be directories.", file=sys.stderr)
        sys.exit(1)

    f1_files = list_images(f1)
    f2_files = list_images(f2)
    f1_common, f2_common, common_names = intersection_by_name(f1_files, f2_files)

    if len(common_names) == 0:
        print("No matching filenames (by stem) found between the two folders.", file=sys.stderr)
        sys.exit(2)

    missing_in_f2 = sorted(set(basename_noext(p) for p in f1_files) - set(common_names))
    missing_in_f1 = sorted(set(basename_noext(p) for p in f2_files) - set(common_names))

    if missing_in_f2:
        print(f"[WARN] {len(missing_in_f2)} files in folder1 have no match in folder2 (by stem).")
    if missing_in_f1:
        print(f"[WARN] {len(missing_in_f1)} files in folder2 have no match in folder1 (by stem).")

    # Prepare thumbnails for each folder in the same order
    size = (args.thumb_w, args.thumb_h)
    thumbs1 = [load_and_fit(p, size, args.keep_aspect) for p in f1_common]
    thumbs2 = [load_and_fit(p, size, args.keep_aspect) for p in f2_common]

    grid1 = make_grid(thumbs1, args.cols, size, args.pad, args.bg)
    grid2 = make_grid(thumbs2, args.cols, size, args.pad, args.bg)

    grid1.save(args.out1)
    grid2.save(args.out2)

    print(f"[OK] Saved: {args.out1.resolve()}")
    print(f"[OK] Saved: {args.out2.resolve()}")


if __name__ == "__main__":
    main()
