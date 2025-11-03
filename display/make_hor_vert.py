#!/usr/bin/env python3
"""
Convert horizontally stacked image grids to vertically stacked,
then place all vertical grids side-by-side in one final composite.
"""

import os
from PIL import Image


def make_vertical_grid(img_path, num_cols=2, out_height=None):
    """
    Split a horizontally stacked image into equal parts (num_cols),
    and stack them vertically.
    """
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    part_w = w // num_cols
    parts = [img.crop((i * part_w, 0, (i + 1) * part_w, h)) for i in range(num_cols)]
    v_stack = Image.new("RGB", (part_w, h * num_cols))
    for i, p in enumerate(parts):
        v_stack.paste(p, (0, i * h))
    if out_height:
        v_stack = v_stack.resize((v_stack.width, out_height))
    return v_stack


def make_composite(folder, num_cols=2, save_path="composite_grid.jpg"):
    """
    Convert all horizontally stacked images in folder to vertical grids
    and place them side-by-side.
    """
    imgs = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder)
         if f.lower().endswith((".jpg", ".png", ".jpeg"))]
    )
    vertical_grids = [make_vertical_grid(img, num_cols=num_cols) for img in imgs]

    # Normalize heights for consistent alignment
    max_h = max(v.height for v in vertical_grids)
    resized = [v.resize((int(v.width * max_h / v.height), max_h)) for v in vertical_grids]

    total_w = sum(v.width for v in resized)
    composite = Image.new("RGB", (total_w, max_h), color=(255, 255, 255))

    x = 0
    for v in resized:
        composite.paste(v, (x, 0))
        x += v.width

    composite.save(save_path)
    print(f"Saved composite to {save_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert horizontally stacked images into vertical grids and combine them."
    )
    parser.add_argument("folder", help="Folder containing horizontally stacked images")
    parser.add_argument("--cols", type=int, default=2, help="Number of horizontal parts per image")
    parser.add_argument("--out", type=str, default="composite_grid.jpg", help="Output image file")

    args = parser.parse_args()
    make_composite(args.folder, num_cols=args.cols, save_path=args.out)
