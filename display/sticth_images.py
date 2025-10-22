#!/usr/bin/env python3
"""
Two-grid stitcher:
For each matched file "A photo of {name}_<idx>.png", split the image vertically into
left and right halves. Build TWO grids with identical layouts:
  - Left halves  -> --out-left  (e.g., "original" grid)
  - Right halves -> --out-right (e.g., "eraser"   grid)

Row = a name; Cols = 5 images chosen by --pick strategy.

Notes
- Indices can be 1–3 digits (e.g., 7, 14, 099, 999).
- If more than 5 matches exist for a name, pick {smallest|largest|random}.
- Accepts multiple --names "A" "B" ... or a single comma-separated string.
- Handles .png and .PNG.
- Each half is letterboxed independently into square cells so grids align cleanly.
"""

import argparse
import glob
import os
import random
import re
from typing import List, Tuple
from PIL import Image, ImageOps  # pip install pillow

# Accept 1–3 digit suffixes
FNAME_RE = re.compile(r"_(\d{1,3})\.(png|PNG)$")

def parse_hex_color(s: str):
    s = s.strip().lstrip("#")
    if len(s) != 6:
        raise ValueError("Background color must be a 6-hex value like #FFFFFF")
    return tuple(int(s[i:i+2], 16) for i in (0, 2, 4))

def load_names(args) -> List[str]:
    # Support: --names "A" "B" "C"  OR  --names "A, B, C"
    if args.names_file:
        with open(args.names_file, "r", encoding="utf-8") as f:
            names = [ln.strip() for ln in f if ln.strip()]
    else:
        names = args.names or []
        if len(names) == 1 and ("," in names[0]):
            names = [n.strip() for n in names[0].split(",") if n.strip()]
    if not names:
        raise SystemExit("No names provided. Use --names or --names-file.")
    return names

def find_images_for_name(image_dir: str, name: str) -> List[Tuple[int, str]]:
    pat1 = os.path.join(image_dir, f"A_photo_of_{glob.escape(name)}_*.png")
    pat2 = os.path.join(image_dir, f"A_photo_of_{glob.escape(name)}_*.PNG")
    candidates = glob.glob(pat1) + glob.glob(pat2)
    parsed: List[Tuple[int, str]] = []
    for p in candidates:
        m = FNAME_RE.search(p)
        if m:
            idx = int(m.group(1))
            parsed.append((idx, p))
    return parsed

def choose_five(parsed: List[Tuple[int, str]], pick: str) -> List[str]:
    if len(parsed) < 5:
        return []
    parsed.sort(key=lambda x: x[0])
    if pick == "smallest":
        chosen = parsed[:5]
    elif pick == "largest":
        chosen = parsed[-5:]
    elif pick == "random":
        chosen = random.sample(parsed, 5)
    else:
        raise ValueError("--pick must be one of: smallest, largest, random")
    chosen.sort(key=lambda x: x[0])  # stable left→right order
    return [p for _, p in chosen]

def letterbox(im: Image.Image, cell: int, bg=(255, 255, 255)) -> Image.Image:
    im = im.convert("RGB")
    w, h = im.size
    if w <= 0 or h <= 0:
        raise ValueError("Invalid image size.")
    scale = min(cell / w, cell / h)
    nw = max(1, int(round(w * scale)))
    nh = max(1, int(round(h * scale)))
    im = im.resize((nw, nh), Image.BICUBIC)
    pad_w = cell - nw
    pad_h = cell - nh
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top
    return ImageOps.expand(im, border=(left, top, right, bottom), fill=bg)

def split_vertical(im: Image.Image) -> Tuple[Image.Image, Image.Image]:
    """Split image into left and right halves (right gets the extra pixel if odd width)."""
    w, h = im.size
    mid = w // 2
    left_box = (0, 0, mid, h)
    right_box = (mid, 0, w, h)
    return im.crop(left_box), im.crop(right_box)

def make_two_grids(image_dir: str, names: List[str], out_left: str, out_right: str,
                   cell: int, gap: int, bg, pick: str):
    rows = len(names)
    cols = 5  # exactly five per name
    W = cols * cell + (cols - 1) * gap
    H = rows * cell + (rows - 1) * gap

    canvas_L = Image.new("RGB", (W, H), color=bg)  # original grid (left halves)
    canvas_R = Image.new("RGB", (W, H), color=bg)  # eraser   grid (right halves)

    for r, name in enumerate(names):
        parsed = find_images_for_name(image_dir, name)
        files = choose_five(parsed, pick=pick)
        if len(files) != 5:
            found = [p for _, p in sorted(parsed, key=lambda x: x[0])]
            raise SystemExit(
                f'Expected at least 5 images for "{name}", found {len(parsed)}.\n'
                f"Looked for: {os.path.join(image_dir, f'A photo of {name}_*.png')} (and .PNG)\n"
                f"Found: {found}"
            )
        for c, fp in enumerate(files):
            with Image.open(fp) as im:
                # Split FIRST, then letterbox each half independently to keep proportions.
                left_half, right_half = split_vertical(im)
                tile_L = letterbox(left_half, cell, bg=bg)
                tile_R = letterbox(right_half, cell, bg=bg)
            x = c * (cell + gap)
            y = r * (cell + gap)
            canvas_L.paste(tile_L, (x, y))
            canvas_R.paste(tile_R, (x, y))

    os.makedirs(os.path.dirname(out_left) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_right) or ".", exist_ok=True)
    canvas_L.save(out_left)
    canvas_R.save(out_right)
    print(f"Saved left-grid  (original halves) to {out_left}")
    print(f"Saved right-grid (eraser   halves) to {out_right}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image-dir", required=True, help="Folder containing images")
    ap.add_argument("--names", nargs="*", help='List of names, e.g. --names "Donald Trump" "Dog"')
    ap.add_argument("--names-file", help="Text file with one name per line")
    ap.add_argument("--out-left", default="grid_left.png", help="Output PNG for LEFT halves grid")
    ap.add_argument("--out-right", default="grid_right.png", help="Output PNG for RIGHT halves grid")
    ap.add_argument("--cell", type=int, default=384, help="Square tile size (default: 384)")
    ap.add_argument("--gap", type=int, default=8, help="Gap between tiles (default: 8)")
    ap.add_argument("--bg", default="#FFFFFF", help="Background color hex (default: #FFFFFF)")
    ap.add_argument("--pick", choices=["smallest", "largest", "random"], default="smallest",
                   help="If >5 matches exist per name, how to choose the 5 (default: smallest)")
    args = ap.parse_args()

    try:
        bg_rgb = parse_hex_color(args.bg)
    except Exception as e:
        raise SystemExit(str(e))

    names = load_names(args)
    make_two_grids(args.image_dir, names, args.out_left, args.out_right,
                   args.cell, args.gap, bg_rgb, args.pick)

if __name__ == "__main__":
    main()
