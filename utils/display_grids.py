#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, sys
from pathlib import Path
from typing import List, Tuple, Dict, Optional
from PIL import Image, ImageOps

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NAME_RE = re.compile(r"^A[_ ]photo[_ ]of[_ ](.+?)_(\d+)$", re.IGNORECASE)

def norm_concept_for_filename(c: str) -> str:
    c = c.strip().replace(" ", "_")
    c = re.sub(r"_+", "_", c)
    return c.lower()

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

def parse_stem(stem: str) -> Optional[Tuple[str, int]]:
    m = NAME_RE.match(stem)
    if not m:
        return None
    concept_raw, idx_str = m.group(1), m.group(2)
    key = norm_concept_for_filename(concept_raw)
    try:
        idx = int(idx_str)
    except ValueError:
        return None
    return key, idx

def load_and_fit(path: Path, tile_w: int, tile_h: int, pad_color=(255, 255, 255)) -> Image.Image:
    im = Image.open(path).convert("RGB")
    fitted = ImageOps.contain(im, (tile_w, tile_h))
    canvas = Image.new("RGB", (tile_w, tile_h), pad_color)
    off_x = (tile_w - fitted.width) // 2
    off_y = (tile_h - fitted.height) // 2
    canvas.paste(fitted, (off_x, off_y))
    return canvas

def build_index(paths: List[Path]) -> Dict[str, List[Tuple[int, Path]]]:
    idx_map: Dict[str, List[Tuple[int, Path]]] = {}
    for p in paths:
        parsed = parse_stem(p.stem)
        if not parsed:
            continue
        key, idx = parsed
        idx_map.setdefault(key, []).append((idx, p))
    for k in idx_map:
        idx_map[k].sort(key=lambda t: t[0])
    return idx_map

def make_grid(
    folder: Path,
    raw_concepts: List[str],
    per_concept: int,
    tile_w: int,
    tile_h: int,
    pad: int,
    row_gap: int,
    out_path: Path,
) -> None:
    paths = list_images(folder)
    if not paths:
        print(f"[ERROR] No images found under: {folder}", file=sys.stderr)
        sys.exit(1)

    index = build_index(paths)

    concepts = [c.strip() for c in raw_concepts if c.strip()]
    concept_keys = [norm_concept_for_filename(c) for c in concepts]

    rows: List[List[Image.Image]] = []
    for user_c, key in zip(concepts, concept_keys):
        if key not in index:
            print(f"[WARN] No files for concept '{user_c}' (key='{key}')", file=sys.stderr)
            continue
        entries = index[key]
        if len(entries) < per_concept:
            print(f"[WARN] Concept '{user_c}' has only {len(entries)} images; requested {per_concept}. Using available.", file=sys.stderr)
        chosen_paths: List[Optional[Path]] = [p for _, p in entries[:per_concept]]
        while len(chosen_paths) < per_concept:
            chosen_paths.append(None)
        row_imgs = [
            (load_and_fit(p, tile_w, tile_h) if p is not None else Image.new("RGB", (tile_w, tile_h), (240, 240, 240)))
            for p in chosen_paths
        ]
        rows.append(row_imgs)

    if not rows:
        print("[ERROR] No valid rows to compose.", file=sys.stderr)
        sys.exit(1)

    cols = per_concept
    rows_n = len(rows)

    # Width: keep column gaps = pad, and outer left/right margins = pad
    W = cols * tile_w + (cols + 1) * pad
    # Height: NO space between rows except a controllable row_gap (default 0); keep top/bottom margins = pad
    H = rows_n * tile_h + 2 * pad + (rows_n - 1) * row_gap

    grid = Image.new("RGB", (W, H), (255, 255, 255))

    for r, row_imgs in enumerate(rows):
        for c, im in enumerate(row_imgs):
            x = pad + c * (tile_w + pad)
            y = pad + r * tile_h + r * row_gap
            grid.paste(im, (x, y))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path)
    print(f"[OK] Saved grid to: {out_path}")

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build a concept-wise image grid.")
    ap.add_argument("--folder", type=Path, required=True, help="Folder with images.")
    ap.add_argument("--concepts", type=str, required=True,
                    help='Comma-separated list, e.g. "Donald Trump, Anne Hathaway, Bill Gates, Elon Musk"')
    ap.add_argument("--per_concept", type=int, default=3, help="Images per concept (columns).")
    ap.add_argument("--tile_w", type=int, default=256, help="Tile width.")
    ap.add_argument("--tile_h", type=int, default=256, help="Tile height.")
    ap.add_argument("--pad", type=int, default=10, help="Padding (pixels) between columns and outer margins).")
    ap.add_argument("--row_gap", type=int, default=0, help="Gap (pixels) between rows; set 0 for no space.")
    ap.add_argument("--out", type=Path, default=Path("grid.png"), help="Output image path.")
    return ap.parse_args()

if __name__ == "__main__":
    args = parse_args()
    concept_list = [s.strip() for s in args.concepts.split(",")]
    make_grid(
        folder=args.folder,
        raw_concepts=concept_list,
        per_concept=args.per_concept,
        tile_w=args.tile_w,
        tile_h=args.tile_h,
        pad=args.pad,
        row_gap=args.row_gap,
        out_path=args.out,
    )
