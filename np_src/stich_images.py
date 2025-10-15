#!/usr/bin/env python3
import argparse, os, sys, re
from PIL import Image

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp"}

# accepted suffixes for left/right
LEFT_SUFFIXES  = ("orig", "original", "before", "org", "gt")
RIGHT_SUFFIXES = ("erase", "erased", "after", "edited", "removed")

LEFT_RE  = re.compile(rf"_(?:{'|'.join(LEFT_SUFFIXES)})$", re.IGNORECASE)
RIGHT_RE = re.compile(rf"_(?:{'|'.join(RIGHT_SUFFIXES)})$", re.IGNORECASE)

def list_images(folder):
    files = []
    for name in os.listdir(folder):
        ext = os.path.splitext(name)[1].lower()
        if ext in VALID_EXTS:
            files.append(name)
    return set(files)

def load_image(path):
    try:
        return Image.open(path).convert("RGB")
    except Exception as e:
        print(f"[WARN] Failed to open {path}: {e}")
        return None

def resize_match(imA, imB, match="height"):
    """Resize B to match A's height/width (keeping aspect)."""
    if match not in {"height", "width", "none"}:
        match = "height"
    if match == "none":
        return imA, imB
    if match == "height":
        h = imA.height
        if imB.height != h:
            w = int(imB.width * (h / imB.height))
            imB = imB.resize((w, h), Image.LANCZOS)
    else:  # match == "width"
        w = imA.width
        if imB.width != w:
            h = int(imB.height * (w / imB.width))
            imB = imB.resize((w, h), Image.LANCZOS)
    return imA, imB

def stitch_side_by_side(left, right, pad=0, bg=(255, 255, 255)):
    """Return a new image with left|right and optional vertical padding."""
    H = max(left.height, right.height)
    W = left.width + pad + right.width
    canvas = Image.new("RGB", (W, H), bg)
    yL = (H - left.height) // 2
    yR = (H - right.height) // 2
    canvas.paste(left, (0, yL))
    canvas.paste(right, (left.width + pad, yR))
    return canvas

def stem_key_for_left(stem: str):
    # remove trailing _orig-like suffix for key
    return LEFT_RE.sub("", stem)

def stem_key_for_right(stem: str):
    # remove trailing _erase-like suffix for key
    return RIGHT_RE.sub("", stem)

def index_by_key(folder, is_left=True):
    """Return dict: base_key -> filename for given folder."""
    keyer = stem_key_for_left if is_left else stem_key_for_right
    mapping = {}
    for name in list_images(folder):
        stem, _ = os.path.splitext(name)
        key = keyer(stem)
        # Only accept if suffix actually matched (i.e., changed)
        if key != stem:
            mapping[key] = name
    return mapping

def main():
    ap = argparse.ArgumentParser(
        description="Pair-stitch *_orig.* with *_erase.* from two subfolders."
    )
    ap.add_argument("root", type=str, help="Root folder containing subfolders")
    ap.add_argument("--left", type=str, default="original",
                    help="Left subfolder name (default: original)")
    ap.add_argument("--right", type=str, default="erase",
                    help="Right subfolder name (default: erase)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output folder (default: <root>/paired_grids)")
    ap.add_argument("--match-dim", type=str, default="height",
                    choices=["height", "width", "none"],
                    help="Resize right to match left's height/width (default: height)")
    ap.add_argument("--pad", type=int, default=0,
                    help="Horizontal padding (pixels) between images")
    ap.add_argument("--bg", type=str, default="255,255,255",
                    help="Background color as R,G,B (default white)")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    if not os.path.isdir(root):
        print(f"[ERR] Root not found: {root}")
        sys.exit(1)

    left_dir = os.path.join(root, args.left)
    right_dir = os.path.join(root, args.right)
    if not os.path.isdir(left_dir):
        print(f"[ERR] Left folder not found: {left_dir}")
        sys.exit(1)
    if not os.path.isdir(right_dir):
        print(f"[ERR] Right folder not found: {right_dir}")
        sys.exit(1)

    out_dir = args.out or os.path.join(root, "paired_grids")
    os.makedirs(out_dir, exist_ok=True)

    # Parse bg color
    try:
        bg = tuple(int(x) for x in args.bg.split(","))
        if len(bg) != 3: raise ValueError
    except Exception:
        print("[WARN] Invalid --bg; using white.")
        bg = (255, 255, 255)

    # Build key->filename mapping for each side
    left_map = index_by_key(left_dir, is_left=True)
    right_map = index_by_key(right_dir, is_left=False)

    # Intersect by base key
    keys = sorted(set(left_map.keys()) & set(right_map.keys()))
    if not keys:
        print(f"[WARN] No matching base keys between {left_dir} and {right_dir}.")
        # small hint if user forgot suffixes
        if not left_map:
            print(f"  Hint: left files must end with _({'|'.join(LEFT_SUFFIXES)})")
        if not right_map:
            print(f"  Hint: right files must end with _({'|'.join(RIGHT_SUFFIXES)})")
        sys.exit(0)

    print(f"[INFO] Found {len(keys)} pairs by base key. Writing to: {out_dir}")

    for key in keys:
        left_name  = left_map[key]
        right_name = right_map[key]
        pL = os.path.join(left_dir, left_name)
        pR = os.path.join(right_dir, right_name)

        imL = load_image(pL)
        imR = load_image(pR)
        if imL is None or imR is None:
            continue

        imL, imR = resize_match(imL, imR, match=args.match_dim)
        grid = stitch_side_by_side(imL, imR, pad=args.pad, bg=bg)

        out_path = os.path.join(out_dir, f"{key}_grid.png")
        try:
            grid.save(out_path, "PNG")
            print(f"[OK] {left_name} + {right_name} -> {out_path}")
        except Exception as e:
            print(f"[WARN] Failed saving {out_path}: {e}")

if __name__ == "__main__":
    main()
