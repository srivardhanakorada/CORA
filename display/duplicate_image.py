#!/usr/bin/env python3
import argparse
from PIL import Image, ImageOps

def parse_hex(s: str):
    s = s.strip().lstrip("#")
    if len(s) not in (6, 8):
        raise ValueError("Use 6- or 8-hex color like #FFFFFF or #FFFFFFFF")
    vals = tuple(int(s[i:i+2], 16) for i in (0,2,4)) + ((int(s[6:8],16),) if len(s)==8 else ())
    return vals

def duplicate_image(
    src_path: str,
    dst_path: str,
    direction: str = "horizontal",
    copies: int = 2,
    gap: int = 0,
    bg: str = None,
):
    im = Image.open(src_path)
    mode = im.mode

    # If adding gaps and image has transparency, use RGBA background unless custom bg provided
    if bg is not None:
        bg_color = parse_hex(bg)
        canvas_mode = "RGBA" if (len(bg_color) == 4 or (mode in ("RGBA","LA"))) else "RGB"
    else:
        # default transparent if source has alpha else white
        if mode in ("RGBA","LA"):
            bg_color = (0,0,0,0)
            canvas_mode = "RGBA"
        else:
            bg_color = (255,255,255)
            canvas_mode = "RGB"

    w, h = im.size
    copies = max(2, copies)  # at least 2

    if direction.lower().startswith("h"):  # horizontal
        W = copies * w + (copies - 1) * gap
        H = h
        canvas = Image.new(canvas_mode, (W, H), bg_color)
        x = 0
        for _ in range(copies):
            canvas.paste(im, (x, 0))
            x += w + gap
    else:  # vertical
        W = w
        H = copies * h + (copies - 1) * gap
        canvas = Image.new(canvas_mode, (W, H), bg_color)
        y = 0
        for _ in range(copies):
            canvas.paste(im, (0, y))
            y += h + gap

    # Convert back to original mode if possible when no transparency is needed
    if canvas_mode == "RGBA" and (bg is not None and len(parse_hex(bg)) == 3):
        # User asked for opaque bg; we can safely drop alpha
        canvas = canvas.convert("RGB")

    canvas.save(dst_path)
    print(f"Saved duplicated image to {dst_path}")

def main():
    ap = argparse.ArgumentParser(description="Duplicate an image side-by-side or stacked.")
    ap.add_argument("input", help="Input image path")
    ap.add_argument("output", help="Output image path (e.g., out.png)")
    ap.add_argument("--direction", choices=["horizontal", "vertical"], default="horizontal",
                    help="Concatenate direction (default: horizontal)")
    ap.add_argument("--copies", type=int, default=2,
                    help="Number of times to replicate the image (default: 2)")
    ap.add_argument("--gap", type=int, default=0, help="Gap in pixels between copies (default: 0)")
    ap.add_argument("--bg", type=str, default=None,
                    help="Background color hex (e.g., #FFFFFF or #00000000). "
                         "Only used for gaps/canvas; default is transparent if input has alpha, else white.")
    args = ap.parse_args()

    duplicate_image(args.input, args.output, args.direction, args.copies, args.gap, args.bg)

if __name__ == "__main__":
    main()
