#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from PIL import Image, ImageOps

def split_horizontal_to_vertical(img: Image.Image, center_gutter: int = 0,
                                 inter_border: int = 0, border_color=(255,255,255)) -> Image.Image:
    """
    Convert a 2-up horizontal grid into a vertical grid.
    Assumes the input image is two equal-width panels side-by-side (optionally with a center gutter).
    - center_gutter: pixels to skip between left and right halves in the source.
    - inter_border: pixels to insert between the two panels in the vertical result.
    """
    w, h = img.size
    if center_gutter < 0:
        raise ValueError("center_gutter must be >= 0")
    if inter_border < 0:
        raise ValueError("inter_border must be >= 0")

    # Compute half widths, accounting for a center gutter
    # left width = (w - center_gutter) // 2, right width = remaining
    left_w = (w - center_gutter) // 2
    right_w = w - center_gutter - left_w

    if left_w <= 0 or right_w <= 0:
        raise ValueError("Invalid sizes after accounting for center_gutter. Check your input/gutter.")

    # Crop left and right
    left = img.crop((0, 0, left_w, h))
    right = img.crop((left_w + center_gutter, 0, left_w + center_gutter + right_w, h))

    # Stack vertically (top: left, bottom: right)
    out_w = max(left.width, right.width)
    out_h = left.height + inter_border + right.height

    out = Image.new("RGB", (out_w, out_h), color=border_color)
    out.paste(left, (0, 0))
    if inter_border > 0:
        # draw the border as a solid rectangle
        border_rect = Image.new("RGB", (out_w, inter_border), color=border_color)
        out.paste(border_rect, (0, left.height))
    out.paste(right, (0, left.height + inter_border))
    return out

def resize_to_same_height(images, target_height=None):
    """
    Resize all images to the same height (keeping aspect ratio).
    If target_height is None, uses the max height among inputs.
    """
    if target_height is None:
        target_height = max(im.height for im in images)
    outs = []
    for im in images:
        if im.height == target_height:
            outs.append(im)
        else:
            new_w = int(round(im.width * (target_height / im.height)))
            outs.append(im.resize((new_w, target_height), Image.LANCZOS))
    return outs

def hconcat(images, h_spacing: int = 0, bg=(255,255,255)) -> Image.Image:
    """
    Horizontally concatenate images (assumes same height).
    """
    if not images:
        raise ValueError("No images to concatenate.")

    h = images[0].height
    if any(im.height != h for im in images):
        raise ValueError("All images must have the same height to hconcat (resize them first).")

    total_width = sum(im.width for im in images) + h_spacing * (len(images) - 1)
    out = Image.new("RGB", (total_width, h), color=bg)

    x = 0
    for idx, im in enumerate(images):
        out.paste(im, (x, 0))
        x += im.width
        if idx < len(images) - 1:
            x += h_spacing
    return out

def add_outer_border(im: Image.Image, border: int, color=(255,255,255)) -> Image.Image:
    if border <= 0:
        return im
    return ImageOps.expand(im, border=border, fill=color)

def main():
    parser = argparse.ArgumentParser(
        description="Convert four horizontal 2-up grids into vertical stacks, then concat horizontally."
    )
    parser.add_argument("img1", help="Path to horizontal 2-up image #1")
    parser.add_argument("img2", help="Path to horizontal 2-up image #2")
    parser.add_argument("img3", help="Path to horizontal 2-up image #3")
    parser.add_argument("img4", help="Path to horizontal 2-up image #4")
    parser.add_argument("-o", "--output", default="final_grid.jpg", help="Output image path")
    parser.add_argument("--center-gutter", type=int, default=0,
                        help="Pixels to skip between left and right halves in each source image")
    parser.add_argument("--inter-border", type=int, default=0,
                        help="Pixels between the two panels in each vertical stack")
    parser.add_argument("--h-spacing", type=int, default=0,
                        help="Pixels between the four vertical stacks in the final horizontal concat")
    parser.add_argument("--bg", default="#FFFFFF",
                        help="Background/border color (hex like #FFFFFF or #000000)")
    parser.add_argument("--target-height", type=int, default=None,
                        help="Force final stacks to this height before horizontal concat (keep aspect).")
    parser.add_argument("--outer-border", type=int, default=0,
                        help="Add an outer border (pixels) to the final image")
    args = parser.parse_args()

    # Parse bg color
    bg = args.bg
    if isinstance(bg, str) and bg.startswith("#") and len(bg) in (4, 7):
        # Convert hex to RGB
        bg = bg.lstrip("#")
        if len(bg) == 3:
            bg = "".join([c*2 for c in bg])
        r = int(bg[0:2], 16)
        g = int(bg[2:4], 16)
        b = int(bg[4:6], 16)
        bg = (r, g, b)
    elif isinstance(bg, str):
        raise ValueError("bg must be a hex color like #RRGGBB")

    # Load images
    imgs = [Image.open(p).convert("RGB") for p in [args.img1, args.img2, args.img3, args.img4]]

    # Convert each horizontal 2-up into a vertical stack
    stacks = [
        split_horizontal_to_vertical(im,
                                     center_gutter=args.center_gutter,
                                     inter_border=args.inter_border,
                                     border_color=bg)
        for im in imgs
    ]

    # Normalize height (so they can be concatenated horizontally)
    stacks = resize_to_same_height(stacks, target_height=args.target_height)

    # Concatenate horizontally
    final = hconcat(stacks, h_spacing=args.h_spacing, bg=bg)

    # Optional outer border
    final = add_outer_border(final, args.outer_border, color=bg)

    # Save
    final.save(args.output)
    print(f"Saved: {args.output}")

if __name__ == "__main__":
    main()
