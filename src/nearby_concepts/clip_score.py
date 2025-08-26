import os
import argparse
import csv
from collections import defaultdict
from typing import List, Dict

import torch
import open_clip
from PIL import Image
from tqdm import tqdm

# ----- Config -----
ALLOWED_CLASSES = [
    "Toyota Corolla",
    "Honda Civic",
    "Ford Focus",
    "Toyota Camry",
    "Honda Accord",
    "Ford Fusion",
    "Nissan Altima",
    "Hyundai Elantra",
    "Chevrolet Malibu",
    "Airplane"
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")


def load_model(device: str):
    model, _, preprocess = open_clip.create_model_and_transforms(
        "ViT-B-32", pretrained="laion2b_s34b_b79k"
    )
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    model = model.to(device).eval()
    return model, preprocess, tokenizer


def list_images(folder: str) -> List[str]:
    if not os.path.isdir(folder):
        return []
    out = []
    for f in os.listdir(folder):
        if f.lower().endswith(IMAGE_EXTS):
            out.append(os.path.join(folder, f))
    return sorted(out)


def extract_class_from_filename(fname: str, allowed_classes: List[str]) -> str:
    """
    Normalize filename and match longest allowed class substring.
    """
    base = os.path.basename(fname)
    name = os.path.splitext(base)[0]
    norm = name.replace("_", " ").lower()

    matches = []
    for cls in allowed_classes:
        if cls.lower() in norm:
            matches.append((len(cls), cls))
    if not matches:
        return ""
    matches.sort(reverse=True)
    return matches[0][1]


@torch.no_grad()
def encode_text(model, tokenizer, device: str, prompt: str) -> torch.Tensor:
    text = tokenizer([prompt]).to(device)
    tfeat = model.encode_text(text)
    tfeat /= tfeat.norm(dim=-1, keepdim=True)
    return tfeat  # (1, D)


@torch.no_grad()
def compute_clip_scores_for_files_batched(
    files: List[str],
    text_features: torch.Tensor,
    model,
    preprocess,
    device: str,
    batch_size: int = 32,
) -> float:
    """
    Batched image encoding for speed. Returns average similarity with text_features.
    """
    if not files:
        return 0.0

    sims = []
    for i in range(0, len(files), batch_size):
        batch_paths = files[i : i + batch_size]
        imgs = []
        for fp in batch_paths:
            try:
                img = Image.open(fp).convert("RGB")
                imgs.append(preprocess(img))
            except Exception as e:
                print(f"[warn] error processing {fp}: {e}")
        if not imgs:
            continue

        images = torch.stack(imgs, dim=0).to(device)
        img_feats = model.encode_image(images)
        img_feats /= img_feats.norm(dim=-1, keepdim=True)
        # cosine similarity to single text vector
        # (B, D) @ (D, 1) -> (B, 1)
        batch_sims = (img_feats @ text_features.T).squeeze(-1).tolist()
        sims.extend(batch_sims)

    if not sims:
        return 0.0
    return float(sum(sims) / len(sims))


def gather_files_by_class(folder: str, allowed_classes: List[str]) -> Dict[str, List[str]]:
    """
    Group image file paths by extracted class (only allowed classes).
    """
    files = list_images(folder)
    buckets = defaultdict(list)
    for fp in files:
        cls = extract_class_from_filename(fp, allowed_classes)
        if cls:
            buckets[cls].append(fp)
    return buckets


def main(parent_folder: str, output_csv_path: str):
    """
    Directory layout (strict):
      parent_folder/
        erase/
          original/   # before (erase set)
          retain/     # after  (erase set)
          combine/    # ignored
        retention/
          original/   # before (retention set)
          retain/     # after  (retention set)
          combine/    # ignored
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, preprocess, tokenizer = load_model(device)

    groups = ["erase", "retention"]
    rows = []
    print("\n=== Computing CLIP scores (per group, per class) ===")

    # Cache text features per class to avoid recompute
    textfeat_cache: Dict[str, torch.Tensor] = {}

    for group in groups:
        group_path = os.path.join(parent_folder, group)
        if not os.path.isdir(group_path):
            print(f"[info] skipping missing group: {group_path}")
            continue

        before_dir = os.path.join(group_path, "original")
        after_dir = os.path.join(group_path, "retain")

        # hard fail if folder missing to avoid mixing or surprises
        if not os.path.isdir(before_dir):
            print(f"[warn] missing 'original' for group '{group}': {before_dir}")
        if not os.path.isdir(after_dir):
            print(f"[warn] missing 'retain' for group '{group}': {after_dir}")

        per_class_before = gather_files_by_class(before_dir, ALLOWED_CLASSES) if os.path.isdir(before_dir) else {}
        per_class_after = gather_files_by_class(after_dir, ALLOWED_CLASSES) if os.path.isdir(after_dir) else {}

        classes_here = sorted(
            set(per_class_before.keys()) | set(per_class_after.keys()),
            key=lambda x: ALLOWED_CLASSES.index(x) if x in ALLOWED_CLASSES else 999,
        )

        for cls in classes_here:
            files_before = per_class_before.get(cls, [])
            files_after = per_class_after.get(cls, [])

            # Prepare text features once per class
            if cls not in textfeat_cache:
                textfeat_cache[cls] = encode_text(
                    model, tokenizer, device, prompt=f"A photo of a {cls}"
                )
            tfeat = textfeat_cache[cls]

            score_before = compute_clip_scores_for_files_batched(
                files_before, tfeat, model, preprocess, device
            )
            score_after = compute_clip_scores_for_files_batched(
                files_after, tfeat, model, preprocess, device
            )
            delta = score_after - score_before

            print(
                f"[{group}] {cls:15s}  "
                f"before: {score_before:.4f} (n={len(files_before):3d})  "
                f"after: {score_after:.4f} (n={len(files_after):3d})  "
                f"Δ: {delta:+.4f}"
            )

            rows.append(
                [
                    group,
                    cls,
                    f"{score_before:.6f}",
                    f"{score_after:.6f}",
                    f"{delta:.6f}",
                    len(files_before),
                    len(files_after),
                ]
            )

    # Write CSV
    header = [
        "group",          # erase | retention
        "class_name",
        "before_score",   # from <group>/original/
        "after_score",    # from <group>/retain/
        "delta_after_minus_before",
        "n_before",
        "n_after",
    ]
    os.makedirs(os.path.dirname(output_csv_path) or ".", exist_ok=True)
    with open(output_csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)

    print(f"\nSaved results → {output_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("parent_folder", help="Folder containing 'erase' and 'retention' subfolders")
    parser.add_argument("output_csv", help="Output CSV path")
    args = parser.parse_args()
    main(args.parent_folder, args.output_csv)
