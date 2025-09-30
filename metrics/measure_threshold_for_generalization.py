#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, re, argparse, numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from tqdm import tqdm

import torch
import open_clip

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# -------- filename parsing --------
# Accepts "A photo of Donald Trump_12.png" OR "A_photo_of_Donald_Trump_12.png"
NAME_RE = re.compile(r"^A[ _]photo[ _]of[ _](.+)_(\d+)$", re.IGNORECASE)

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

def parse_key(path: Path) -> Optional[Tuple[str, str]]:
    """
    Returns (concept_name_normalized, idx_str) or None if not matching.
    Normalizes concept by converting underscores to spaces and collapsing spaces.
    """
    stem = path.stem  # without extension
    m = NAME_RE.match(stem)
    if not m:
        return None
    concept_raw, idx = m.group(1), m.group(2)
    concept_norm = normalize_concept(concept_raw)
    return concept_norm, idx

def normalize_concept(s: str) -> str:
    # turn underscores to spaces, collapse whitespace, lowercase
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip().lower()
    return s

def collect_by_concepts(all_paths: List[Path], concepts: List[str]) -> List[Path]:
    wanted = set(normalize_concept(c) for c in concepts)
    picked = []
    for p in all_paths:
        pk = parse_key(p)
        if not pk:
            continue
        concept_norm, _ = pk
        if concept_norm in wanted:
            picked.append(p)
    return picked

# -------- CLIP scoring --------
@torch.no_grad()
def clip_scores(
    image_paths: List[Path],
    model, preprocess, tokenizer, device: str,
    text: str,
    batch_size: int = 32,
) -> np.ndarray:
    if not image_paths:
        return np.zeros((0,), dtype=np.float32)

    txt = tokenizer([text]).to(device)
    with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        t_emb = model.encode_text(txt)
    t_emb = t_emb / t_emb.norm(dim=-1, keepdim=True)  # [1, d]

    scores = []
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Scoring"):
        batch_paths = image_paths[i:i+batch_size]
        imgs = []
        for p in batch_paths:
            img = Image.open(p).convert("RGB")
            imgs.append(preprocess(img))
        imgs = torch.stack(imgs, dim=0).to(device)

        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            i_emb = model.encode_image(imgs)
        i_emb = i_emb / i_emb.norm(dim=-1, keepdim=True)
        cs = (i_emb @ t_emb.T).squeeze(1)
        scores.extend(cs.detach().float().cpu().tolist())
    return np.array(scores, dtype=np.float32)

def recall_target_threshold(pos: np.ndarray, target_recall: float = 0.95) -> float:
    # t is the (1 - recall) quantile of positive scores
    q = min(max(1.0 - target_recall, 0.0), 1.0)
    return float(np.quantile(pos, q))

def pr_at_threshold(pos: np.ndarray, neg: np.ndarray, t: float):
    tp = int((pos >= t).sum()); fn = int((pos < t).sum())
    fp = int((neg >= t).sum()); tn = int((neg < t).sum())
    precision = tp / (tp + fp + 1e-9)
    recall    = tp / (tp + fn + 1e-9)
    return dict(t=t, tp=tp, fp=fp, tn=tn, fn=fn, precision=precision, recall=recall)

# -------- main --------
def main():
    ap = argparse.ArgumentParser(description="Filter Trump/non-Trump from one folder by filename and calibrate a CLIP threshold.")
    ap.add_argument("--folder", required=True, type=Path, help="Folder containing all images like 'A photo of {concept}_{idx}.png'")
    ap.add_argument("--targets", required=True, type=str,
                    help="Comma-separated list of target concepts to use as POSITIVES (e.g., 'Donald Trump').")
    ap.add_argument("--non_targets", required=True, type=str,
                    help="Comma-separated list of NON-target concepts to use as NEGATIVES.")
    ap.add_argument("--text", type=str, default="Donald Trump",
                    help="Detection text for CLIP scoring (default: 'Donald Trump').")
    ap.add_argument("--recall", type=float, default=0.95, help="Target recall for threshold on positives (default 0.95).")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--model", default="ViT-L-14-336")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--save_csv", type=Path, default=Path("clip_trump_scores.csv"))
    args = ap.parse_args()

    # Load CLIP
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    model = model.to(args.device).eval()

    # Gather all files
    all_paths = list_images(args.folder)

    # Parse target/non-target lists
    targets = [s.strip() for s in args.targets.split(",") if s.strip()]
    non_targets = [s.strip() for s in args.non_targets.split(",") if s.strip()]
    if not targets or not non_targets:
        raise RuntimeError("Both --targets and --non_targets must be non-empty lists.")

    pos_paths = collect_by_concepts(all_paths, targets)
    neg_paths = collect_by_concepts(all_paths, non_targets)

    print(f"#candidates in folder: {len(all_paths)}")
    print(f"#positives (targets):   {len(pos_paths)} from {targets}")
    print(f"#negatives (non-targets): {len(neg_paths)} from {non_targets}")

    if len(pos_paths) == 0:
        raise RuntimeError("No positive images found for provided --targets.")
    if len(neg_paths) == 0:
        raise RuntimeError("No negative images found for provided --non_targets.")

    # Score
    pos_scores = clip_scores(pos_paths, model, preprocess, tokenizer, args.device, text=args.text, batch_size=args.batch_size)
    neg_scores = clip_scores(neg_paths, model, preprocess, tokenizer, args.device, text=args.text, batch_size=args.batch_size)

    # Calibrate threshold
    t = recall_target_threshold(pos_scores, args.recall)
    stats = pr_at_threshold(pos_scores, neg_scores, t)

    print(f"\n[Calibration] text='{args.text}'  recall target={args.recall:.2f}")
    print(f"Threshold t = {t:.4f}")
    print(f"Pos mean={pos_scores.mean():.4f} std={pos_scores.std():.4f}  Neg mean={neg_scores.mean():.4f} std={neg_scores.std():.4f}")
    print(f"precision={stats['precision']:.4f}  recall={stats['recall']:.4f}  tp={stats['tp']} fp={stats['fp']} tn={stats['tn']} fn={stats['fn']}")
    print(f"\nRule: image_has_trump := (CLIP_cosine(image, '{args.text}') >= {t:.4f})")

    # Save per-image scores CSV
    import csv
    args.save_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.save_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "set", "concept", "idx", "clip_cosine", "predicted_trump@t"])
        for path, score in zip(pos_paths, pos_scores):
            concept, idx = parse_key(path)
            w.writerow([str(path), "pos", concept, idx, f"{score:.6f}", int(score >= t)])
        for path, score in zip(neg_paths, neg_scores):
            concept, idx = parse_key(path)
            w.writerow([str(path), "neg", concept, idx, f"{score:.6f}", int(score >= t)])
    print(f"\nSaved per-image scores to: {args.save_csv}")

if __name__ == "__main__":
    main()
