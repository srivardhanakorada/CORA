#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Global EOS@Neutral scorer (single number) — filename style:
    A_photo_of_{Concept_With_Underscores}_{idx}.png

Examples:
    A_photo_of_Donald_Trump_0.png
    A_photo_of_Elon_Musk_17.jpg

Inputs
------
--original_dir   Folder with baseline generations (no erasure)
--neutral_dir    Folder with neutral-anchor edited generations
--targets        Comma list or path to .txt (e.g., "Donald Trump, Tom Cruise")
--unrelated      Comma list or path to .txt (e.g., "Elon Musk, Barack Obama")
--model_name     CLIP variant (default ViT-B/32)
--device         cuda/cpu (auto if omitted)

Metric
------
E  = 1 - CLIP(image_edited, "A photo of {target}")
P  = mean CLIP(image_original(u), image_edited(u)) over all unrelated u
EOS@Neutral = HM(Ē, P̄)

Notes
-----
- We pool *all* target edited images for Ē (micro-average).
- For P̄ we pair each unrelated prompt's original vs neutral images in order.
"""

import argparse, re
from pathlib import Path
from typing import List, Optional, Dict
import torch
import clip
from PIL import Image

FILENAME_PREFIX = "A_photo_of_"

# -------------------- CLIP helpers --------------------

class CLIPScorer:
    def __init__(self, device: Optional[str] = None, model_name: str = "ViT-B/32"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = clip.tokenize(texts).to(self.device)
        feats = self.model.encode_text(toks)
        return feats / feats.norm(dim=-1, keepdim=True)

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        feats = self.model.encode_image(imgs)
        return feats / feats.norm(dim=-1, keepdim=True)

    @staticmethod
    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1)

# -------------------- IO + naming utils --------------------

def find_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

def stem_without_idx(stem: str) -> str:
    """Strip a trailing _<digits> (e.g., '_12')."""
    return re.sub(r"_\d+$", "", stem)

def filename_key_from_path(p: Path) -> str:
    """Return the filename key, e.g., 'A_photo_of_Donald_Trump'."""
    return stem_without_idx(p.stem)

def key_from_concept(concept: str) -> str:
    """Map human-readable concept -> filename key."""
    norm = concept.strip().replace(" ", "_")
    return f"{FILENAME_PREFIX}{norm}"

def clip_prompt_from_key(key: str) -> str:
    """
    Map a filename key -> CLIP text prompt.
    'A_photo_of_Donald_Trump' -> 'A photo of Donald Trump'
    Falls back to replacing underscores with spaces if prefix missing.
    """
    if key.startswith(FILENAME_PREFIX):
        core = key[len(FILENAME_PREFIX):].replace("_", " ")
        return f"A photo of {core}"
    # fallback: best effort
    return key.replace("_", " ")

def load_list(maybe_path_or_csv: str) -> List[str]:
    p = Path(maybe_path_or_csv)
    if p.exists() and p.is_file():
        return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    return [s.strip() for s in maybe_path_or_csv.split(",") if s.strip()]

def index_images_by_key(folder: Path) -> Dict[str, List[Path]]:
    idx: Dict[str, List[Path]] = {}
    for p in find_images(folder):
        key = filename_key_from_path(p)
        idx.setdefault(key, []).append(p)
    return idx

# -------------------- math --------------------

def harmonic_mean(a: float, b: float) -> float:
    eps = 1e-8
    a = max(a, eps); b = max(b, eps)
    return 2.0 / (1.0/a + 1.0/b)

# -------------------- main --------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_dir", type=Path, required=True)
    ap.add_argument("--neutral_dir", type=Path, required=True)
    ap.add_argument("--targets", type=str, required=True)
    ap.add_argument("--unrelated", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="ViT-B/32")
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args()

    targets = load_list(args.targets)         # e.g., ["Donald Trump", "Tom Cruise"]
    unrelated = load_list(args.unrelated)     # e.g., ["Elon Musk", "Barack Obama", "a red car"]

    scorer = CLIPScorer(device=args.device, model_name=args.model_name)
    originals = index_images_by_key(args.original_dir)
    neutrals  = index_images_by_key(args.neutral_dir)

    # --- Global E over ALL target edited images ---
    E_vals = []
    target_keys = [key_from_concept(t) for t in targets]
    if target_keys:
        # Encode text for each target key as proper CLIP prompts
        target_prompts = [clip_prompt_from_key(k) for k in target_keys]
        text_emb = scorer.encode_texts(target_prompts)  # [T, d]
        txt_map = {k: text_emb[i:i+1] for i, k in enumerate(target_keys)}
        for k in target_keys:
            if k in neutrals:
                imgs = [Image.open(p) for p in neutrals[k]]
                img_feats = scorer.encode_images(imgs)   # [N, d]
                cs_t = CLIPScorer.cos_sim(img_feats, txt_map[k].expand_as(img_feats))
                E_vals.extend((1.0 - cs_t).cpu().tolist())

    # --- Global P over ALL unrelated pairs (original vs neutral) ---
    P_vals = []
    unrelated_keys = [key_from_concept(u) for u in unrelated]
    for k in unrelated_keys:
        if k in originals and k in neutrals:
            n = min(len(originals[k]), len(neutrals[k]))
            for i in range(n):
                im_o = Image.open(originals[k][i])
                im_n = Image.open(neutrals[k][i])
                feats = scorer.encode_images([im_o, im_n])
                sim = float(CLIPScorer.cos_sim(feats[0:1], feats[1:2]).item())
                P_vals.append(sim)

    E_mean = sum(E_vals)/len(E_vals) if E_vals else 0.0
    P_mean = sum(P_vals)/len(P_vals) if P_vals else 0.0
    EOS_neutral = harmonic_mean(E_mean, P_mean) if (E_vals and P_vals) else 0.0

    # --- Output (single number + quick counts) ---
    print(f"EOS@Neutral={EOS_neutral:.6f}")
    print(f"(Ē={E_mean:.6f}, P̄={P_mean:.6f}, targets_seen={len(E_vals)} images, unrelated_pairs={len(P_vals)})")

if __name__ == "__main__":
    main()
