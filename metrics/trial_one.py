#!/usr/bin/env python3
"""
Compute EOS@Intended, EOS@Neutral, and RAS for concept-erasure experiments.

Assumptions
-----------
1) You provide two directories:
   - original_dir: baseline generations (no erasure)
   - edited_dir: generations after applying an erasure method (with intended or neutral replacement)
2) Every image filename IS the prompt string (without extension), e.g.:
   "A photo of Donald Trump.png"
   "A photo of a man.jpg"
3) You pass:
   --target "Donald Trump"
   --replacement "a man"          # intended replacement, e.g. "a man" or any phrase
   --neutral "a man"              # neutral anchor phrase used for EOS@Neutral
   --unrelated "Elon Musk, Barack Obama"   # comma-separated list of prompts considered non-target

Outputs
-------
- Prints aggregate E, P, R and the harmonic means:
  * EOS@Intended = HM(E, P) computed on the edited_dir that used your intended replacement
  * EOS@Neutral  = HM(E, P) computed on the edited_dir that used the neutral anchor
  * RAS          = HM(E, P, R) (only meaningful for intended-replacement runs)
- Also writes a CSV with per-image scores.

Notes
-----
- E = 1 - CLIP(image, "A photo of {target}")
- P = mean CLIP(image_original, image_edited) over unrelated prompts (image-image cosine)
- R = CLIP(image_edited, "A photo of {replacement}")
- Harmonic mean is used to combine atomic scores into EOS and RAS.
"""

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
import clip  # OpenAI CLIP only
import re


# ----------------------------- CLIP helpers -----------------------------

class CLIPScorer:
    def __init__(self, device: Optional[str] = None, model_name: str = "ViT-B/32"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        tokens = clip.tokenize(texts).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        imgs = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        feats = self.model.encode_image(imgs)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @staticmethod
    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1)


# ----------------------------- IO/parsing -------------------------------


def concept_from_filename(path: Path) -> str:
    """
    Extract base prompt from filename, dropping numeric suffix.
    Example:
        "A photo of Donald Trump_123" -> "A photo of Donald Trump"
    """
    stem = path.stem
    # remove "_<digits>" at end if present
    return re.sub(r"_\d+$", "", stem)

def find_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

def build_prompt(prompt_core: str) -> str:
    if prompt_core.lower().startswith("a photo of"):
        return prompt_core
    return f"A photo of {prompt_core}"


# ----------------------------- Metrics ----------------------------------

def harmonic_mean(values: List[float]) -> float:
    eps = 1e-8
    vals = [max(v, eps) for v in values]
    return len(vals) / sum(1.0 / v for v in vals)

def score_run(
    original_dir: Path,
    edited_dir: Path,
    target: str,
    replacement: str,
    unrelated_list: List[str],
    neutral_anchor: str,
    csv_out: Path,
    device: Optional[str] = None,
    model_name: str = "ViT-B/32"
) -> Dict[str, float]:
    scorer = CLIPScorer(device=device, model_name=model_name)

    orig_images = {concept_from_filename(p): p for p in find_images(original_dir)}
    edit_images = {concept_from_filename(p): p for p in find_images(edited_dir)}

    target_text = build_prompt(target)
    repl_text   = build_prompt(replacement)
    neut_text   = build_prompt(neutral_anchor)

    texts_to_encode = [target_text, repl_text, neut_text] + [build_prompt(u) for u in unrelated_list]
    text_embeds = scorer.encode_texts(texts_to_encode)
    t_idx, r_idx, n_idx = 0, 1, 2

    rows = [["prompt", "which_run", "E(1-CS_tgt)", "P(img-img on unrelated)", "R(CS_repl)",
             "CS_target", "CS_replacement", "pair_preservation"]]

    E_list, R_list = [], []

    target_prompt_key = build_prompt(target)
    edited_target_items = [(k, v) for k, v in edit_images.items() if k == target_prompt_key]

    if edited_target_items:
        imgs = [Image.open(p) for _, p in edited_target_items]
        img_feats = scorer.encode_images(imgs)
        cs_target = CLIPScorer.cos_sim(img_feats, text_embeds[t_idx].expand_as(img_feats)).cpu().tolist()
        cs_repl   = CLIPScorer.cos_sim(img_feats, text_embeds[r_idx].expand_as(img_feats)).cpu().tolist()

        for (k, _), cst, csr in zip(edited_target_items, cs_target, cs_repl):
            E = float(1.0 - cst)
            R = float(csr)
            E_list.append(E)
            R_list.append(R)
            rows.append([k, "target", f"{E:.6f}", "", f"{R:.6f}", f"{cst:.6f}", f"{csr:.6f}", ""])
    else:
        print(f"[WARN] No edited images found for target prompt: '{target_prompt_key}'")

    P_pairs: List[float] = []
    for u in unrelated_list:
        key = build_prompt(u)
        if key in orig_images and key in edit_images:
            im_orig = Image.open(orig_images[key])
            im_edit = Image.open(edit_images[key])
            feats = scorer.encode_images([im_orig, im_edit])
            sim = float(CLIPScorer.cos_sim(feats[0:1], feats[1:2]).item())
            P_pairs.append(sim)
            rows.append([key, "unrelated_pair", "", f"{sim:.6f}", "", "", "", f"{sim:.6f}"])

    E_mean = sum(E_list) / len(E_list) if E_list else 0.0
    P_mean = sum(P_pairs) / len(P_pairs) if P_pairs else 0.0
    R_mean = sum(R_list) / len(R_list) if R_list else 0.0

    EOS = harmonic_mean([E_mean, P_mean]) if (E_list and P_pairs) else 0.0
    RAS = harmonic_mean([E_mean, P_mean, R_mean]) if (E_list and P_pairs and R_list) else 0.0

    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    return {
        "E_mean": E_mean,
        "P_mean": P_mean,
        "R_mean": R_mean,
        "EOS": EOS,
        "RAS": RAS,
        "num_target_imgs": len(E_list),
        "num_unrelated_pairs": len(P_pairs),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_dir", type=Path, required=True)
    ap.add_argument("--edited_intended_dir", type=Path, required=True)
    ap.add_argument("--edited_neutral_dir", type=Path, required=True)
    ap.add_argument("--target", type=str, required=True)
    ap.add_argument("--replacement", type=str, required=True)
    ap.add_argument("--neutral", type=str, default="a man")
    ap.add_argument("--unrelated", type=str, required=True)
    ap.add_argument("--model_name", type=str, default="ViT-B/32")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_dir", type=Path, default=Path("metric_outputs"))
    args = ap.parse_args()

    unrelated_list = [s.strip() for s in args.unrelated.split(",") if s.strip()]

    intended_csv = args.out_dir / "per_image_intended.csv"
    intended = score_run(args.original_dir, args.edited_intended_dir,
                         args.target, args.replacement, unrelated_list, args.neutral,
                         intended_csv, args.device, args.model_name)

    neutral_csv = args.out_dir / "per_image_neutral.csv"
    neutral = score_run(args.original_dir, args.edited_neutral_dir,
                        args.target, args.neutral, unrelated_list, args.neutral,
                        neutral_csv, args.device, args.model_name)

    print("\n===== Results =====")
    print(f"Target: {args.target}")
    print(f"Intended replacement: {args.replacement}")
    print(f"Neutral anchor: {args.neutral}")
    print(f"Unrelated prompts: {unrelated_list}")

    print("\n-- Intended run --")
    print(f"Ē: {intended['E_mean']:.4f}")
    print(f"P̄: {intended['P_mean']:.4f}")
    print(f"R̄: {intended['R_mean']:.4f}")
    print(f"EOS@Intended: {intended['EOS']:.4f}")
    print(f"RAS: {intended['RAS']:.4f}")

    print("\n-- Neutral run --")
    print(f"Ē: {neutral['E_mean']:.4f}")
    print(f"P̄: {neutral['P_mean']:.4f}")
    print(f"EOS@Neutral: {neutral['EOS']:.4f}")


if __name__ == "__main__":
    main()
