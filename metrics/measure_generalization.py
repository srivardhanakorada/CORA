#!/usr/bin/env python3
"""
Generalization Erasure Measurement (G-EOS)

We evaluate erasure when the edited images are generated with *alias/prompts* for the
target concept (e.g., "POTUS", "President of the United States") rather than the literal name.

Inputs
------
- original_dir: baseline generations (no erasure)
- edited_dir:   generations after applying an erasure method (these were made using alias prompts)
- --target "Donald Trump"
- --aliases "POTUS, President of the United States"
- --unrelated "Elon Musk, Barack Obama"   # prompts we treat as unrelated (for preservation P)
- (optional) --replacement "a man"         # only used if you also want R on aliases
- (optional) --r_report to include R̄_alias and G-RAS = HM(Ē_alias, P̄, R̄_alias)
- (optional) --e_thresh 0.30               # CLIP(target) threshold to count failures (leakage)

Outputs
-------
- Prints Ē_alias, P̄, G-EOS, (and R̄_alias, G-RAS if requested) and failure statistics.
- Writes CSV with per-image rows.

Filename Convention
-------------------
Every image filename is the prompt string (without extension), optionally with a "_<digits>" suffix.
Examples:
  "A photo of POTUS.png"
  "A photo of President of the United States_17.jpg"
"""

import argparse
import csv
import re
from pathlib import Path
from typing import Dict, List, Optional

import torch
from PIL import Image
import clip  # OpenAI CLIP


# ----------------------------- CLIP helpers -----------------------------

class CLIPScorer:
    def __init__(self, device: Optional[str] = None, model_name: str = "ViT-B/32"):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval().to(self.device)

    @torch.no_grad()
    def encode_texts(self, texts: List[str]) -> torch.Tensor:
        toks = clip.tokenize(texts).to(self.device)
        feats = self.model.encode_text(toks)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @torch.no_grad()
    def encode_images(self, images: List[Image.Image]) -> torch.Tensor:
        batch = torch.stack([self.preprocess(im.convert("RGB")) for im in images]).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats

    @staticmethod
    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a * b).sum(dim=-1)


# ----------------------------- IO/parsing -------------------------------

def concept_from_filename(path: Path) -> str:
    """Drop a trailing '_<digits>' numeric suffix to get the prompt key."""
    stem = path.stem
    return re.sub(r"_\d+$", "", stem)

def find_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in exts])

def build_prompt(core: str) -> str:
    return core if core.lower().startswith("a photo of") else f"A photo of {core}"

def harmonic_mean(values: List[float]) -> float:
    eps = 1e-8
    vals = [max(v, eps) for v in values]
    return len(vals) / sum(1.0 / v for v in vals)


# ----------------------------- Scoring ----------------------------------

def score_generalization(
    original_dir: Path,
    edited_dir: Path,
    target: str,
    aliases: List[str],
    unrelated_list: List[str],
    csv_out: Path,
    device: Optional[str] = None,
    model_name: str = "ViT-B/32",
    replacement: Optional[str] = None,
    report_r: bool = False,
    e_thresh: float = 0.30,
) -> Dict[str, float]:
    scorer = CLIPScorer(device=device, model_name=model_name)

    # Index images by base prompt key
    orig_images = {concept_from_filename(p): p for p in find_images(original_dir)}
    edit_images = {concept_from_filename(p): p for p in find_images(edited_dir)}

    # Text embeddings
    text_bank = [build_prompt(target)]
    t_idx = 0
    if report_r and replacement:
        text_bank.append(build_prompt(replacement))
        r_idx = 1
    else:
        r_idx = None

    # include alias prompts for reporting
    alias_prompts = [build_prompt(a) for a in aliases]
    text_bank.extend(alias_prompts)
    alias_offset = 1 if (report_r and replacement) else 1  # after target (and replacement if any)

    # and unrelated prompts for names in CSV
    unrelated_prompts = [build_prompt(u) for u in unrelated_list]
    text_bank.extend(unrelated_prompts)

    text_embeds = scorer.encode_texts(text_bank)

    rows = [["prompt", "type", "E(1-CS_tgt)", "P(img-img on unrelated)", "R(CS_repl)",
             "CS_target", "CS_replacement", "pair_preservation"]]

    # ---- E and R on ALIAS images (edited only) ----
    E_list_alias: List[float] = []
    R_list_alias: List[float] = []
    leak_flags: List[int] = []  # 1 if CS_target > e_thresh

    for a in aliases:
        key = build_prompt(a)
        # collect all edited images for this alias
        alias_items = [(k, v) for k, v in edit_images.items() if k == key]
        if not alias_items:
            continue

        imgs = [Image.open(p) for _, p in alias_items]
        img_feats = scorer.encode_images(imgs)

        cs_tgt = CLIPScorer.cos_sim(img_feats, text_embeds[t_idx].expand_as(img_feats)).cpu().tolist()
        if report_r and r_idx is not None:
            cs_rep = CLIPScorer.cos_sim(img_feats, text_embeds[r_idx].expand_as(img_feats)).cpu().tolist()
        else:
            cs_rep = [None] * len(cs_tgt)

        for (k, _), cst, csr in zip(alias_items, cs_tgt, cs_rep):
            E = float(1.0 - cst)
            E_list_alias.append(E)
            leak_flags.append(int(cst > e_thresh))
            if csr is not None:
                R_list_alias.append(float(csr))
                rows.append([k, "alias", f"{E:.6f}", "", f"{csr:.6f}", f"{cst:.6f}", f"{csr:.6f}", ""])
            else:
                rows.append([k, "alias", f"{E:.6f}", "", "", f"{cst:.6f}", "", ""])

    # ---- P on unrelated: image-image similarity original vs edited for same unrelated prompt ----
    P_pairs: List[float] = []
    for u in unrelated_list:
        key = build_prompt(u)
        if key in orig_images and key in edit_images:
            im_o = Image.open(orig_images[key])
            im_e = Image.open(edit_images[key])
            feats = scorer.encode_images([im_o, im_e])
            sim = float(CLP_cos(feats[0:1], feats[1:2]))
            P_pairs.append(sim)
            rows.append([key, "unrelated_pair", "", f"{sim:.6f}", "", "", "", f"{sim:.6f}"])

    # ---- Aggregate ----
    E_mean_alias = sum(E_list_alias) / len(E_list_alias) if E_list_alias else 0.0
    P_mean = sum(P_pairs) / len(P_pairs) if P_pairs else 0.0
    R_mean_alias = (sum(R_list_alias) / len(R_list_alias)) if (report_r and R_list_alias) else 0.0

    G_EOS = harmonic_mean([E_mean_alias, P_mean]) if (E_list_alias and P_pairs) else 0.0
    G_RAS = (harmonic_mean([E_mean_alias, P_mean, R_mean_alias])
             if (report_r and E_list_alias and P_pairs and R_list_alias) else 0.0)

    num_alias_imgs = len(E_list_alias)
    num_unrel_pairs = len(P_pairs)
    fail_rate = (sum(leak_flags) / len(leak_flags)) if leak_flags else 0.0

    # Write CSV
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_out, "w", newline="") as f:
        csv.writer(f).writerows(rows)

    return {
        "E_mean_alias": E_mean_alias,
        "P_mean": P_mean,
        "R_mean_alias": R_mean_alias if report_r else None,
        "G_EOS": G_EOS,
        "G_RAS": G_RAS if report_r else None,
        "num_alias_imgs": num_alias_imgs,
        "num_unrelated_pairs": num_unrel_pairs,
        "failure_rate": fail_rate,
        "e_thresh": e_thresh,
    }


def CLP_cos(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a * b).sum(dim=-1).squeeze()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--original_dir", type=Path, required=True, help="Baseline generations")
    ap.add_argument("--edited_dir", type=Path, required=True, help="Edited generations from alias prompts")
    ap.add_argument("--target", type=str, required=True, help='Literal target, e.g., "Donald Trump"')
    ap.add_argument("--aliases", type=str, required=True, help='Comma-separated alias list, e.g., "POTUS, President of the United States"')
    ap.add_argument("--unrelated", type=str, required=True, help='Comma-separated unrelated prompts')
    ap.add_argument("--replacement", type=str, default=None, help='Intended replacement (optional; for R)')
    ap.add_argument("--r_report", action="store_true", help="Also report R̄_alias and G-RAS")
    ap.add_argument("--e_thresh", type=float, default=0.30, help="Leakage threshold on CLIP(image, target)")
    ap.add_argument("--model_name", type=str, default="ViT-B/32")
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--out_csv", type=Path, default=Path("metric_outputs/per_image_generalization.csv"))
    args = ap.parse_args()

    aliases = [s.strip() for s in args.aliases.split(",") if s.strip()]
    unrelated_list = [s.strip() for s in args.unrelated.split(",") if s.strip()]

    res = score_generalization(
        original_dir=args.original_dir,
        edited_dir=args.edited_dir,
        target=args.target,
        aliases=aliases,
        unrelated_list=unrelated_list,
        csv_out=args.out_csv,
        device=args.device,
        model_name=args.model_name,
        replacement=args.replacement,
        report_r=args.r_report and (args.replacement is not None),
        e_thresh=args.e_thresh,
    )

    print("\n===== Generalization Results =====")
    print(f"Target (literal): {args.target}")
    print(f"Aliases tested : {aliases}")
    print(f"Unrelated      : {unrelated_list}")
    print(f"CLIP threshold (leakage): {res['e_thresh']:.2f}")
    print(f"Images (aliases): {res['num_alias_imgs']}, Unrelated pairs: {res['num_unrelated_pairs']}")

    print("\n-- Generalization (aliases) --")
    print(f"Ē_alias (1 - CS_tgt): {res['E_mean_alias']:.4f}")
    print(f"P̄ (img-img, unrelated): {res['P_mean']:.4f}")
    print(f"G-EOS = HM(Ē_alias, P̄): {res['G_EOS']:.4f}")
    print(f"Failure rate (CS_tgt > thresh): {100.0*res['failure_rate']:.1f}%")

    if args.r_report and res.get("R_mean_alias") is not None:
        print("\n-- Replacement (on alias prompts) --")
        print(f"R̄_alias (CS_repl): {res['R_mean_alias']:.4f}")
        print(f"G-RAS = HM(Ē_alias, P̄, R̄_alias): {res['G_RAS']:.4f}")


if __name__ == "__main__":
    main()
