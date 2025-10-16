#!/usr/bin/env python3
"""
ESR / PSR / HM evaluation with a SINGLE CANONICAL TARGET.

Filenames contain the literal substring:
    "A photo of {concept}"
(case-insensitive match; spaces are significant exactly as typed).

Definitions (this script):
- Canonical target = the FIRST entry in --targets.
- Target membership: an image belongs to the targets pool if its filename contains
  "a photo of {t}" for ANY t in targets (canonical or paraphrase).
- For every TARGET image, we compute CS(img, "A photo of {canonical_target}.")  <-- NOTE: canonical only.
- For every NON-TARGET image of concept c, we compute CS(img, "A photo of {c}.")
- Reported metrics:
    ESR := 1 - mean_CS_over_targets
    PSR := mean_CS_over_non_targets
    HM  := harmonic mean of (ESR, PSR), computed on ESR/PSR clamped to [0,1] ONLY for HM.

Outputs:
- Console: ESR, PSR, HM.
- CSV: per-concept mean CS (for targets: CS vs canonical; for non-targets: CS vs their own),
       plus overall ESR/PSR/HM.

Dependencies:
    pip install torch torchvision pillow tqdm git+https://github.com/openai/CLIP.git
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import torch
import clip  # type: ignore
from PIL import Image
from tqdm import tqdm

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".gif"}


def read_concepts(spec: Optional[str]) -> List[str]:
    """Read concepts from JSON list, TXT (one per line), or comma-separated string; returns lowercased."""
    if not spec:
        return []
    p = spec.strip()
    if not p:
        return []
    if os.path.isfile(p):
        if p.lower().endswith(".json"):
            with open(p, "r", encoding="utf-8") as f:
                arr = json.load(f)
            return [str(x).strip().lower() for x in arr if str(x).strip()]
        else:
            with open(p, "r", encoding="utf-8") as f:
                arr = [line.strip() for line in f if line.strip()]
            return [x.lower() for x in arr]
    # inline comma-separated
    return [x.strip().lower() for x in p.split(",") if x.strip()]


def list_images(folder: str) -> List[str]:
    out = []
    for root, _, files in os.walk(folder):
        for name in files:
            if os.path.splitext(name)[1].lower() in IMAGE_EXTS:
                out.append(os.path.join(root, name))
    return out


def any_match_target(fn_noext_lower: str, targets_lower: List[str]) -> bool:
    """Return True if filename contains 'a photo of {t}' for ANY target t (case-insensitive, space-sensitive)."""
    for t in targets_lower:
        if f"a photo of {t}" in fn_noext_lower:
            return True
    return False


def match_exact_concept(fn_noext_lower: str, concepts_lower: List[str]) -> Optional[str]:
    """
    For non-target concepts: return the concept whose 'a photo of {concept}'
    occurs in the filename. If multiple match, choose the longest.
    """
    matches = [c for c in concepts_lower if f"a photo of {c}" in fn_noext_lower]
    if not matches:
        return None
    matches.sort(key=len, reverse=True)
    return matches[0]


@torch.no_grad()
def compute_metrics(
    device: str,
    model_name: str,
    folder: str,
    target_concepts: List[str],
    non_target_concepts: List[str],
    csv_out: Optional[str] = None,
) -> Dict[str, object]:
    # Sanity: need at least one target to define canonical
    if not target_concepts:
        raise ValueError("At least one target concept is required (canonical target is the first).")

    # Canonical target
    canonical_target = target_concepts[0]
    targets_all = target_concepts  # includes paraphrases

    # Load CLIP
    model, preprocess = clip.load(model_name, device=device, jit=False)
    model.eval()

    # Prepare text prompts:
    # - Canonical target text embedding
    # - Non-targets text embeddings (each to its own prompt)
    prompt_texts: List[Tuple[str, str]] = []
    prompt_texts.append((canonical_target, f"A photo of {canonical_target}."))  # key 'canonical'
    for c in non_target_concepts:
        prompt_texts.append((c, f"A photo of {c}."))

    tokenized = clip.tokenize([p for _, p in prompt_texts]).to(device)
    text_features = model.encode_text(tokenized).float()
    text_features = text_features / text_features.norm(dim=-1, keepdim=True)

    # Map:
    #   "canonical_target" -> emb (under key canonical_target)
    #   each non-target concept -> its own emb
    concept2text = {name: text_features[i] for i, (name, _) in enumerate(prompt_texts)}

    # Pools & per-concept collectors
    # For targets: we will store per-paraphrase concept rows too, but CS is vs CANONICAL emb.
    all_concepts_list = targets_all + non_target_concepts
    per_concept_cs: Dict[str, List[float]] = {c: [] for c in all_concepts_list}
    targets_pool: List[float] = []
    nontargets_pool: List[float] = []

    img_paths = list_images(folder)
    skipped = 0
    multimatch_notes = 0

    # Lower lists for matching
    targets_lower = targets_all
    non_targets_lower = non_target_concepts  # used to find which non-target a file belongs to

    for p in tqdm(img_paths, desc="Scoring images with CLIP"):
        fn = os.path.basename(p)
        fn_noext = os.path.splitext(fn)[0]
        fn_lower = fn_noext.lower()

        try:
            img = Image.open(p).convert("RGB")
        except Exception as e:
            print(f"[WARN] Could not open {p}: {e}", file=sys.stderr)
            continue

        img_in = preprocess(img).unsqueeze(0).to(device)
        img_feat = model.encode_image(img_in).float()
        img_feat = img_feat / img_feat.norm(dim=-1, keepdim=True)

        # 1) Target membership (ANY target/paraphrase match)
        if any_match_target(fn_lower, targets_lower):
            # compute CS vs CANONICAL target
            can_text = concept2text[canonical_target]
            cs = (img_feat @ can_text.unsqueeze(1)).squeeze().item()
            # For bookkeeping: try to attribute to the longest matching target token for per-concept row
            matches = [t for t in targets_lower if f"a photo of {t}" in fn_lower]
            if len(matches) > 1:
                multimatch_notes += 1
                matches.sort(key=len, reverse=True)
            label = matches[0] if matches else canonical_target
            per_concept_cs[label].append(float(cs))
            targets_pool.append(float(cs))
            continue

        # 2) Otherwise, try non-target assignment for per-concept & PSR
        non_match = match_exact_concept(fn_lower, non_targets_lower)
        if non_match is not None:
            nt_text = concept2text[non_match]
            cs = (img_feat @ nt_text.unsqueeze(1)).squeeze().item()
            per_concept_cs[non_match].append(float(cs))
            nontargets_pool.append(float(cs))
        else:
            # unknown -> skip
            skipped += 1

    if multimatch_notes > 0:
        print(f"[INFO] {multimatch_notes} files matched multiple target phrases; chose the longest for per-concept logging.")
    if skipped > 0:
        print(f"[INFO] Skipped {skipped} files that matched neither targets nor non-targets.")

    # Aggregates
    def mean(lst: List[float]) -> float:
        return float(sum(lst) / len(lst)) if lst else float("nan")

    mean_cs_targets = mean(targets_pool)
    mean_cs_nontargets = mean(nontargets_pool)

    ESR = (1.0 - mean_cs_targets) if targets_pool else float("nan")
    PSR = mean_cs_nontargets if nontargets_pool else float("nan")

    # HM on clamped ESR/PSR
    def clamp01(x: float) -> float:
        if x != x:  # NaN
            return x
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def harmonic_mean(a: float, b: float) -> float:
        if (a != a) or (b != b):
            return float("nan")
        ac, bc = clamp01(a), clamp01(b)
        s = ac + bc
        return (2.0 * ac * bc / s) if s > 0 else float("nan")

    HM = harmonic_mean(ESR, PSR)

    # Per-concept means (NOTE: for targets these are vs CANONICAL)
    per_concept_mean = {c: mean(per_concept_cs[c]) for c in all_concepts_list}

    # Save CSV
    if csv_out:
        os.makedirs(os.path.dirname(csv_out) or ".", exist_ok=True)
        with open(csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["concept",
                        "group",
                        "num_images",
                        "mean_CS",
                        "note"])
            for c in targets_all:
                w.writerow([c, "target", len(per_concept_cs[c]), per_concept_mean[c],
                            f"CS vs CANONICAL '{canonical_target}'"])
            for c in non_target_concepts:
                w.writerow([c, "non_target", len(per_concept_cs[c]), per_concept_mean[c],
                            "CS vs own concept"])
            w.writerow([])
            w.writerow(["ESR (1 - mean_CS_targets)", "", len(targets_pool), ESR, f"canonical: {canonical_target}"])
            w.writerow(["PSR (mean_CS_non_targets)", "", len(nontargets_pool), PSR, ""])
            w.writerow(["HM(ESR, PSR) [ESR/PSR clamped to 0..1 for HM only]", "", "", HM, ""])

    return {
        "ESR": {"value": ESR, "num_images": len(targets_pool)},
        "PSR": {"value": PSR, "num_images": len(nontargets_pool)},
        "HM": HM,
        "per_concept_mean_CS": per_concept_mean,
        "canonical_target": canonical_target,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True, help="Folder containing generated images")
    ap.add_argument("--targets", required=True, help="Target concepts (first is canonical): JSON/TXT or comma-separated")
    ap.add_argument("--non_targets", required=True, help="Non-target concepts: JSON/TXT or comma-separated")
    ap.add_argument("--model", default="ViT-L/14", help="CLIP model (e.g., ViT-B/32, ViT-L/14)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--csv_out", default="esr_psr_metrics.csv")
    args = ap.parse_args()

    targets = read_concepts(args.targets)
    non_targets = read_concepts(args.non_targets)

    if not targets:
        print("[ERROR] --targets is empty; need at least one target to define canonical.", file=sys.stderr)
        sys.exit(2)

    overlap = set(targets).intersection(non_targets)
    if overlap:
        print(f"[WARN] Concepts in both target and non-target lists: {sorted(overlap)}", file=sys.stderr)

    res = compute_metrics(
        device=args.device,
        model_name=args.model,
        folder=args.folder,
        target_concepts=targets,
        non_target_concepts=non_targets,
        csv_out=args.csv_out,
    )

    print("\n=== Results ===")
    print(f"Canonical target: {res['canonical_target']}")
    print(f"ESR = 1 - mean_CS(targets vs canonical): {res['ESR']['value']:.4f} over {res['ESR']['num_images']} images")
    print(f"PSR = mean_CS(non-targets vs own):        {res['PSR']['value']:.4f} over {res['PSR']['num_images']} images")
    print(f"HM(ESR, PSR):                             {res['HM']:.4f}")


if __name__ == "__main__":
    main()
