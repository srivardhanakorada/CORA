#!/usr/bin/env python3
"""
Batch concept-filtered FID over CORA and AdaVD layouts.

Filename convention matched (case-insensitive):
  "A photo of {concept}_{i}.<ext>"
  "A photo of an {concept}_{i}.<ext>"   # "an " is optional
where i ∈ [0, 9999] (1–4 digits), ext ∈ {png,jpg,jpeg,webp,bmp}

Folder conventions:

CORA:
  <cora_root>/<subject>/<mode>/{original, erase}
    modes: "neut", "int"
    source = "erase", target = "original"

AdaVD:
  <adavd_root>/<subject>/{retain, original}
    source = "retain", target = "original"

For each (source,target) pair, we:
  1) Discover concepts present in each folder (by filename).
  2) Compute FID only on an intersection of concepts (optionally whitelisted).
  3) Append rows to a CSV.

Usage:
  python -W ignore metrics/fid_concept_batch.py \
    --cora_root outputs_cora_anc \
    --adavd_root outputs_adavd \
    --out_csv metrics/results/fid_concept_summary.csv \
    --device cuda:0 --batch_size 64 --num_workers 4

Optional whitelist:
  --whitelist_file metrics/concepts.txt   # one concept per line
"""

import argparse, csv, re, sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple, Dict

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ---------------- I/O utils ----------------

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

# Match stems like:
#   A photo of <concept>_<digits>
#   A photo of an <concept>_<digits>
STEM_RX = re.compile(r"^(?i)A photo of (an )?(?P<concept>.+)_(?P<idx>[0-9]{1,4})$")

def stem_of(p: Path) -> str:
    return p.stem

def discover_concepts(folder: Path) -> Set[str]:
    if not folder.exists():
        return set()
    concepts: Set[str] = set()
    for p in list_images(folder):
        m = STEM_RX.match(stem_of(p))
        if m:
            concepts.add(m.group("concept"))
    return concepts

def count_concept_matches(folder: Path, concept: str) -> int:
    if not folder.exists():
        return 0
    esc = re.escape(concept)
    rx = re.compile(rf"^(?i)A photo of (an )?{esc}_[0-9]{{1,4}}$")
    n = 0
    for p in list_images(folder):
        if rx.match(stem_of(p)):
            n += 1
    return n

def filter_paths_by_concept(paths: Sequence[Path], concept: str) -> List[Path]:
    esc = re.escape(concept)
    rx = re.compile(rf"^(?i)A photo of (an )?{esc}_[0-9]{{1,4}}$")
    return [p for p in paths if rx.match(stem_of(p))]

# ---------------- Dataset ----------------

class ImageFolderConcept(Dataset):
    def __init__(self, folder: Path, transform: nn.Module, concept: str):
        all_paths = list_images(folder)
        self.paths = filter_paths_by_concept(all_paths, concept)
        if len(self.paths) == 0:
            raise ValueError(f"No matching files for concept '{concept}' in {folder}")
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

# ---------------- Model & FID ----------------

def inception_transform(weights: Optional[Inception_V3_Weights] = None) -> nn.Module:
    if weights is not None:
        return weights.transforms()
    return T.Compose([
        T.Resize(342, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])

class InceptionPool3(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            weights = Inception_V3_Weights.IMAGENET1K_V1
        except Exception:
            weights = Inception_V3_Weights.DEFAULT
        net = inception_v3(weights=weights)  # keep aux_logits as in weights
        net.eval()
        self.weights = weights
        self.features = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
            nn.MaxPool2d(3,2), net.Conv2d_3b_1x1, net.Conv2d_4a_3x3,
            nn.MaxPool2d(3,2), net.Mixed_5b, net.Mixed_5c, net.Mixed_5d,
            net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1)),
        ).to(device)
        for p in self.features.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.features(x)
        return y.view(y.size(0), -1)  # (N, 2048)

@torch.no_grad()
def compute_stats(model: nn.Module, loader: DataLoader, device: str):
    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        feats.append(model(batch).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    mu = feats.mean(0)
    sigma = np.cov(feats, rowvar=False)
    return mu, sigma, feats.shape[0]

def _sqrtm_psd(mat: np.ndarray) -> np.ndarray:
    mat = (mat + mat.T) * 0.5
    vals, vecs = np.linalg.eigh(mat)
    vals = np.clip(vals, 0.0, None)
    return (vecs * np.sqrt(vals)) @ vecs.T

def frechet_distance(mu1, s1, mu2, s2) -> float:
    diff = mu1 - mu2
    sqrt_s1 = _sqrtm_psd(s1)
    mid = sqrt_s1 @ s2 @ sqrt_s1
    sqrt_mid = _sqrtm_psd(mid)
    fid = float(diff.dot(diff) + np.trace(s1) + np.trace(s2) - 2.0*np.trace(sqrt_mid))
    return max(fid, 0.0)

def compute_fid_for_concept(source: Path, target: Path, concept: str,
                            model: InceptionPool3, device: str,
                            bs: int, nw: int) -> Tuple[float, float, int, int]:
    tx = inception_transform(model.weights)
    ds_src = ImageFolderConcept(source, tx, concept)
    ds_tgt = ImageFolderConcept(target, tx, concept)
    pin = isinstance(device, str) and device.startswith("cuda")
    dl_src = DataLoader(ds_src, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    dl_tgt = DataLoader(ds_tgt, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    mu_s, s_s, n_s = compute_stats(model, dl_src, device)
    mu_t, s_t, n_t = compute_stats(model, dl_tgt, device)
    fid = frechet_distance(mu_s, s_s, mu_t, s_t)
    inv_fid = 1.0 / (1.0 + fid)
    return fid, inv_fid, int(n_s), int(n_t)

# ---------------- Batch traversal ----------------

def list_subjects(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def resolve_cora_pair(cora_root: Path, subject: str, mode: str) -> Optional[Tuple[Path, Path]]:
    base = cora_root / subject / mode
    src = base / "erase"
    tgt = base / "original"
    if src.exists() and tgt.exists():
        return src, tgt
    return None

def resolve_adavd_pair(adavd_root: Path, subject: str) -> Optional[Tuple[Path, Path]]:
    base = adavd_root / subject
    src = base / "retain"
    tgt = base / "original"
    if src.exists() and tgt.exists():
        return src, tgt
    return None

def load_whitelist(path: Optional[Path]) -> Optional[Set[str]]:
    if path is None:
        return None
    if not path.exists():
        print(f"[WARN] whitelist file not found: {path}", file=sys.stderr)
        return None
    with open(path, "r") as f:
        items = [line.strip() for line in f if line.strip()]
    return set(items) if items else None

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cora_root", type=Path, default=Path("outputs_cora_anc"))
    ap.add_argument("--adavd_root", type=Path, default=Path("outputs_adavd"))
    ap.add_argument("--out_csv", type=Path, default=Path("metrics/results/fid_concept_summary.csv"))
    ap.add_argument("--whitelist_file", type=Path, default=None, help="Optional file: one concept per line")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)  # e.g., cuda:0
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionPool3(device=device)

    whitelist = load_whitelist(args.whitelist_file)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["method","subject","mode","concept","SOURCE","TARGET","N_source","N_target","FID(SRC→TGT)","invFID"])

        # ----- CORA -----
        for subj in list_subjects(args.cora_root):
            for mode in ("neut","int"):
                pair = resolve_cora_pair(args.cora_root, subj, mode)
                if pair is None:
                    print(f"[WARN] Skipping CORA/{subj}/{mode}: missing erase/original", file=sys.stderr)
                    continue
                src, tgt = pair
                # discover & intersect concepts
                src_concepts = discover_concepts(src)
                tgt_concepts = discover_concepts(tgt)
                common = src_concepts & tgt_concepts
                if whitelist is not None:
                    common = {c for c in common if c in whitelist}
                if not common:
                    print(f"[WARN] CORA/{subj}/{mode}: no overlapping concepts", file=sys.stderr)
                    continue
                for concept in sorted(common):
                    try:
                        fid, inv, n_s, n_t = compute_fid_for_concept(src, tgt, concept, model, device, args.batch_size, args.num_workers)
                    except ValueError as e:
                        print(f"[WARN] CORA/{subj}/{mode} [{concept}]: {e}", file=sys.stderr)
                        continue
                    w.writerow(["CORA", subj, mode, concept, str(src), str(tgt), n_s, n_t, f"{fid:.6f}", f"{inv:.6f}"])
                    print(f"[CORA] {subj}/{mode} [{concept}] FID={fid:.6f} invFID={inv:.6f} Nsrc={n_s} Ntgt={n_t}")

        # ----- AdaVD -----
        for subj in list_subjects(args.adavd_root):
            pair = resolve_adavd_pair(args.adavd_root, subj)
            if pair is None:
                print(f"[WARN] Skipping AdaVD/{subj}: missing retain/original", file=sys.stderr)
                continue
            src, tgt = pair
            src_concepts = discover_concepts(src)
            tgt_concepts = discover_concepts(tgt)
            common = src_concepts & tgt_concepts
            if whitelist is not None:
                common = {c for c in common if c in whitelist}
            if not common:
                print(f"[WARN] AdaVD/{subj}: no overlapping concepts", file=sys.stderr)
                continue
            for concept in sorted(common):
                try:
                    fid, inv, n_s, n_t = compute_fid_for_concept(src, tgt, concept, model, device, args.batch_size, args.num_workers)
                except ValueError as e:
                    print(f"[WARN] AdaVD/{subj} [{concept}]: {e}", file=sys.stderr)
                    continue
                w.writerow(["AdaVD", subj, "-", concept, str(src), str(tgt), n_s, n_t, f"{fid:.6f}", f"{inv:.6f}"])
                print(f"[AdaVD] {subj} [{concept}] FID={fid:.6f} invFID={inv:.6f} Nsrc={n_s} Ntgt={n_t}")

    print(f"\n[Done] Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()
