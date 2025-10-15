#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, csv, re, sys
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NAME_RE   = re.compile(r"^A_photo_of_(.+)_(\d+)$")  # captures concept_with_underscores, idx

# ---------------- IO / parsing ----------------

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

def parse_key(stem: str) -> Optional[Tuple[str, str]]:
    """
    Returns (concept_key, concept_core) or None if not matching.
      stem='A_photo_of_Donald_Trump_12'  -> ('A_photo_of_Donald_Trump', 'Donald_Trump')
    """
    m = NAME_RE.match(stem)
    if not m: return None
    core = m.group(1)
    return (f"A_photo_of_{core}", core)

def group_by_concept(folder: Path) -> Dict[str, List[Path]]:
    buckets: Dict[str, List[Path]] = {}
    for p in list_images(folder):
        stem = p.stem
        res = parse_key(stem)
        if res is None: 
            # Silently ignore files that do not match the required pattern
            continue
        key, _ = res
        buckets.setdefault(key, []).append(p)
    return buckets

def normalize_concept_token(s: str) -> str:
    """
    Accepts 'Donald Trump' or 'Donald_Trump' or 'A_photo_of_Donald_Trump' and
    returns the filename concept key 'A_photo_of_Donald_Trump'.
    """
    s = s.strip()
    if s.startswith("A_photo_of_"):
        return s
    s = s.replace(" ", "_")
    return f"A_photo_of_{s}"

def parse_concepts_arg(arg: Optional[str]) -> Optional[List[str]]:
    if not arg:
        return None
    p = Path(arg)
    toks: List[str]
    if p.exists() and p.is_file():
        toks = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        toks = [t.strip() for t in arg.split(",") if t.strip()]
    return [normalize_concept_token(t) for t in toks]

# ---------------- Inception backbone ----------------

class ImageListDataset(Dataset):
    def __init__(self, paths: List[Path], transform: nn.Module):
        if len(paths) == 0:
            raise ValueError("Empty image list for dataset.")
        self.paths = paths
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

def inception_transform(weights: Optional[Inception_V3_Weights] = None) -> nn.Module:
    if weights is not None:
        return weights.transforms()
    return T.Compose([
        T.Resize(342, interpolation=T.InterpolationMode.BILINEAR),
        T.CenterCrop(299),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

class InceptionPool3(nn.Module):
    def __init__(self, device: str = "cpu"):
        super().__init__()
        try:
            weights = Inception_V3_Weights.IMAGENET1K_V1
        except Exception:
            weights = Inception_V3_Weights.DEFAULT
        net = inception_v3(weights=weights)
        net.eval()
        self.weights = weights
        self.features = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
            nn.MaxPool2d(3,2), net.Conv2d_3b_1x1, net.Conv2d_4a_3x3,
            nn.MaxPool2d(3,2), net.Mixed_5b, net.Mixed_5c, net.Mixed_5d,
            net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c, nn.AdaptiveAvgPool2d((1,1)),
        ).to(device)
        for p in self.features.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x):
        y = self.features(x)
        return y.view(y.size(0), -1)  # (N, 2048)

# ---------------- Stats / FID ----------------

@torch.no_grad()
def compute_stats(model: nn.Module, loader: DataLoader, device: str):
    feats = []
    for batch in loader:
        batch = batch.to(device, non_blocking=True)
        feats.append(model(batch).cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    return feats.mean(0), np.cov(feats, rowvar=False), feats.shape[0]

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

def compute_fid_between_lists(src_paths: List[Path], tgt_paths: List[Path],
                              model: InceptionPool3, device: str, bs: int, nw: int):
    tx = inception_transform(model.weights)
    ds_src = ImageListDataset(src_paths, tx)
    ds_tgt = ImageListDataset(tgt_paths, tx)
    pin = isinstance(device, str) and device.startswith("cuda")
    dl_src = DataLoader(ds_src, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    dl_tgt = DataLoader(ds_tgt, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    mu_s, s_s, n_s = compute_stats(model, dl_src, device)
    mu_t, s_t, n_t = compute_stats(model, dl_tgt, device)
    fid = frechet_distance(mu_s, s_s, mu_t, s_t)
    inv_fid = 1.0 / (1.0 + fid)
    return fid, inv_fid, int(n_s), int(n_t)

# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=Path, required=True,
                    help="Folder with edited images (no subfolders).")
    ap.add_argument("--tgt_root", type=Path, required=True,
                    help="Folder with reference/original images (no subfolders).")
    ap.add_argument("--concepts", type=str, default=None,
                    help="Optional: comma list or .txt of concepts. Accepts 'Donald Trump' or 'Donald_Trump' or 'A_photo_of_Donald_Trump'. "
                         "If omitted, uses intersection of concepts inferred from filenames in both roots.")
    ap.add_argument("--out_csv", type=Path, default=Path("metrics/fid_summary.csv"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)  # e.g., cuda:0
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionPool3(device=device)

    # Group files by concept key derived from filename
    src_groups = group_by_concept(args.src_root)
    tgt_groups = group_by_concept(args.tgt_root)

    # Resolve concept list
    if args.concepts:
        wanted = set(parse_concepts_arg(args.concepts) or [])
    else:
        wanted = set(src_groups.keys()) & set(tgt_groups.keys())

    concepts = sorted([k for k in wanted if (k in src_groups and k in tgt_groups)])
    if not concepts:
        print("[ERROR] No concepts to evaluate. "
              "Check filename pattern or pass --concepts.", file=sys.stderr)
        sys.exit(1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["concept_key","N_src","N_tgt","FID(SRC→TGT)","invFID"])

        for key in concepts:
            src_paths = src_groups.get(key, [])
            tgt_paths = tgt_groups.get(key, [])
            if not src_paths or not tgt_paths:
                print(f"[WARN] Skipping {key}: "
                      f"SRC n={len(src_paths)} TGT n={len(tgt_paths)}", file=sys.stderr)
                continue
            fid, inv, n_s, n_t = compute_fid_between_lists(src_paths, tgt_paths, model,
                                                           device=args.device or device,
                                                           bs=args.batch_size, nw=args.num_workers)
            w.writerow([key, n_s, n_t, f"{fid:.6f}", f"{inv:.6f}"])
            print(f"[FID] {key}: FID(SRC→TGT)={fid:.6f} invFID={inv:.6f} (Nsrc={n_s}, Ntgt={n_t})")

    print(f"\n[Done] Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()
