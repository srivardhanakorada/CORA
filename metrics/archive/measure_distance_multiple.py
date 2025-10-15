#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, re, sys
from pathlib import Path
from typing import List, Optional, Dict, Tuple
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NAME_RE  = re.compile(r"^A_photo_of_(.+)_(\d+)$")  # concept_with_underscores, idx

# ---------- IO / parsing ----------

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

def parse_key(stem: str) -> Optional[str]:
    """
    Returns the concept key 'A_photo_of_{Concept_With_Underscores}' or None if not matching.
    Example: 'A_photo_of_Donald_Trump_12' -> 'A_photo_of_Donald_Trump'
    """
    m = NAME_RE.match(stem)
    if not m: return None
    return f"A_photo_of_{m.group(1)}"

def normalize_concept_token(s: str) -> str:
    """
    Accepts 'Donald Trump' or 'Donald_Trump' or 'A_photo_of_Donald_Trump'
    -> 'A_photo_of_Donald_Trump'
    """
    s = s.strip()
    if s.startswith("A_photo_of_"): return s
    return f"A_photo_of_{s.replace(' ', '_')}"

def parse_concepts_arg(arg: Optional[str]) -> Optional[List[str]]:
    if not arg: return None
    p = Path(arg)
    if p.exists() and p.is_file():
        toks = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
    else:
        toks = [t.strip() for t in arg.split(",") if t.strip()]
    return [normalize_concept_token(t) for t in toks]

def collect_paths_by_concept(root: Path) -> Dict[str, List[Path]]:
    buckets: Dict[str, List[Path]] = {}
    for p in list_images(root):
        key = parse_key(p.stem)
        if key is None:  # ignore non-matching filenames
            continue
        buckets.setdefault(key, []).append(p)
    return buckets

# ---------- Inception backbone ----------

class ImageListDataset(Dataset):
    def __init__(self, paths: List[Path], transform: nn.Module):
        if len(paths) == 0:
            raise ValueError("Empty image list for dataset.")
        self.paths = paths; self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

def inception_transform(weights: Optional[Inception_V3_Weights]) -> nn.Module:
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
        net = inception_v3(weights=weights); net.eval()
        self.weights = weights
        self.features = nn.Sequential(
            net.Conv2d_1a_3x3, net.Conv2d_2a_3x3, net.Conv2d_2b_3x3,
            nn.MaxPool2d(3,2), net.Conv2d_3b_1x1, net.Conv2d_4a_3x3,
            nn.MaxPool2d(3,2), net.Mixed_5b, net.Mixed_5c, net.Mixed_5d,
            net.Mixed_6a, net.Mixed_6b, net.Mixed_6c, net.Mixed_6d, net.Mixed_6e,
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c, nn.AdaptiveAvgPool2d((1,1)),
        ).to(device)
        for p in self.features.parameters(): p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x):
        y = self.features(x)
        return y.view(y.size(0), -1)  # (N, 2048)

# ---------- Stats / FID ----------

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

def compute_fid_global(src_paths: List[Path], tgt_paths: List[Path],
                       model: InceptionPool3, device: str, bs: int, nw: int) -> Tuple[float,int,int]:
    tx = inception_transform(model.weights)
    ds_src = ImageListDataset(src_paths, tx)
    ds_tgt = ImageListDataset(tgt_paths, tx)
    pin = isinstance(device, str) and device.startswith("cuda")
    dl_src = DataLoader(ds_src, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    dl_tgt = DataLoader(ds_tgt, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    mu_s, s_s, n_s = compute_stats(model, dl_src, device)
    mu_t, s_t, n_t = compute_stats(model, dl_tgt, device)
    fid = frechet_distance(mu_s, s_s, mu_t, s_t)
    return fid, int(n_s), int(n_t)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src_root", type=Path, required=True, help="Folder with edited images (no subfolders).")
    ap.add_argument("--tgt_root", type=Path, required=True, help="Folder with reference/original images (no subfolders).")
    ap.add_argument("--concepts", type=str, default=None,
                    help="Optional: comma list or .txt of concepts to include. "
                         "Accepts 'Donald Trump' or 'Donald_Trump' or 'A_photo_of_Donald_Trump'. "
                         "If omitted, uses intersection of concepts inferred from filenames.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)  # e.g., cuda:0
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionPool3(device=device)

    # Bucket by concept (from filenames) in each root
    src_groups = collect_paths_by_concept(args.src_root)
    tgt_groups = collect_paths_by_concept(args.tgt_root)

    # Determine which concepts to include
    if args.concepts:
        wanted = set(parse_concepts_arg(args.concepts) or [])
    else:
        wanted = set(src_groups.keys()) & set(tgt_groups.keys())

    # Flatten image lists across all selected concepts
    src_all, tgt_all = [], []
    for key in sorted(wanted):
        s = src_groups.get(key, []); t = tgt_groups.get(key, [])
        if s and t:
            src_all.extend(s); tgt_all.extend(t)
        else:
            # silently skip concepts that don't appear in both sets
            pass

    if not src_all or not tgt_all:
        print("[ERROR] No overlapping images across SRC/TGT after filtering.", file=sys.stderr)
        sys.exit(1)

    fid, n_s, n_t = compute_fid_global(src_all, tgt_all, model, device, args.batch_size, args.num_workers)
    inv_fid = 1.0 / (1.0 + fid)

    # ---- Single-number output ----
    print(f"FID_global(SRC→TGT)={fid:.6f}  invFID={inv_fid:.6f}  (Nsrc={n_s}, Ntgt={n_t}, concepts={len(wanted)})")

if __name__ == "__main__":
    main()
