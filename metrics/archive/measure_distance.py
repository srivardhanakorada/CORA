#!/usr/bin/env python3
import argparse, csv, sys
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from torchvision.models import inception_v3, Inception_V3_Weights

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

def list_images(folder: Path) -> List[Path]:
    return sorted([p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS])

class ImageFolderDataset(Dataset):
    def __init__(self, folder: Path, transform: nn.Module):
        self.paths = list_images(folder)
        if len(self.paths) == 0:
            raise ValueError(f"No images found in {folder}")
        self.transform = transform
    def __len__(self): return len(self.paths)
    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        return self.transform(img)

def inception_transform(weights: Optional[Inception_V3_Weights] = None) -> nn.Module:
    # Prefer model-recommended transforms; otherwise fall back.
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
        # Do NOT override aux_logits when using pretrained weights.
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
            net.Mixed_7a, net.Mixed_7b, net.Mixed_7c,
            nn.AdaptiveAvgPool2d((1,1)),
        ).to(device)
        for p in self.features.parameters():
            p.requires_grad_(False)
    @torch.no_grad()
    def forward(self, x):
        y = self.features(x)
        return y.view(y.size(0), -1)  # (N, 2048)

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

def compute_fid_between_folders(source: Path, target: Path, model: InceptionPool3,
                                device: str, bs: int, nw: int):
    """FID(source, target) with target as the reference distribution."""
    tx = inception_transform(model.weights)
    ds_src = ImageFolderDataset(source, tx)
    ds_tgt = ImageFolderDataset(target, tx)
    pin = isinstance(device, str) and device.startswith("cuda")
    dl_src = DataLoader(ds_src, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    dl_tgt = DataLoader(ds_tgt, batch_size=bs, shuffle=False, num_workers=nw, pin_memory=pin)
    mu_s, s_s, n_s = compute_stats(model, dl_src, device)
    mu_t, s_t, n_t = compute_stats(model, dl_tgt, device)
    fid = frechet_distance(mu_s, s_s, mu_t, s_t)
    inv_fid = 1.0 / (1.0 + fid)
    return fid, inv_fid, int(n_s), int(n_t)

def safe_pair(name, src, tgt) -> Optional[Tuple[Path, Path]]:
    if src.exists() and tgt.exists() and list_images(src) and list_images(tgt):
        return (src, tgt)
    missing = []
    if not src.exists() or not list_images(src): missing.append(f"SRC missing/empty: {src}")
    if not tgt.exists() or not list_images(tgt): missing.append(f"TGT missing/empty: {tgt}")
    print(f"[WARN] Skipping {name}: " + " | ".join(missing), file=sys.stderr)
    return None

def list_subjects(root: Path) -> List[str]:
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cora_root", type=Path, default=Path("outputs_cora_anc"))
    ap.add_argument("--adavd_root", type=Path, default=Path("outputs_adavd"))
    ap.add_argument("--out_csv", type=Path, default=Path("metrics/results/fid_summary.csv"))
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--device", type=str, default=None)  # e.g., cuda:0
    args = ap.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    model = InceptionPool3(device=device)

    cora_subjects  = set(list_subjects(args.cora_root))
    adavd_subjects = set(list_subjects(args.adavd_root))
    subjects = sorted(cora_subjects | adavd_subjects)

    if not subjects:
        print(f"[ERROR] No subjects found under '{args.cora_root}' or '{args.adavd_root}'.", file=sys.stderr)
        sys.exit(1)

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.writer(f)
        # Note: columns explicitly label SOURCE and TARGET per your spec.
        w.writerow(["method","subject","mode","SOURCE","TARGET","N_source","N_target","FID(SRC→TGT)","invFID"])

        for subj in subjects:
            # ----- CORA: modes neut and int -----
            if subj in cora_subjects:
                for mode in ("neut","int"):
                    # TARGET = original, SOURCE = erase
                    tgt = args.cora_root / subj / mode / "original"
                    src = args.cora_root / subj / mode / "erase"
                    pair = safe_pair(f"CORA/{subj}/{mode}", src, tgt)
                    if pair:
                        fid, inv, n_s, n_t = compute_fid_between_folders(src, tgt, model, device, args.batch_size, args.num_workers)
                        w.writerow(["CORA", subj, mode, str(src), str(tgt), n_s, n_t, f"{fid:.6f}", f"{inv:.6f}"])
                        print(f"[CORA] {subj}/{mode}: FID(SRC→TGT)={fid:.6f} invFID={inv:.6f}")

            # ----- AdaVD: no modes -----
            if subj in adavd_subjects:
                # TARGET = original, SOURCE = retain
                tgt = args.adavd_root / subj / "original"
                src = args.adavd_root / subj / "retain"
                pair = safe_pair(f"AdaVD/{subj}", src, tgt)
                if pair:
                    fid, inv, n_s, n_t = compute_fid_between_folders(src, tgt, model, device, args.batch_size, args.num_workers)
                    w.writerow(["AdaVD", subj, "-", str(src), str(tgt), n_s, n_t, f"{fid:.6f}", f"{inv:.6f}"])
                    print(f"[AdaVD] {subj}: FID(SRC→TGT)={fid:.6f} invFID={inv:.6f}")

    print(f"\n[Done] Wrote: {args.out_csv}")

if __name__ == "__main__":
    main()
