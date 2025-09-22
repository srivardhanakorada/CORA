import argparse, csv, json, sys
from pathlib import Path
from typing import List, Optional
from PIL import Image, UnidentifiedImageError  # type:ignore
import torch  # type:ignore
from transformers import CLIPModel, CLIPProcessor  # type:ignore

VALID_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}

# ---------------------------- utils ----------------------------
def canon(s: str) -> str:
    return " ".join(s.lower().replace("_", " ").split())

def list_images(d: Path) -> List[Path]:
    if not d.exists():
        return []
    out = []
    for p in sorted(d.rglob("*")):
        if p.is_file() and p.suffix.lower() in VALID_EXTS:
            out.append(p)
    return out

def load_rgb(p: Path) -> Image.Image:
    with Image.open(p) as im:
        return im.convert("RGB")

def batched(iterable, n):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

def harmonic_mean(acc_e: float, acc_s: float) -> float:
    # H = 2 / ((1-Acc_e)^-1 + (Acc_s)^-1)
    eps = 1e-12
    eff = max(0.0, min(1.0, 1.0 - acc_e))
    spe = max(0.0, min(1.0, acc_s))
    return 2.0 / ((1.0 / (eff + eps)) + (1.0 / (spe + eps)))

def label_from_path(p: Path) -> str:
    """If the leaf folder is 'erase' or 'retain', use the parent name as label."""
    leaf = p.name.lower()
    return p.parent.name if leaf in {"erase", "retain"} else p.name

# ---------------------- CLIP single-template --------------------
class CLIPSingleTemplate:
    """
    Zero-shot classifier over given candidate names using ONLY:
        template = "a photo of {name}"
    Supports optional rejection threshold tau.
    """
    def __init__(self, candidates: List[str],
                 model_name: str = "openai/clip-vit-large-patch14",
                 device: Optional[str] = None,
                 tau: Optional[float] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tau = tau  # cosine similarity threshold for rejection
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.proc = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()

        self.disp_names = candidates[:]   # report strings
        prompts = [f"a photo of {name.replace('_', ' ')}" for name in self.disp_names]
        with torch.no_grad():
            toks = self.proc(text=prompts, return_tensors="pt", padding=True).to(self.device)
            T = self.model.get_text_features(**toks)          # (K, D)
            T = T / T.norm(dim=-1, keepdim=True)
        self.text = T                                         # (K, D)

    @torch.no_grad()
    def predict_top1_batch(self, pil_images: List[Image.Image]) -> List[str]:
        pixel = self.proc(images=pil_images, return_tensors="pt").to(self.device)
        V = self.model.get_image_features(**pixel)            # (B, D)
        V = V / V.norm(dim=-1, keepdim=True)
        sims = V @ self.text.T                                # (B, K)
        vals, idx = sims.max(dim=-1)                          # (B,)
        preds = []
        for v, i in zip(vals.tolist(), idx.tolist()):
            if self.tau is not None and float(v) < float(self.tau):
                preds.append("__other__")
            else:
                preds.append(self.disp_names[i])
        return preds

# --------------------------- scoring ----------------------------
def score_single_target_cs(
    target_dir: Path,
    retain_dirs: List[Path],
    model_name: str = "openai/clip-vit-large-patch14",
    batch_size: int = 16,
    device: Optional[str] = None,
    out_csv: Optional[Path] = None,
    out_json: Optional[Path] = None,
    canonical_target: Optional[str] = None,
    tau: Optional[float] = None
):
    # Resolve labels (handles .../<Name>/erase|retain). Allow canonical override.
    target_name_path = label_from_path(target_dir)
    target_name = canonical_target if canonical_target else target_name_path
    retain_names = [label_from_path(p) for p in retain_dirs]

    # Warn if any label is 'erase'/'retain'
    for nm in [target_name, *retain_names]:
        if nm.lower() in {"erase", "retain"}:
            print(f"[WARN] Resolved label is '{nm}'. Check your paths.", file=sys.stderr)

    # Gather paths
    erase_imgs = list_images(target_dir)
    if not erase_imgs:
        raise FileNotFoundError(f"No images found in {target_dir}")

    retain_imgs, retain_labels = [], []
    for rd, rn in zip(retain_dirs, retain_names):
        imgs = list_images(rd)
        if not imgs:
            print(f"WARNING: no images in retain dir {rd}", file=sys.stderr)
        for p in imgs:
            retain_imgs.append(p)
            retain_labels.append(rn)
    if not retain_imgs:
        raise FileNotFoundError("No retain images found in any of the provided retain_dirs")

    # Build detector with only the given names (no extra templates)
    detector = CLIPSingleTemplate([target_name] + retain_names,
                                  model_name=model_name,
                                  device=device,
                                  tau=tau)

    # Helper to load in batches safely
    def predict_paths(paths: List[Path]) -> List[str]:
        preds: List[str] = []
        for chunk in batched(paths, batch_size):
            imgs = []
            for p in chunk:
                try:
                    imgs.append(load_rgb(p))
                except (UnidentifiedImageError, OSError) as e:
                    print(f"Skipping unreadable image: {p} ({e})", file=sys.stderr)
                    imgs.append(Image.new("RGB", (224, 224), (0, 0, 0)))  # placeholder
            preds.extend(detector.predict_top1_batch(imgs))
        return preds

    # Predictions
    preds_erase  = predict_paths(erase_imgs)
    preds_retain = predict_paths(retain_imgs)

    # Metrics (note: '__other__' never equals any label -> counts as "not target" / "incorrect")
    acc_e = sum(canon(p) == canon(target_name) for p in preds_erase) / len(preds_erase)
    acc_s = sum(canon(p) == canon(gt)         for p, gt in zip(preds_retain, retain_labels)) / len(preds_retain)
    H = harmonic_mean(acc_e, acc_s)

    # Package results
    res = {
        "template": "a photo of {}",
        "model": model_name,
        "target": target_name.replace("_", " "),
        "retain_set": [x.replace("_", " ") for x in retain_names],
        "N_erase": len(preds_erase),
        "N_retain": len(preds_retain),
        "Acc_e": acc_e,  # lower is better
        "Acc_s": acc_s,  # higher is better
        "H": H,
        "tau": tau,
        "canonical_target_used": bool(canonical_target)
    }

    if out_csv:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            fields = ["template","model","target","retain_set","N_erase","N_retain","Acc_e","Acc_s","H","tau","canonical_target_used"]
            w = csv.DictWriter(f, fieldnames=fields)
            row = {**res, "retain_set": ";".join(res["retain_set"])}
            w.writeheader(); w.writerow(row)

    if out_json:
        out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(res, f, indent=2)

    return res

# ----------------------------- CLI ------------------------------
def parse_args():
    ap = argparse.ArgumentParser(
        description="Single-target erasure scoring with CLIP CS using ONLY 'a photo of {}'."
    )
    ap.add_argument("--target_dir", type=str, required=True,
                    help="Folder of images for the ERASED target (can be an alias/generalization folder).")
    ap.add_argument("--retain_dirs", type=str, nargs="+", required=True,
                    help="Folders for NON-TARGET identities (e.g., .../Elon Musk/retain ...)")
    ap.add_argument("--model", type=str, default="openai/clip-vit-large-patch14",
                    help="CLIP model name (e.g., openai/clip-vit-base-patch32 for lighter GPU)")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda", "mps"],
                    help="Force device. Omit to auto-select.")
    ap.add_argument("--out_csv", type=str, default="single_target_cs.csv")
    ap.add_argument("--out_json", type=str, default="single_target_cs.json")
    ap.add_argument("--canonical_target", type=str, default="Spider Man",
                    help='Override the derived target label (e.g., "Donald Trump").')
    ap.add_argument("--tau", type=float, default=None,
                    help="Optional CS threshold for rejection to '__other__'. Typical range 0.25–0.35 for ViT-L/14.")
    return ap.parse_args()

def main():
    args = parse_args()
    res = score_single_target_cs(
        target_dir=Path(args.target_dir),
        retain_dirs=[Path(p) for p in args.retain_dirs],
        model_name=args.model,
        batch_size=args.batch_size,
        device=args.device,
        out_csv=Path(args.out_csv),
        out_json=Path(args.out_json),
        canonical_target=args.canonical_target,
        tau=args.tau,
    )

    print("\n=== Single-Target CS Metric (template: 'a photo of {}') ===")
    print(f"Resolved target label : {res['target']} (canonical={res['canonical_target_used']})")
    print(f"Resolved retain labels: {', '.join(res['retain_set'])}")
    print(f"N_erase        : {res['N_erase']}")
    print(f"N_retain       : {res['N_retain']}")
    print(f"Acc_e (lower)  : {res['Acc_e']:.3f}")
    print(f"Acc_s (higher) : {res['Acc_s']:.3f}")
    print(f"H (higher)     : {res['H']:.3f}")
    if res['tau'] is not None:
        print(f"Rejection tau  : {res['tau']}")
        print("Note: '__other__' counts as NOT target for erase, and INCORRECT for retain.")
    print()

if __name__ == "__main__":
    main()
