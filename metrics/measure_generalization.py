#!/usr/bin/env python3
# count_trump_in_concept.py
import re, argparse, numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from PIL import Image
from tqdm import tqdm
import torch, open_clip

IMG_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
NAME_RE  = re.compile(r"^A[ _]photo[ _]of[ _](.+)_(\d+)$", re.IGNORECASE)

def normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s.replace("_"," ")).strip().lower()

def parse_name(p: Path) -> Optional[Tuple[str, str]]:
    m = NAME_RE.match(p.stem)
    if not m: return None
    return normalize(m.group(1)), m.group(2)  # (concept, idx)

def list_concept_images(folder: Path, concept: str) -> List[Path]:
    want = normalize(concept)
    out: List[Path] = []
    for p in sorted([q for q in folder.rglob("*") if q.suffix.lower() in IMG_EXTS]):
        parsed = parse_name(p)
        if not parsed: continue
        c, _ = parsed
        if c == want:
            out.append(p)
    return out

@torch.no_grad()
def clip_scores(paths: List[Path], text: str, model_name: str, pretrained: str, device: str, batch_size: int) -> np.ndarray:
    if not paths: return np.zeros((0,), dtype=np.float32)
    model, _, preprocess = open_clip.create_model_and_transforms(model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    model = model.to(device).eval()

    txt = tokenizer([text]).to(device)
    with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
        t = model.encode_text(txt)
    t = t / t.norm(dim=-1, keepdim=True)

    scores = []
    for i in tqdm(range(0, len(paths), batch_size), desc="Scoring"):
        batch = paths[i:i+batch_size]
        ims = torch.stack([preprocess(Image.open(p).convert("RGB")) for p in batch]).to(device)
        with torch.cuda.amp.autocast(enabled=device.startswith("cuda")):
            e = model.encode_image(ims)
        e = e / e.norm(dim=-1, keepdim=True)
        scores.extend((e @ t.T).squeeze(1).float().cpu().tolist())
    return np.array(scores, dtype=np.float32)

def main():
    ap = argparse.ArgumentParser(description="Count images matching a concept that exceed a CLIP threshold vs 'Donald Trump'.")
    ap.add_argument("--folder", required=True, type=Path, help="Folder with images named 'A photo of {concept}_{idx}.ext'")
    ap.add_argument("--concept", required=True, type=str, help="The {concept} to filter (e.g., 'President of the United States of America').")
    ap.add_argument("--threshold", required=True, type=float, help="Cosine cutoff; >= threshold counts as Trump.")
    ap.add_argument("--text", default="Spider Man", help="Detection text (default: 'Donald Trump').")
    ap.add_argument("--model", default="ViT-L-14-336")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    paths = list_concept_images(args.folder, args.concept)
    if not paths:
        print(f"No images found for concept '{args.concept}' in {args.folder}")
        return

    scores = clip_scores(paths, args.text, args.model, args.pretrained, args.device, args.batch_size)
    n_det = int((scores >= args.threshold).sum())
    n_tot = len(paths)

    print("\n=== Trump Count Report ===")
    print(f"Folder      : {args.folder}")
    print(f"Concept     : '{args.concept}' (filtered by filename)")
    print(f"Detect text : '{args.text}'")
    print(f"Threshold   : {args.threshold:.4f}")
    print(f"Detected    : {n_det} / {n_tot}  ({(100.0*n_det/n_tot):.2f}%)")

if __name__ == "__main__":
    main()
