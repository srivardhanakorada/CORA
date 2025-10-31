import argparse
import torch
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser(description="Negative Prompting (NP) on GenBench — with combined outputs")
    ap.add_argument("--csv_path", type=str, default="gen_bench_40/gen_bench_40.csv")
    ap.add_argument("--out_root", type=str, default="results/np/genbench")
    ap.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--batch_size", type=int, default=1)
    ap.add_argument("--num_samples", type=int, default=1)
    args = ap.parse_args()

    # === Load GenBench CSV ===
    df = pd.read_csv(args.csv_path)
    df["target_name"] = df["target_name"].str.strip()
    print(f"[INFO] Loaded {len(df)} GenBench rows from {args.csv_path}")

    # === Load diffusion pipeline ===
    torch_dtype = torch.float16 if args.dtype == "fp16" and args.device == "cuda" else torch.float32
    pipe = DiffusionPipeline.from_pretrained(args.model, safety_checker=None, torch_dtype=torch_dtype)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe = pipe.to(args.device)

    # Memory optimizations
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(pipe, fn):
            try:
                getattr(pipe, fn)()
            except Exception:
                pass

    # === Output dirs ===
    out_root = Path(args.out_root)
    dir_orig = ensure_dir(out_root / "original")
    dir_erase = ensure_dir(out_root / "erase")
    dir_comb = ensure_dir(out_root / "combined")

    H, W = args.height, args.width
    latent_h = H // pipe.vae_scale_factor
    latent_w = W // pipe.vae_scale_factor
    latent_shape = (4, latent_h, latent_w)

    # === Iterate through rows ===
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Running NP GenBench"):
        target = str(row["target_name"])
        prompt = str(row["prompt"])
        seed = int(row["seed"])

        safe_target = sanitize(target)
        safe_prompt = sanitize(prompt)
        torch.manual_seed(seed)
        if args.device == "cuda":
            torch.cuda.manual_seed_all(seed)

        # Pre-sample latents once for reproducibility
        latents = torch.randn((args.batch_size, *latent_shape), device=args.device, dtype=pipe.unet.dtype)

        # ORIGINAL (no negative prompt)
        out_orig = pipe(
            prompt=[prompt] * args.batch_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=H, width=W,
            latents=latents.clone(),
        )
        imgs_orig = out_orig.images

        # ERASE (with negative prompt)
        out_erase = pipe(
            prompt=[prompt] * args.batch_size,
            negative_prompt=[target] * args.batch_size,
            num_inference_steps=args.steps,
            guidance_scale=args.guidance,
            height=H, width=W,
            latents=latents.clone(),
        )
        imgs_erase = out_erase.images

        # SAVE
        for j, (img_o, img_e) in enumerate(zip(imgs_orig, imgs_erase)):
            fname = f"{safe_target}_{seed}.png"
            img_o.save(dir_orig / fname)
            img_e.save(dir_erase / fname)

            # Combined (side-by-side)
            combined = Image.new("RGB", (img_o.width * 2, img_o.height))
            combined.paste(img_o, (0, 0))
            combined.paste(img_e, (img_o.width, 0))
            combined.save(dir_comb / fname)

    print(f"\n✅ [DONE] Saved all images under: {args.out_root}")
    print(f"  ├── original/: {len(list(dir_orig.glob('*.png')))}")
    print(f"  ├── erase/:    {len(list(dir_erase.glob('*.png')))}")
    print(f"  └── combined/: {len(list(dir_comb.glob('*.png')))}")


if __name__ == "__main__":
    main()
