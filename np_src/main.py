# ======= NP (CORA-style seeding) =======
import argparse
from pathlib import Path
from typing import List
import torch
from PIL import Image
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p

def sanitize(s: str) -> str:
    return "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in s)

def chunked(xs: List[int], n: int):
    for i in range(0, len(xs), n):
        yield i, xs[i:i+n]

def main():
    ap = argparse.ArgumentParser(description="Batched generation with negative prompt using 'A photo of {}' (CORA-style seeding)")
    ap.add_argument("--out_root", type=str, default="outputs_np")
    ap.add_argument("--target_root", type=str, default="Donald Trump")
    ap.add_argument("--names", type=str, nargs="+",
                    default=["Donald Trump", "Lemon", "Dog", "President of the United States of America"])
    ap.add_argument("--negative_prompt", type=str, default="Donald Trump")
    ap.add_argument("--num_samples", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=10)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--dtype", type=str, default="fp16", choices=["fp16", "fp32"])  # match CORA default fp16 on cuda
    args = ap.parse_args()

    out_root = Path(args.out_root)
    torch_dtype = torch.float16 if (args.dtype == "fp16" and args.device == "cuda") else torch.float32

    # Load pipeline like CORA
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        safety_checker=None,
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(args.device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    # (Optional) small memory helpers
    for fn in ("enable_attention_slicing", "enable_vae_slicing", "enable_vae_tiling"):
        if hasattr(pipe, fn):
            try: getattr(pipe, fn)()
            except Exception: pass

    # Set global seed ONCE like CORA (no per-image Generators)
    torch.manual_seed(args.seed)
    if args.device == "cuda":
        try:
            torch.cuda.manual_seed_all(args.seed)
        except Exception:
            pass

    print(f"[INFO] Out root   : {out_root}")
    print(f"[INFO] Concepts   : {args.names}")
    print(f"[INFO] Neg prompt : {args.negative_prompt}")
    print(f"[INFO] Seed base  : {args.seed} (global, CORA-style)")
    print(f"[INFO] Batch size : {args.batch_size}\n")

    H = W = 512
    latent_h = H // pipe.vae_scale_factor  # 64 for SD 1.x
    latent_w = W // pipe.vae_scale_factor  # 64 for SD 1.x
    latent_shape = (4, latent_h, latent_w)

    for name in args.names:
        prompt = f"A photo of {name}."
        safe_name = sanitize(name)

        base_dir = out_root
        dir_orig  = ensure_dir(base_dir / "original")
        dir_erase = ensure_dir(base_dir / "erase")

        # We iterate samples in micro-batches;
        # for each chunk we pre-sample a batch of latents ONCE and reuse them
        total = args.num_samples
        for start_idx in range(0, total, args.batch_size):
            B = min(args.batch_size, total - start_idx)

            prompts_batch = [prompt] * B
            negs_batch    = [args.negative_prompt] * B

            # === CORA-style: pre-sample latents with global RNG and pass them in ===
            latents = torch.randn((B, *latent_shape), device=args.device, dtype=pipe.unet.dtype)

            # Originals (no NP) — reuse EXACT latents
            out_o = pipe(
                prompt=prompts_batch,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                height=H, width=W,
                latents=latents.clone()  # important: pipeline consumes/updates latents
            )
            imgs_o = out_o.images

            # Negative prompt with SAME latents
            out_np = pipe(
                prompt=prompts_batch,
                negative_prompt=negs_batch,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                height=H, width=W,
                latents=latents.clone()
            )
            imgs_np = out_np.images

            # Save
            for j, (im_o, im_np) in enumerate(zip(imgs_o, imgs_np)):
                idx = start_idx + j
                im_o.save(dir_orig  / f"A photo of {safe_name}_{idx:03d}.png")
                im_np.save(dir_erase / f"A photo of {safe_name}_{idx:03d}.png")

        print(f"[DONE] {name}: {args.num_samples} originals -> {dir_orig}, {args.num_samples} erase -> {dir_erase}")

if __name__ == "__main__":
    main()