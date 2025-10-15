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
        yield i, xs[i:i+n]  # (start_index, chunk)

def main():
    ap = argparse.ArgumentParser(description="Batched generation with negative prompt using 'A photo of {}'")
    ap.add_argument("--out_root", type=str, default="outputs_np", help="Output root directory")
    ap.add_argument("--target_root", type=str, default="Donald Trump", help="Top-level bucket (canonical target name)")
    ap.add_argument("--names", type=str, nargs="+",
                    default=["Donald Trump", "Lemon", "Dog", "President of the United States of America"],
                    help="Concepts rendered with template 'A photo of {}.'")
    ap.add_argument("--negative_prompt", type=str, default="Donald Trump",
                    help="Negative prompt string used for all 'erase' images")
    ap.add_argument("--num_samples", type=int, default=1, help="images per concept")
    ap.add_argument("--batch_size", type=int, default=1, help="micro-batch size for throughput")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--seed", type=int, default=0, help="base seed; sample i uses seed+i")
    ap.add_argument("--model", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu", "mps"])
    ap.add_argument("--dtype", type=str, default="fp32", choices=["fp16", "fp32"])
    args = ap.parse_args()

    out_root = Path(args.out_root)
    target_root = args.target_root

    # dtype / device
    torch_dtype = torch.float16 if (args.dtype == "fp16" and args.device == "cuda") else torch.float32

    # Load pipeline
    pipe = DiffusionPipeline.from_pretrained(
        args.model,
        safety_checker=None,
        torch_dtype=torch_dtype
    )
    pipe = pipe.to(args.device)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    # memory helpers if available
    try: pipe.enable_attention_slicing()
    except Exception: pass
    try: pipe.enable_vae_slicing()
    except Exception: pass
    try: pipe.enable_vae_tiling()
    except Exception: pass
    try: pipe.enable_model_cpu_offload()  # if you’re tight on VRAM
    except Exception: pass

    print(f"[INFO] Out root   : {out_root}")
    print(f"[INFO] Target root: {target_root}")
    print(f"[INFO] Concepts   : {args.names}")
    print(f"[INFO] Neg prompt : {args.negative_prompt}")
    print(f"[INFO] Seed base  : {args.seed}")
    print(f"[INFO] Batch size : {args.batch_size} (micro-batching)")
    print()

    for name in args.names:
        prompt = f"A photo of {name}."
        safe_name = sanitize(name)

        base_dir = out_root / target_root / name
        dir_orig  = ensure_dir(base_dir / "original")
        dir_erase = ensure_dir(base_dir / "erase")

        # seeds for this concept
        seeds = [args.seed + i for i in range(args.num_samples)]

        for start_idx, seed_chunk in chunked(seeds, args.batch_size):
            B = len(seed_chunk)
            # Build lists for this micro-batch
            prompts_batch = [prompt] * B
            negs_batch    = [args.negative_prompt] * B

            # Reproducible RNGs per image
            if args.device == "cuda":
                gens_orig = [torch.Generator(device="cuda").manual_seed(s) for s in seed_chunk]
                gens_np   = [torch.Generator(device="cuda").manual_seed(s) for s in seed_chunk]
            else:
                gens_orig = [torch.Generator().manual_seed(s) for s in seed_chunk]
                gens_np   = [torch.Generator().manual_seed(s) for s in seed_chunk]

            # Originals (no NP)
            out_o = pipe(
                prompt=prompts_batch,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=gens_orig
            )
            imgs_o = out_o.images  # List[Image.Image], length B

            # Negative-prompt with SAME seeds
            out_np = pipe(
                prompt=prompts_batch,
                negative_prompt=negs_batch,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                generator=gens_np
            )
            imgs_np = out_np.images

            # Save with global indices matching seed order
            for j, (im_o, im_np) in enumerate(zip(imgs_o, imgs_np)):
                idx = start_idx + j
                im_o.save(dir_orig  / f"{safe_name}_{idx:03d}_orig.png")
                im_np.save(dir_erase / f"{safe_name}_{idx:03d}_erase.png")

        print(f"[DONE] {name}: {args.num_samples} originals -> {dir_orig}, {args.num_samples} erase -> {dir_erase}")

if __name__ == "__main__":
    main()
