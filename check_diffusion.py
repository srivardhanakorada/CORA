#!/usr/bin/env python3
import os, sys, time
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

def make_pipeline(model_id: str, device: str = None, fp16: bool = True):
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if (fp16 and device == "cuda") else torch.float32

    print(f"[info] loading {model_id} (dtype={dtype}, device={device})...")
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,           # disable NSFW filter at load
    )
    # Extra guard: ensure no safety checker is called
    pipe.safety_checker = None

    # Faster sampler (feel free to change)
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    pipe = pipe.to(device)
    pipe.enable_attention_slicing()
    if device == "cuda" and dtype == torch.float16:
        pipe.enable_model_cpu_offload() if not torch.cuda.is_available() else None
    return pipe, device, dtype

def main():
    # You can swap this for another SD model if you prefer (e.g., "stabilityai/stable-diffusion-2-1")
    model_id = os.environ.get("SD_MODEL", "CompVis/stable-diffusion-v1-4")
    steps = int(os.environ.get("STEPS", "30"))
    guidance = float(os.environ.get("GUIDANCE", "7.5"))
    height = int(os.environ.get("H", "512"))
    width  = int(os.environ.get("W", "512"))

    pipe, device, dtype = make_pipeline(model_id)

    print("\n=== Interactive Stable Diffusion (NSFW checker disabled) ===")
    print("Type a prompt and press Enter to generate.")
    print("Special commands: /quit or /exit to leave, /seed <int> to set seed.")
    print("Images are saved to temp.png (overwritten each time).\n")

    generator = None  # torch.Generator for seeding (optional)

    while True:
        try:
            prompt = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n[info] bye.")
            break

        if not prompt:
            continue
        if prompt.lower() in ("/quit", "/exit"):
            print("[info] bye.")
            break
        if prompt.lower().startswith("/seed"):
            parts = prompt.split()
            if len(parts) == 2 and parts[1].lstrip("+-").isdigit():
                seed = int(parts[1])
                generator = torch.Generator(device=device).manual_seed(seed)
                print(f"[info] seed set to {seed}")
            else:
                print("[warn] usage: /seed 1234")
            continue

        t0 = time.time()
        with torch.inference_mode():
            # Optional negative prompt for general cleanup; leave empty if you want raw behavior
            result = pipe(
                prompt,
                guidance_scale=guidance,
                num_inference_steps=steps,
                height=height,
                width=width,
                generator=generator,
            )
        image = result.images[0]
        out_path = "temp.png"
        image.save(out_path)
        dt = time.time() - t0
        print(f"[done] saved {out_path} in {dt:.2f}s")

if __name__ == "__main__":
    main()
