#!/usr/bin/env python3
"""
main_cora_genbench.py
Run CORA erasure evaluation on GenBench-40 benchmark CSV.

Each row in the CSV defines one prompt, target_name, category, and seed.
The script generates:
  - Original image
  - CORA-erased image
  - Combined (side-by-side) image

Saved as:
  results/cora/genbench/{target_name}_{seed}.png
"""

import os
import re
import argparse
import pandas as pd
from tqdm import tqdm
from PIL import Image

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# === Imports from your CORA base code ===
from utils import seed_everything, get_token, get_textencoding, process_img
from main_cora_anc import (
    set_attenprocessor,
    record_concept_maps,
    build_cora_params_select_anchor,
    diffusion,
)


# --- Safe decoder ---
@torch.no_grad()
def decode_latents_batch(vae, latents):
    """
    Decode latent tensor [B,4,64,64] into list of RGB PIL images.
    Accepts either a single tensor or a batch.
    """
    if isinstance(latents, list):
        # flatten any [1,4,64,64] in list
        latents = torch.cat([x.squeeze(0).unsqueeze(0) for x in latents], dim=0)
    elif latents.ndim == 5 and latents.size(1) == 1:
        latents = latents.squeeze(1)
    elif latents.ndim != 4:
        raise ValueError(f"Unexpected latent shape: {tuple(latents.shape)}")

    latents = latents.to(vae.device)
    imgs = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    return [process_img(img) for img in imgs]


# --- Combined generation and saving (with separate originals/erased) ---
@torch.no_grad()
def generate_and_save(
    pipe,
    unet,
    vae,
    tokenizer,
    text_encoder,
    cora_params,
    uncond_encoding,
    prompt,
    target_name,
    category,
    seed,
    save_root,
    beta=0.5,
    tau=0.1,
    total_timesteps=30,
    guidance_scale=7.5,
):
    import os, re
    from PIL import Image

    device, dtype = pipe.device, pipe.unet.dtype
    latents = torch.randn(1, 4, 64, 64, device=device, dtype=dtype)

    enc = get_textencoding(get_token(prompt, tokenizer), text_encoder).to(device, dtype=dtype)
    text_embeddings = torch.cat([uncond_encoding, enc])

    # --- Original ---
    set_attenprocessor(unet, atten_type="original", params=None, record=False, only_cross=True)
    latents_orig = diffusion(
        unet,
        pipe.scheduler,
        latents.clone(),
        text_embeddings,
        total_timesteps,
        guidance_scale=guidance_scale,
        desc=f"{target_name} | original",
    )
    img_orig = decode_latents_batch(vae, latents_orig)[0]

    # --- CORA Erase ---
    set_attenprocessor(
        unet,
        atten_type="erase",
        params=cora_params,
        record=False,
        beta=beta,
        tau=tau,
        only_cross=True,
    )
    latents_erase = diffusion(
        unet,
        pipe.scheduler,
        latents.clone(),
        text_embeddings,
        total_timesteps,
        guidance_scale=guidance_scale,
        desc=f"{target_name} | CORA erase",
    )
    img_erase = decode_latents_batch(vae, latents_erase)[0]

    # --- Ensure directories ---
    safe_name = re.sub(r"[^\w]", "_", target_name)
    fname = f"{safe_name}_{seed}.png"

    root_dir = os.path.abspath(save_root)
    orig_dir = os.path.join(root_dir, "original")
    erase_dir = os.path.join(root_dir, "erased")
    os.makedirs(root_dir, exist_ok=True)
    os.makedirs(orig_dir, exist_ok=True)
    os.makedirs(erase_dir, exist_ok=True)

    # --- Save separate images ---
    path_orig = os.path.join(orig_dir, fname)
    path_erase = os.path.join(erase_dir, fname)
    img_orig.save(path_orig)
    img_erase.save(path_erase)

    # --- Combine side-by-side and save at root ---
    combined = Image.new("RGB", (img_orig.width * 2, img_orig.height))
    combined.paste(img_orig, (0, 0))
    combined.paste(img_erase, (img_orig.width, 0))
    path_combined = os.path.join(root_dir, fname)
    combined.save(path_combined)

    return {
        "combined": path_combined,
        "original": path_orig,
        "erased": path_erase,
    }

# --- Main ---
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_path", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="results/cora/genbench")
    parser.add_argument("--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--total_timesteps", type=int, default=30)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    df = pd.read_csv(args.csv_path)
    seed_everything(args.seed)

    # --- Load diffusion model ---
    print("[INFO] Loading Stable Diffusion pipeline...")
    pipe = DiffusionPipeline.from_pretrained(
        args.sd_ckpt, torch_dtype=torch.float16, safety_checker=None
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    device, dtype = pipe.device, pipe.unet.dtype
    uncond_encoding = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    # --- Iterate over unique targets ---
    unique_targets = df["target_name"].unique().tolist()

    for target in unique_targets:
        print(f"\n[INFO] Building CORA params for target = {target}")
        anchor = "a man"  # generic anchor
        preserve = ["Barack Obama", "Joe Biden", "Venom", "Anne Hathaway", "Marvel", "DC Comics", "Donald Duck", "iPhone", "Captain America"]

        tgt_map = record_concept_maps(unet, pipe, target, args.guidance_scale)["values"]
        anc_map = record_concept_maps(unet, pipe, anchor, args.guidance_scale)["values"]
        pres_maps = [record_concept_maps(unet, pipe, p, args.guidance_scale)["values"] for p in preserve]
        records_bundle = {"values": {"target": tgt_map, "anchors": [anc_map], "preserve": pres_maps}}
        cora_params, _, _ = build_cora_params_select_anchor(records_bundle, device, dtype)

        sub_df = df[df["target_name"] == target]
        print(f"[INFO] Generating {len(sub_df)} samples for {target}")

        for _, row in tqdm(sub_df.iterrows(), total=len(sub_df), desc=f"{target}"):
            prompt = row["prompt"]
            category = row["category"]
            seed_val = int(row["seed"])

            generate_and_save(
                pipe,
                unet,
                vae,
                tokenizer,
                text_encoder,
                cora_params,
                uncond_encoding,
                prompt,
                target,
                category,
                seed_val,
                args.save_root,
                beta=args.beta,
                tau=args.tau,
                total_timesteps=args.total_timesteps,
                guidance_scale=args.guidance_scale,
            )


if __name__ == "__main__":
    main()
