#!/usr/bin/env python3
# Inspect SDXL UNet structure — list all transformer_blocks.*.attn2 modules

import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

def main():
    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    unet = pipe.unet

    print("\n=== High-level blocks ===")
    print(f"down_blocks: {len(unet.down_blocks)}")
    print(f"up_blocks:   {len(unet.up_blocks)}")
    print(f"mid_block:   {type(unet.mid_block).__name__}")

    def list_block_attn(block, block_name):
        total_tf = 0
        total_attn2 = 0
        print(f"\n-- {block_name} --")
        for name, module in block.named_modules():
            # SDXL uses Transformer2DModel with .transformer_blocks (list)
            if hasattr(module, "transformer_blocks"):
                for i, tf in enumerate(module.transformer_blocks):
                    total_tf += 1
                    has_attn2 = hasattr(tf, "attn2")
                    total_attn2 += int(has_attn2)
                    attn = getattr(tf, "attn2", None)
                    heads = getattr(attn, "heads", None)
                    cross_dim = getattr(attn, "cross_attention_dim", None)
                    print(
                        f"{block_name}.{name}.transformer_blocks.{i}.attn2"
                        f" | present={has_attn2} | heads={heads} | cross_dim={cross_dim}"
                    )
        print(f"Summary {block_name}: transformer_blocks={total_tf}, attn2_present={total_attn2}")

    # Down blocks
    for idx, db in enumerate(unet.down_blocks):
        list_block_attn(db, f"down_blocks.{idx}")

    # Mid block
    list_block_attn(unet.mid_block, "mid_block")

    # Up blocks
    for idx, ub in enumerate(unet.up_blocks):
        list_block_attn(ub, f"up_blocks.{idx}")

    # Also print a compact list you can use as keys for params/processor installs
    keys = []
    for name, module in unet.named_modules():
        if ".transformer_blocks." in name and name.endswith(".attn2"):
            keys.append(name)
    print("\n=== attn2 keys (use these for CORA params) ===")
    for k in keys:
        print(k)
    print(f"Total attn2 modules found: {len(keys)}")

if __name__ == "__main__":
    main()
