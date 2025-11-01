#!/usr/bin/env python3
# ============================================================
# CORA-SDXL Generator (Baseline + Erase, with Debug)
# - SDXL Base 1.0 dual-encoder conditioning
# - Safe fp32 decode
# - CORA edits on cross-attn value stream (attn2)
# - DEBUG: attach/record counts + layerwise gating stats
# ============================================================

import os, copy, argparse
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
from diffusers.models.attention_processor import Attention, CrossAttention
from utils import seed_everything

# ----------------------------
# SDXL text encoding helpers
# ----------------------------

def _pad_1280(x: torch.Tensor) -> torch.Tensor:
    """Pad pooled embeddings to 1280 dims (required by SDXL UNet)."""
    if x.shape[-1] == 1280:
        return x
    pad = torch.zeros(x.size(0), 1280, device=x.device, dtype=x.dtype)
    feat = min(x.shape[-1], 1280)
    pad[:, :feat] = x[:, :feat]
    return pad

@torch.no_grad()
def encode_text_sdxl(pipe, prompt: str, device, dtype):
    """Encode prompt using SDXL's dual text encoders (bigG + CLIP-L)."""
    tok1, enc1 = pipe.tokenizer, pipe.text_encoder       # bigG (1280)
    tok2, enc2 = pipe.tokenizer_2, pipe.text_encoder_2   # CLIP-L (768)

    e1 = tok1(prompt, padding="max_length", truncation=True,
              max_length=tok1.model_max_length, return_tensors="pt")
    e2 = tok2(prompt, padding="max_length", truncation=True,
              max_length=tok2.model_max_length, return_tensors="pt")

    with torch.autocast("cuda", torch.float16):
        out1 = enc1(**{k: v.to(device) for k, v in e1.items()}, return_dict=True)
        out2 = enc2(**{k: v.to(device) for k, v in e2.items()}, return_dict=True)

    # Encoder 1: OpenCLIP-bigG (1280-dim)
    h1 = out1.last_hidden_state
    pooled1 = out1.pooler_output if getattr(out1, "pooler_output", None) is not None else h1.mean(dim=1)

    # Encoder 2: CLIP-L (768-dim → pad to 1280)
    if getattr(out2, "last_hidden_state", None) is not None:
        h2 = out2.last_hidden_state
    else:
        pooled_tmp = getattr(out2, "text_embeds", None)
        if pooled_tmp is None:
            raise RuntimeError("Neither last_hidden_state nor text_embeds found in encoder 2 output.")
        h2 = pooled_tmp.unsqueeze(1).repeat(1, h1.size(1), 1)

    pooled2 = out2.text_embeds if getattr(out2, "text_embeds", None) is not None else h2.mean(dim=1)
    pooled2 = _pad_1280(pooled2)

    # Align token dims and concat to 2048
    S = min(h1.size(1), h2.size(1))
    enc = torch.cat([h1[:, :S, :].to(dtype), h2[:, :S, :].to(dtype)], dim=-1)  # [B,S,2048]
    return enc, pooled1.to(dtype), pooled2.to(dtype)

@torch.no_grad()
def get_uncond_encoding_sdxl(pipe, device, dtype):
    """Encoding for unconditional branch (empty string prompt)."""
    return encode_text_sdxl(pipe, "", device, dtype)

def get_sdxl_cond_kwargs(pipe, pooled1, pooled2, batch, h=1024, w=1024):
    """Construct `added_cond_kwargs` for SDXL UNet."""
    device = pipe.device
    pooled1, pooled2 = _pad_1280(pooled1), _pad_1280(pooled2)
    time_ids = torch.tensor([h, w, h, w, 0, 0], device=device, dtype=torch.float32).unsqueeze(0).repeat(batch, 1)
    return {"text_embeds": pooled1, "add_text_embeds": pooled2, "time_ids": time_ids}

# ----------------------------
# CORA attention processor
# ----------------------------

class CORAProcessor(nn.Module):
    def __init__(self, name=None, params=None, record=False, beta=0.5, tau=0.1):
        super().__init__()
        self.name = name
        self.params = params
        self.record = record
        self.beta = beta
        self.tau = tau
        self.records = {"values": {}} if record else {}

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            b, c, h, w = hidden_states.shape
            hidden_states = hidden_states.view(b, c, h * w).transpose(1, 2)

        bsz, seq_len, _ = (hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape)
        attention_mask = attn.prepare_attention_mask(attention_mask, seq_len, bsz)

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        query = attn.to_q(hidden_states)
        key   = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attn_probs = attn.get_attention_scores(query, key, attention_mask)

        # Only operate on CLIP-length cross-attn (77 tokens)
        if encoder_hidden_states.shape[1] != 77:
            hidden_states = torch.bmm(attn_probs, value)
        else:
            if self.record:
                heads = attn.heads
                batch = value.size(0) // heads
                self.records["values"][self.name] = (value.detach(), heads, int(batch))
            elif self.params is not None:
                heads_cur = attn.heads
                d_head = value.size(2)
                batch_cur = max(1, value.size(0) // heads_cur)
                v_bhsd = value.view(batch_cur, heads_cur, seq_len, d_head)
                v_flat = v_bhsd.permute(0, 2, 1, 3).contiguous().view(batch_cur, seq_len, heads_cur * d_head)

                B_pad = self.params["B_pad"]  # [S,HD,K]
                u_hat = self.params["u_hat"]  # [S,HD]
                a_hat = self.params["a_hat"]  # [S,HD]

                # Project & edit
                if B_pad.size(2) > 0 and torch.count_nonzero(B_pad).item() > 0:
                    coeffs = torch.einsum('bsh,shk->bsk', v_flat, B_pad)
                    v_pres = torch.einsum('bsk,shk->bsh', coeffs, B_pad)
                else:
                    v_pres = torch.zeros_like(v_flat)

                v_free = v_flat - v_pres
                t = torch.einsum('bsh,sh->bs', v_free, u_hat)
                denom = torch.linalg.norm(v_free, dim=2) + 1e-8
                mask = (t.abs() / denom) >= self.tau

                # ---- DEBUG: layer-wise gating stats (sparse) ----
                with torch.no_grad():
                    frac_triggered = mask.float().mean().item()
                    avg_t = t.abs().mean().item()
                    avg_norm = denom.mean().item()
                    # Print for a small subset to avoid log spam
                    if torch.rand(1).item() < 0.02:
                        print(f"[CORA-DBG] {self.name} | trigger={frac_triggered:.3f} "
                              f"| |t|={avg_t:.4f} | norm={avg_norm:.4f} | beta={self.beta} tau={self.tau}")

                v_free_new = v_free - t.unsqueeze(-1) * u_hat.unsqueeze(0) + self.beta * t.unsqueeze(-1) * a_hat.unsqueeze(0)
                v_out = torch.where(mask.unsqueeze(-1), v_pres + v_free_new, v_flat)
                value = v_out.view(batch_cur, seq_len, heads_cur, d_head).permute(0, 2, 1, 3).contiguous().view(batch_cur * heads_cur, seq_len, d_head)

            hidden_states = torch.bmm(attn_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

def set_cora_processors(unet, params=None, record=False, beta=0.5, tau=0.1):
    """
    Attach CORA processors only to cross-attention layers that support set_processor().
    (These are CrossAttention instances, typically attn2 in transformer_blocks.)
    """
    attached = []
    for name, m in unet.named_modules():
        # Only modify valid CrossAttention layers
        if (
            isinstance(m, CrossAttention)
            or ("transformer_blocks" in name and ("attn2" in name or "cross" in name.lower()))
        ):
            # Ensure this module supports .set_processor()
            if hasattr(m, "set_processor"):
                proc_params = params.get(name) if (params and name in params) else None
                m.set_processor(CORAProcessor(name, proc_params, record, beta, tau))
                attached.append(name)

    print(f"[DEBUG] CORA attached to {len(attached)} cross-attention modules")
    for n in attached[:6]:
        print(f"    ↳ {n}")
    if len(attached) > 6:
        print("    ...")
    return unet


# ----------------------------
# Record & build CORA params
# ----------------------------

def _to_bh_sd(x: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    # [B*H, S, D] -> [B, H, S, D]
    return x.view(batch, heads, x.size(1), x.size(2))

def _hsd_to_flat_token(x_hsd: torch.Tensor) -> torch.Tensor:
    # [H,S,D] -> [S, H*D]
    H, S, D = x_hsd.shape
    return x_hsd.permute(1, 0, 2).contiguous().view(S, H * D)

@torch.no_grad()
def record_concept_maps(unet, pipe, concept_text: str, guidance_scale: float):
    """Single-step forward to record attention values for a given concept (with debug prints)."""
    device, dtype = pipe.device, pipe.unet.dtype
    enc, p1, p2 = encode_text_sdxl(pipe, concept_text, device, dtype)
    uncond, pu1, pu2 = get_uncond_encoding_sdxl(pipe, device, dtype)

    text = torch.cat([uncond, enc], dim=0)
    pooled1 = torch.cat([pu1, p1], dim=0)
    pooled2 = torch.cat([pu2, p2], dim=0)

    latents = torch.zeros(1, 4, 128, 128, device=device, dtype=dtype)
    scheduler = copy.deepcopy(pipe.scheduler)
    scheduler.set_timesteps(1)
    t = scheduler.timesteps[0]
    lat_in = torch.cat([latents] * 2)
    lat_in = scheduler.scale_model_input(lat_in, t)
    cond_kwargs = get_sdxl_cond_kwargs(pipe, pooled1, pooled2, batch=2, h=1024, w=1024)

    # --- attach and run ---
    rec_unet = set_cora_processors(copy.deepcopy(unet), params=None, record=True)
    _ = rec_unet(lat_in, t, encoder_hidden_states=text, added_cond_kwargs=cond_kwargs).sample

    # --- collect with normalized names ---
    vis = {"values": {}}
    total_rec = 0
    for k, proc in rec_unet.attn_processors.items():
        if hasattr(proc, "records") and "values" in proc.records:
            for name_in_record, v in proc.records["values"].items():
                vis["values"][name_in_record] = v
                total_rec += 1

    all_cross = [n for n in rec_unet.attn_processors.keys()
             if ("transformer_blocks" in n) and ("attn2" in n or "cross" in n.lower())]
    normalized_vis = {n.replace(".processor", "") for n in vis["values"].keys()}
    normalized_all = {n.replace(".processor", "") for n in all_cross}
    missed = normalized_all - normalized_vis

    print(f"[DEBUG] Recorded {len(normalized_vis)} cross-attention modules with captured values for concept: {concept_text}")
    print(f"[DEBUG] Missed {len(missed)} cross-attention modules (no record)")
    if len(missed) > 0:
        print("    Examples:", list(missed)[:5])


    # Debug summary
    print(f"[DEBUG] Recorded {len(normalized_vis)} attn2 modules with captured values for concept: {concept_text}")
    print(f"[DEBUG] Missed {len(missed)} attn2 modules (no record)")
    if len(missed) > 0:
        ex = list(missed)[:5]
        print("    Examples:", ex)

    return vis

@torch.no_grad()
def build_cora_params(records_bundle: Dict[str, Dict], device, model_dtype) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    CORA precompute in fp32 to avoid dtype mismatches:
      - Build preserve projector (QR) in fp32
      - Deflate + normalize in fp32
      - Cast final tensors to model dtype at the end
    """
    vals = records_bundle["values"]
    tgt_map = vals["target"]
    anc_map = vals["anchor"]
    pres_list = vals.get("preserve", [])
    modules = sorted(set(tgt_map.keys()) & set(anc_map.keys()))
    out: Dict[str, Dict[str, torch.Tensor]] = {}

    for mod in modules:
        t_val, h_t, b_t = tgt_map[mod]
        a_val, h_a, b_a = anc_map[mod]

        # reshape and select conditional stream idx=1
        t_bhsd = _to_bh_sd(t_val, b_t, h_t)
        a_bhsd = _to_bh_sd(a_val, b_a, h_a)
        t_hsd  = t_bhsd[1] if t_bhsd.size(0) > 1 else t_bhsd[0]
        a_hsd  = a_bhsd[1] if a_bhsd.size(0) > 1 else a_bhsd[0]

        # fp32 for algebra
        tar = _hsd_to_flat_token(t_hsd).to(torch.float32)  # [S,HD]
        anc = _hsd_to_flat_token(a_hsd).to(torch.float32)  # [S,HD]
        S, HD = tar.shape
        K = len(pres_list)

        # Build preserve basis in fp32
        B_pad = torch.zeros(S, HD, K, device=device, dtype=torch.float32)
        for i, m in enumerate(pres_list):
            if mod not in m:
                continue
            p_val, h_p, b_p = m[mod]
            p_bhsd = _to_bh_sd(p_val, b_p, h_p)
            p_hsd = p_bhsd[1] if p_bhsd.size(0) > 1 else p_bhsd[0]
            pf = _hsd_to_flat_token(p_hsd).to(torch.float32)
            B_pad[:, :, i] = pf

        # Orthonormalize columns per token
        if K > 0:
            for j in range(S):
                pj = B_pad[j]  # [HD, K]
                if pj.abs().sum() == 0:
                    continue
                q, _ = torch.linalg.qr(pj, mode='reduced')  # fp32
                r = min(q.shape[1], K)
                B_pad[j, :, :r] = q[:, :r]

        def _deflate(v: torch.Tensor) -> torch.Tensor:
            if K == 0:
                return v
            c = torch.einsum('sh,shk->sk', v, B_pad)           # fp32
            proj = torch.einsum('shk,sk->sh', B_pad, c)
            return v - proj

        # Target direction u_hat
        u = _deflate(tar)
        u = u / (torch.linalg.norm(u, dim=1, keepdim=True) + 1e-8)

        # Anchor direction a_hat (orthogonal to u)
        a = _deflate(anc)
        a = a - (torch.einsum('sh,sh->s', u, a).unsqueeze(1)) * u
        a = a / (torch.linalg.norm(a, dim=1, keepdim=True) + 1e-8)

        # Cast final tensors to model dtype & make contiguous
        out[mod] = {
            "B_pad": B_pad.to(device=device, dtype=model_dtype).contiguous(),
            "u_hat": u.to(device=device, dtype=model_dtype).contiguous(),
            "a_hat": a.to(device=device, dtype=model_dtype).contiguous(),
        }

    print(f"[DEBUG] Built CORA params for {len(out)} modules (intersection of target & anchor captures).")
    return out

# ----------------------------
# Diffusion & decode
# ----------------------------

@torch.no_grad()
def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps,
              guidance_scale, pipe, pooled1, pooled2, height, width):
    """Classifier-free guidance diffusion loop."""
    scheduler.set_timesteps(total_timesteps)
    latents = latents * scheduler.init_noise_sigma

    for t in tqdm(scheduler.timesteps):
        lat_in = torch.cat([latents] * 2, dim=0)
        lat_in = scheduler.scale_model_input(lat_in, t)
        cond_kwargs = get_sdxl_cond_kwargs(pipe, pooled1, pooled2,
                                           batch=lat_in.shape[0], h=height, w=width)
        noise_pred = unet(lat_in, t,
                          encoder_hidden_states=text_embeddings,
                          added_cond_kwargs=cond_kwargs).sample
        n_uncond, n_text = noise_pred.chunk(2)
        latents = scheduler.step(
            n_uncond + guidance_scale * (n_text - n_uncond), t, latents
        ).prev_sample

        # ---- DEBUG: per-step latent stats (every ~5 steps) ----
        # Note: t is a tensor; print every few steps by modulo on int(t)
        t_int = int(t.item()) if torch.is_tensor(t) else int(t)
        if total_timesteps > 0 and (t_int % max(1, total_timesteps // 6) == 0):
            print(f"[STEP-DBG] t={t_int} | latent mean={latents.mean():.4f} std={latents.std():.4f}")

    return latents

@torch.no_grad()
def decode_latents_fp32(vae, latents) -> List[Image.Image]:
    """Decode latents to RGB safely in fp32."""
    if isinstance(latents, list):
        latents = torch.stack(latents, dim=0)
    latents = (1.0 / float(vae.config.scaling_factor)) * latents

    vae_dtype = vae.dtype
    vae.to(dtype=torch.float32)
    with torch.autocast("cuda", enabled=False):
        imgs = vae.decode(latents.float(), return_dict=False)[0]
    vae.to(dtype=vae_dtype)

    imgs = (imgs / 2 + 0.5).clamp(0, 1)
    pil_imgs = []
    for img in imgs:
        img = torch.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)
        arr = (img.permute(1, 2, 0).cpu().numpy() * 255).round().astype("uint8")
        pil_imgs.append(Image.fromarray(arr))
    return pil_imgs

# ----------------------------
# Main
# ----------------------------

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--target_concept", required=True)
    parser.add_argument("--anchor_concept", default="a man")
    parser.add_argument("--preserve_concepts", default="")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_samples", type=int, default=1)  # ✅ added
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--xformers", action="store_true")
    args = parser.parse_args()

    assert args.num_samples % args.batch_size == 0, \
        "--num_samples must be a multiple of --batch_size"

    seed_everything(args.seed, True)

    pipe = DiffusionPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16, variant="fp16"
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if args.xformers:
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"[WARN] xformers not enabled: {e}")

    unet, vae = pipe.unet, pipe.vae
    device, model_dtype = pipe.device, pipe.unet.dtype
    bs = args.batch_size

    # ---- Record maps (fp16 capture, we'll upcast in builder) ----
    tgt = record_concept_maps(unet, pipe, args.target_concept, args.guidance_scale)["values"]
    anc = record_concept_maps(unet, pipe, args.anchor_concept, args.guidance_scale)["values"]
    pres = []
    for ptxt in [s.strip() for s in args.preserve_concepts.split(",") if s.strip()]:
        pres.append(record_concept_maps(unet, pipe, ptxt, args.guidance_scale)["values"])

    cora_params = build_cora_params(
        {"values": {"target": tgt, "anchor": anc, "preserve": pres}},
        device, model_dtype
    )

    # ---- Encode text for generation ----
    enc_uncond, p1u, p2u = get_uncond_encoding_sdxl(pipe, device, model_dtype)
    enc_cond,   p1c, p2c = encode_text_sdxl(pipe, args.prompt, device, model_dtype)
    text_embeddings = torch.cat([enc_uncond.repeat(bs, 1, 1),
                                 enc_cond.repeat(bs, 1, 1)], dim=0)
    pooled1 = torch.cat([p1u.repeat(bs, 1), p1c.repeat(bs, 1)], dim=0)
    pooled2 = torch.cat([p2u.repeat(bs, 1), p2c.repeat(bs, 1)], dim=0)

    H, W = int(args.height), int(args.width)
    h_lat, w_lat = H // 8, W // 8
    os.makedirs(os.path.join(args.save_path, "original"), exist_ok=True)
    os.makedirs(os.path.join(args.save_path, "erase"), exist_ok=True)

    # ---- Repeat for num_samples ----
    total_batches = args.num_samples // bs
    img_counter = 0
    for rep in range(total_batches):
        latents = torch.randn(bs, 4, h_lat, w_lat, device=device, dtype=model_dtype)

        # ---- Original ----
        set_cora_processors(unet, params=None, record=False)
        orig_latents = diffusion(unet, pipe.scheduler, latents.clone(), text_embeddings,
                                 args.steps, args.guidance_scale, pipe, pooled1, pooled2, H, W)

        # ---- Erased ----
        set_cora_processors(unet, params=cora_params, record=False, beta=args.beta, tau=args.tau)
        erase_latents = diffusion(unet, pipe.scheduler, latents.clone(), text_embeddings,
                                  args.steps, args.guidance_scale, pipe, pooled1, pooled2, H, W)

        # ---- Decode & save ----
        orig_imgs  = decode_latents_fp32(vae, orig_latents)
        erase_imgs = decode_latents_fp32(vae, erase_latents)

        for i in range(len(orig_imgs)):
            orig_imgs[i].save(os.path.join(args.save_path, "original", f"{img_counter}.png"))
            erase_imgs[i].save(os.path.join(args.save_path, "erase", f"{img_counter}.png"))
            img_counter += 1

    print(f"[OK] Saved {img_counter} samples to {args.save_path}")

if __name__ == "__main__":
    main()
