#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import copy
import argparse
from typing import Dict, List, Tuple
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from template import template_dict
from utils import *  # seed_everything, get_token, get_textencoding, get_eot_idx, get_spread_embedding, process_img

# =========================
#   GLOBAL INSTRUMENTATION
# =========================
CURRENT_TIMESTEP = None
CURRENT_PROMPT_TOKENS: List[str] = []
CURRENT_PROMPT_STR: str = ""
EFFECTIVE_LOG: List[Dict] = []  # rows appended across the run

def _ping(msg: str):
    print(msg, flush=True)

# ===== Utilities for shapes =====
def _to_bh_sd(x: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    # [B*H, S, D] -> [B, H, S, D]
    return x.view(batch, heads, x.size(1), x.size(2))

def _hsd_to_flat_token(x_hsd: torch.Tensor) -> torch.Tensor:
    # [H,S,D] -> [S, H*D]
    H, S, D = x_hsd.shape
    return x_hsd.permute(1, 0, 2).contiguous().view(S, H * D)

# ===== Recording (unchanged semantics; one forward step) =====
@torch.no_grad()
def record_concept_maps(unet, pipe, concept_text: str, guidance_scale: float):
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    device, dtype = pipe.device, pipe.unet.dtype

    enc = get_textencoding(get_token(concept_text, tokenizer), text_encoder)
    idx = get_eot_idx(get_token(concept_text, tokenizer))
    spread = get_spread_embedding(enc, idx).to(device, dtype=dtype)

    uncond = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    rec_unet = set_attenprocessor(
        copy.deepcopy(unet), atten_type="original", record=True, only_cross=True
    )

    vis_map = {"values": {}}

    scheduler = copy.deepcopy(pipe.scheduler)
    scheduler.set_timesteps(1)

    latents = torch.zeros(1, 4, 64, 64, device=device, dtype=dtype)
    text_embeddings = torch.cat([uncond, spread], dim=0)

    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, scheduler.timesteps[0])
    _ = rec_unet(latent_model_input, scheduler.timesteps[0], encoder_hidden_states=text_embeddings).sample

    for proc in rec_unet.attn_processors.values():
        if hasattr(proc, "records"):
            for k, v in proc.records.get("values", {}).items():
                if k not in vis_map["values"]:
                    vis_map["values"][k] = v  # (value, heads, batch)

    return vis_map

# ===== Precompute CORA params (fp32 precompute, model-dtype runtime) =====
@torch.no_grad()
def build_cora_params(records_bundle: Dict[str, Dict], device, dtype) -> Dict[str, Dict[str, torch.Tensor]]:
    vals = records_bundle.get("values", {})
    tgt_map = vals.get("target", {})
    anc_map = vals.get("anchor", {})
    pres_list_maps: List[Dict[str, Tuple[torch.Tensor, int, int]]] = vals.get("preserve", [])

    modules = sorted(set(tgt_map.keys()) & set(anc_map.keys()))
    K = len(pres_list_maps)

    out: Dict[str, Dict[str, torch.Tensor]] = {}

    for mod in modules:
        target_value, heads_t, batch_t = tgt_map[mod]
        anchor_value, heads_a, batch_a = anc_map[mod]

        tar_bhsd = _to_bh_sd(target_value, batch_t, heads_t)
        anc_bhsd = _to_bh_sd(anchor_value, batch_a, heads_a)
        tar_idx = 1 if tar_bhsd.size(0) > 1 else 0
        anc_idx = 1 if anc_bhsd.size(0) > 1 else 0
        tar_hsd = tar_bhsd[tar_idx]  # [H,S,D]
        anc_hsd = anc_bhsd[anc_idx]  # [H,S,D]

        tar_flat = _hsd_to_flat_token(tar_hsd).to(torch.float32)  # [S, HD]
        anc_flat = _hsd_to_flat_token(anc_hsd).to(torch.float32)  # [S, HD]
        S, HD = tar_flat.shape

        # Build orthonormal preserve basis per token
        B_pad = torch.zeros(S, HD, K, device=device, dtype=torch.float32)
        if K > 0:
            for k_idx, m in enumerate(pres_list_maps):
                if mod not in m: continue
                p_val, h_p, b_p = m[mod]
                p_bhsd = _to_bh_sd(p_val, b_p, h_p)
                p_idx = 1 if p_bhsd.size(0) > 1 else 0
                p_flat = _hsd_to_flat_token(p_bhsd[p_idx]).to(torch.float32)  # [S,HD]
                B_pad[:, :, k_idx] = p_flat

            B_ortho = torch.zeros_like(B_pad)
            for j in range(S):
                pj = B_pad[j]  # [HD,K]
                if pj.abs().sum() == 0: continue
                q, _ = torch.linalg.qr(pj, mode='reduced')  # [HD,r]
                r = min(q.shape[1], K)
                B_ortho[j, :, :r] = q[:, :r]
            B_pad = B_ortho  # [S,HD,K]

        def _deflate(vec_flat: torch.Tensor) -> torch.Tensor:
            if K == 0: return vec_flat
            c = torch.einsum('sh,shk->sk', vec_flat, B_pad)          # [S,K]
            proj = torch.einsum('shk,sk->sh', B_pad, c)              # [S,HD]
            return vec_flat - proj

        u_def = _deflate(tar_flat)
        u_hat = u_def / (torch.linalg.norm(u_def, dim=1, keepdim=True) + 1e-8)  # [S,HD]

        a_def = _deflate(anc_flat)
        a_perp = a_def - (torch.einsum('sh,sh->s', u_hat, a_def).unsqueeze(1)) * u_hat
        a_hat = a_perp / (torch.linalg.norm(a_perp, dim=1, keepdim=True) + 1e-8)  # [S,HD]

        out[mod] = {
            "B_pad": B_pad.to(device=device, dtype=dtype).contiguous(),
            "u_hat": u_hat.to(device=device, dtype=dtype).contiguous(),
            "a_hat": a_hat.to(device=device, dtype=dtype).contiguous(),
        }

    return out

# ===== Fast Processor with per-token, per-layer CORA logging =====
class VisualAttentionProcess(nn.Module):
    def __init__(self, module_name=None, atten_type="original", params=None, record=False, beta=0.5, tau=0.1,
                 log_cora=False, log_round_decimals=2):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.params = params
        self.record = record
        self.beta = beta
        self.tau = tau
        self.records = {"values": {}} if record else {}
        self.log_cora = log_cora
        self.log_round_decimals = log_round_decimals

    def __call__(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor")
        attn.processor = AttnProcessor(
            module_name=self.module_name,
            atten_type=self.atten_type,
            params=self.params,
            record=self.record,
            beta=self.beta,
            tau=self.tau,
            log_cora=self.log_cora,
            log_round_decimals=self.log_round_decimals,
        )
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)

class AttnProcessor:
    def __init__(self, module_name=None, atten_type="original", params=None, record=False, beta=0.5, tau=0.1,
                 log_cora=False, log_round_decimals=2):
        self.module_name = module_name
        self.atten_type = atten_type
        self.params = params
        self.record = record
        self.records = {"values": {}} if record else {}
        self.beta = beta
        self.tau = tau
        self.log_cora = log_cora
        self.log_round_decimals = log_round_decimals

    def _push_metrics(self, frac: torch.Tensor, gate: torch.Tensor, removed: torch.Tensor, replaced: torch.Tensor):
        # Log one row per token for this (layer, timestep, prompt)
        ts = int(CURRENT_TIMESTEP.item()) if CURRENT_TIMESTEP is not None else -1
        tokens = CURRENT_PROMPT_TOKENS
        prompt = CURRENT_PROMPT_STR
        L = frac.shape[1]  # sequence length
        for i in range(L):
            EFFECTIVE_LOG.append({
                "timestep": ts,
                "layer": self.module_name,
                "token_idx": int(i),
                "token_text": tokens[i] if i < len(tokens) else "",
                "prompt": prompt,
                "frac": float(torch.clamp(frac[0, i], 0).item()),
                "gate": float(gate[0, i].item()),
                "removed": float(removed[0, i].item()),
                "replaced": float(replaced[0, i].item()),
            })

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
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)  # [B*H, S, D]

        attn_probs = attn.get_attention_scores(query, key, attention_mask)

        if encoder_hidden_states.shape[1] != 77:
            # self-attn or non-CLIP: bypass
            hidden_states = torch.bmm(attn_probs, value)
        else:
            if self.record:
                heads = attn.heads
                batch = value.size(0) // heads
                self.records["values"][self.module_name] = (value.detach(), heads, int(batch))
            elif self.params is not None and self.atten_type == "erase":
                heads_cur = attn.heads
                d_head = value.size(2)
                batch_cur = max(1, value.size(0) // heads_cur)

                # Precomputed per-module params
                B_pad = self.params["B_pad"]  # [S,HD,K]
                u_hat = self.params["u_hat"]  # [S,HD]
                a_hat = self.params["a_hat"]  # [S,HD]

                # [B,H,S,D] -> [B,S,HD]
                v_bhsd = value.view(batch_cur, heads_cur, seq_len, d_head)
                v_flat = v_bhsd.permute(0, 2, 1, 3).contiguous().view(batch_cur, seq_len, heads_cur * d_head)

                # Preserve projection
                if B_pad.size(2) > 0 and torch.count_nonzero(B_pad).item() > 0:
                    coeffs = torch.einsum('bsh,shk->bsk', v_flat, B_pad)
                    v_pres = torch.einsum('bsk,shk->bsh', coeffs, B_pad)
                else:
                    v_pres = torch.zeros_like(v_flat)

                v_free = v_flat - v_pres  # [B,S,HD]
                t = torch.einsum('bsh,sh->bs', v_free, u_hat)  # [B,S]
                denom = torch.linalg.norm(v_free, dim=2) + 1e-8  # [B,S]
                frac = t.abs() / denom  # [B,S]
                gate = (frac >= self.tau)  # [B,S] (bool)

                # CORA edit
                v_free_new = v_free - t.unsqueeze(-1) * u_hat.unsqueeze(0) + self.beta * t.unsqueeze(-1) * a_hat.unsqueeze(0)
                v_free_out = torch.where(gate.unsqueeze(-1), v_free_new, v_free)
                v_new_flat = v_pres + v_free_out  # [B,S,HD]

                # === LOGGING (per token, per layer, per timestep) ===
                if self.log_cora:
                    removed = t.abs()                  # [B,S]
                    replaced = (self.beta * t.abs())   # [B,S]
                    # zero out BOS token for readability
                    if removed.shape[1] > 0:
                        removed[:, 0] = 0.0
                        replaced[:, 0] = 0.0
                        frac[:, 0] = 0.0
                        gate[:, 0] = False

                    # ---- key fix: log the CONDITIONAL half (not uncond) ----
                    B = frac.shape[0]
                    cond_start = B // 2
                    cond_frac     = frac[cond_start:, :].mean(0, keepdim=True)        # [1,S]
                    cond_gate     = gate[cond_start:, :].float().mean(0, keepdim=True)  # [1,S] mean of {0,1}
                    cond_removed  = removed[cond_start:, :].mean(0, keepdim=True)     # [1,S]
                    cond_replaced = replaced[cond_start:, :].mean(0, keepdim=True)    # [1,S]

                    self._push_metrics(cond_frac, cond_gate, cond_removed, cond_replaced)

                # Back to [B*H,S,D]
                v_new = v_new_flat.view(batch_cur, seq_len, heads_cur, d_head).permute(0, 2, 1, 3).contiguous()
                value = v_new.view(batch_cur * heads_cur, seq_len, d_head)

            hidden_states = torch.bmm(attn_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor

# ===== Install processors (attn2 only by default) =====
def set_attenprocessor(unet, atten_type="original", params=None, record=False, beta=0.5, tau=0.1, only_cross: bool = True,
                       log_cora: bool = False, log_round_decimals: int = 2):
    for name, m in unet.named_modules():
        if name.endswith("attn2") or (not only_cross and name.endswith("attn1")):
            m.set_processor(
                VisualAttentionProcess(
                    module_name=name,
                    atten_type=atten_type,
                    params=(params.get(name) if params is not None else None),
                    record=record,
                    beta=beta,
                    tau=tau,
                    log_cora=log_cora,
                    log_round_decimals=log_round_decimals,
                )
            )
    return unet

# ===== Diffusion loop with optional logging =====
@torch.no_grad()
def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0, guidance_scale=7.5, record=False, desc=None):
    visualize_map = {"values": {}} if record else {}
    scheduler.set_timesteps(total_timesteps)

    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        # record timestep for logging
        global CURRENT_TIMESTEP
        CURRENT_TIMESTEP = timestep

        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if record:
            for proc in unet.attn_processors.values():
                if hasattr(proc, "records"):
                    for k, v in proc.records.get("values", {}).items():
                        if k not in visualize_map["values"]:
                            visualize_map["values"][k] = v

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map) if record else latents

# ===== Batch decode =====
@torch.no_grad()
def decode_latents_batch(vae, latents_list: List[torch.Tensor]) -> List[Image.Image]:
    if len(latents_list) == 0:
        return []
    latents = torch.stack(latents_list, dim=0)
    imgs = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    out = [process_img(img) for img in imgs]
    return out

# ===== Main =====
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="original,erase")  # original, erase
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--total_timesteps", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--erase_type", type=str, default="", help="instance, style, celebrity")
    parser.add_argument("--target_concept", type=str, default="")
    parser.add_argument("--anchor_concept", type=str, default="a man")
    parser.add_argument("--preserve_concepts", type=str, default="")
    parser.add_argument("--beta", type=float, default=0.5)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--contents", type=str, default="")
    parser.add_argument("--save_path", type=str, required=True)
    # performance toggles
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    # logging & plotting
    parser.add_argument("--log_cora", action="store_true", help="Record per-token, per-layer CORA removal metrics")
    parser.add_argument("--round_decimals", type=int, default=2)
    parser.add_argument("--plot", action="store_true", help="Make a quick layer-vs-weight plot")
    parser.add_argument("--plot_tokens", type=str, default="", help="Comma-separated tokens to plot; default top-k by mean frac")
    parser.add_argument("--plot_topk", type=int, default=12)

    args = parser.parse_args()
    assert args.num_samples >= args.batch_size
    bs = args.batch_size
    mode_list = args.mode.replace(" ", "").split(",")
    concept_list = [s.strip() for s in args.contents.split(",") if s.strip()]
    assert args.target_concept.strip() != "", "Provide --target_concept"

    if args.tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    if args.xformers:
        try: pipe.enable_xformers_memory_efficient_attention()
        except Exception: pass

    if args.compile:
        try: pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
        except Exception: pass

    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    device, dtype = pipe.device, pipe.unet.dtype
    uncond_encoding = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    # ---- Record maps & Build CORA params ----
    cora_params = None
    if "erase" in mode_list:
        tgt_map = record_concept_maps(unet, pipe, args.target_concept.strip(), args.guidance_scale)["values"]
        preserve_list = [s.strip() for s in args.preserve_concepts.split(",") if s.strip()]
        pres_maps = [record_concept_maps(unet, pipe, pc, args.guidance_scale)["values"] for pc in preserve_list] if len(preserve_list) > 0 else []
        anc_map = record_concept_maps(unet, pipe, args.anchor_concept.strip(), args.guidance_scale)["values"]
        records_bundle = {"values": {"target": tgt_map, "anchor": anc_map, "preserve": pres_maps}}
        cora_params = build_cora_params(records_bundle, device, dtype)
        if args.debug:
            _ping(f"[DEBUG] CORA params for {len(cora_params)} modules.")

    # ---- Sampling ----
    seed_everything(args.seed, True)
    prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]

    for i in range(int(args.num_samples // bs)):
        latents = torch.randn(bs, 4, 64, 64, device=device, dtype=dtype)
        for concept, prompts in zip(concept_list, prompt_list):
            for prompt in prompts:
                Images = {}
                encoding = get_textencoding(get_token(prompt, tokenizer), text_encoder).to(device, dtype=dtype)
                txt = torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0)

                # record decoded tokens for current prompt (normalize)
                tok_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
                global CURRENT_PROMPT_TOKENS, CURRENT_PROMPT_STR
                CURRENT_PROMPT_TOKENS = [tokenizer.decode([t]).strip().lower() for t in tok_ids]
                CURRENT_PROMPT_STR = prompt

                if "original" in mode_list:
                    set_attenprocessor(unet, atten_type="original", params=None, record=False, only_cross=True, log_cora=False)
                    Images["original"] = diffusion(
                        unet=unet,
                        scheduler=pipe.scheduler,
                        latents=latents,
                        start_timesteps=0,
                        text_embeddings=txt,
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        desc=f"{prompt} | original",
                    )

                if "erase" in mode_list:
                    set_attenprocessor(
                        unet,
                        atten_type="erase",
                        params=cora_params,
                        record=False,
                        beta=args.beta,
                        tau=args.tau,
                        only_cross=True,
                        log_cora=args.log_cora,
                        log_round_decimals=args.round_decimals,
                    )
                    Images["erase"] = diffusion(
                        unet=unet,
                        scheduler=pipe.scheduler,
                        latents=latents,
                        start_timesteps=0,
                        text_embeddings=txt,
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        desc=f"{prompt} | CORA erase",
                    )

                # ---- Save images ----
                save_path = args.save_path
                for mode in mode_list:
                    os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1:
                    os.makedirs(os.path.join(save_path, "combine"), exist_ok=True)

                decoded = {name: decode_latents_batch(vae, [img for img in img_list]) for name, img_list in Images.items()}

                def combine_h(ims: List[Image.Image]) -> Image.Image:
                    widths, heights = zip(*(im.size for im in ims))
                    canvas = Image.new("RGB", (sum(widths), max(heights)))
                    x = 0
                    for im in ims:
                        canvas.paste(im, (x, 0)); x += im.size[0]
                    return canvas

                for idx in range(len(decoded[mode_list[0]])):
                    fname = re.sub(r"[^\w\s]", "", prompt).replace(", ", "_") + f"_{int(idx + bs * i)}.png"
                    row = []
                    for mode in mode_list:
                        decoded[mode][idx].save(os.path.join(save_path, mode, fname))
                        row.append(decoded[mode][idx])
                    if len(mode_list) > 1:
                        combine_h(row).save(os.path.join(save_path, "combine", fname))

    # ======= WRITE LOGS (always create files, even if empty) =======
    import pandas as pd
    os.makedirs(args.save_path, exist_ok=True)
    raw_path = os.path.join(args.save_path, "cora_effective_weight_raw.csv")
    agg_path = os.path.join(args.save_path, "cora_effective_weight_mean.csv")

    if EFFECTIVE_LOG:
        df = pd.DataFrame(EFFECTIVE_LOG)
    else:
        df = pd.DataFrame(columns=[
            "timestep","layer","token_idx","token_text","prompt","frac","gate","removed","replaced"
        ])

    # Aggregate mean over timesteps per (layer, token_text)
    if len(df) > 0:
        agg = (df.groupby(["layer", "token_text"], dropna=False)[["frac","gate","removed","replaced"]]
                 .mean().reset_index())
    else:
        agg = pd.DataFrame(columns=["layer","token_text","frac","gate","removed","replaced"])

    # Round numeric output
    num_cols = ["frac","gate","removed","replaced"]
    df_out, agg_out = df.copy(), agg.copy()
    for col in num_cols:
        if col in df_out.columns: df_out[col] = pd.to_numeric(df_out[col], errors="coerce").round(args.round_decimals)
        if col in agg_out.columns: agg_out[col] = pd.to_numeric(agg_out[col], errors="coerce").round(args.round_decimals)

    df_out.to_csv(raw_path, index=False)
    agg_out.to_csv(agg_path, index=False)
    _ping(f"[CORA] EFFECTIVE_LOG rows: {len(EFFECTIVE_LOG)}")
    _ping(f"[CORA] Wrote:\n  {raw_path}\n  {agg_path}")

    # ======= OPTIONAL PLOT =======
    if args.plot and len(agg_out) > 0:
        def _make_plots(agg_df, out_dir, tokens_csv, topk, round_dec):
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FormatStrFormatter
            import numpy as np
            import pandas as pd

            sub = agg_df.copy()
            sub["token_text"] = sub["token_text"].astype(str).str.lower()

            if tokens_csv.strip():
                tok_list = [t.strip().lower() for t in tokens_csv.split(",") if t.strip()]
            else:
                topk_idx = (sub.groupby("token_text")["frac"]
                            .mean().sort_values(ascending=False)
                            .head(topk).index.tolist())
                tok_list = topk_idx

            sub = sub[sub["token_text"].isin(tok_list)]
            if sub.empty:
                _ping("[CORA] Plotting skipped: token filter produced empty data.")
                return

            piv = (sub.pivot_table(index="token_text", columns="layer", values="frac", aggfunc="mean")
                     .fillna(0.0))
            piv = piv.applymap(lambda x: round(float(x), round_dec))
            layers = list(piv.columns)
            x = np.arange(len(layers))

            map_df = pd.DataFrame({"layer_idx": x, "layer": layers})
            map_path = os.path.join(out_dir, "cora_layer_index_map.csv")
            map_df.to_csv(map_path, index=False)

            plt.figure(figsize=(max(8, len(layers)*0.5), max(4, len(piv.index)*0.6)))
            for token in piv.index:
                y = piv.loc[token, layers].values.astype(float)
                plt.plot(x, y, marker="o", label=token)

            ax = plt.gca()
            ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{round_dec}f'))
            plt.xticks(x, [str(i) for i in x])
            plt.xlabel("layer index")
            plt.ylabel("mean frac (= |t| / ||v_free||)")
            plt.title("CORA mean removal strength across layers")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(loc="best", fontsize=9)
            plt.tight_layout()

            out_path = os.path.join(out_dir, "cora_line_weight.png")
            plt.savefig(out_path, dpi=220)
            plt.close()
            _ping(f"[CORA] Plots:\n  {out_path}\n  {map_path}")

        _make_plots(agg_out, args.save_path, args.plot_tokens, args.plot_topk, args.round_decimals)


if __name__ == "__main__":
    main()
