#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import re
import csv
import copy
import argparse
from typing import Dict, List, Tuple, Optional, Any

from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore
import torch  # type: ignore
from torch import nn  # type: ignore
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler  # type: ignore

from template import template_dict
from utils import *  # seed_everything, get_token, get_textencoding, get_eot_idx, get_spread_embedding, process_img

# ============================================================
# Multi-Concept CORA
# Modes:
#  (A) Legacy per-target pools
#  (B) Single-anchor CSV mode:
#      - erase all CSV targets
#      - sample ONLY targets + non-targets (preserves are EXCLUDED from sampling)
#      - build preserve projector from CSV 'preserve' rows
#
# Saving:
#  - Uses a SINGLE output directory via --save_path
#  - Images saved into: <save_path>/<mode>/   (and <save_path>/combine/ if both modes)
#  - Filenames include prompt slug for uniqueness
# ============================================================

# ---------------------------- shape utils ----------------------------
def _to_bh_sd(x: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    return x.view(batch, heads, x.size(1), x.size(2))  # [B*H, S, D] -> [B, H, S, D]

def _hsd_to_flat_token(x_hsd: torch.Tensor) -> torch.Tensor:
    H, S, D = x_hsd.shape  # [H,S,D] -> [S, H*D]
    return x_hsd.permute(1, 0, 2).contiguous().view(S, H * D)

# ---------------------------- recording ----------------------------
@torch.no_grad()
def record_concept_maps(unet, pipe, concept_text: str, guidance_scale: float):
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    device, dtype = pipe.device, pipe.unet.dtype

    enc = get_textencoding(get_token(concept_text, tokenizer), text_encoder)
    idx = get_eot_idx(get_token(concept_text, tokenizer))
    spread = get_spread_embedding(enc, idx).to(device, dtype=dtype)

    uncond = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    rec_unet = set_attenprocessor(copy.deepcopy(unet), atten_type="original", record=True, only_cross=True)

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

# ---------------------------- build params ----------------------------
@torch.no_grad()
def build_cora_params_multi(
    records_bundle: Dict[str, Any],
    device,
    dtype,
) -> Tuple[Dict[str, Dict[str, torch.Tensor]], List[int], List[List[float]]]:
    """
    records_bundle["values"]:
      - "targets":      [ {mod: (value, heads, batch)}, ... ]           # len R
      - "anchor_pools": [ [ {mod:(value,heads,batch)}, ... ], ... ]     # len R; each len M_r
      - "preserve":     [ {mod:(value,heads,batch)}, ... ]              # len K (K can be 0)
    Returns:
      per-module params: {"B_pad":[S,HD,K], "U_pad":[S,HD,R], "A_pad":[S,HD,R]}
      chosen_idxs: anchor choice per target
      scores_all: per-target candidate energies
    """
    vals = records_bundle.get("values", {})
    tgt_maps: List[Dict[str, Tuple[torch.Tensor, int, int]]] = vals["targets"]
    pools: List[List[Dict[str, Tuple[torch.Tensor, int, int]]]] = vals["anchor_pools"]
    pres_list_maps: List[Dict[str, Tuple[torch.Tensor, int, int]]] = vals.get("preserve", [])

    R = len(tgt_maps)
    assert R >= 1, "At least one target required."
    assert len(pools) == R, "anchor_pools must align with targets."

    module_sets = [set(tgt_maps[r].keys()) for r in range(R)]
    for r in range(R):
        for cand in pools[r]:
            module_sets.append(set(cand.keys()))
    for m in pres_list_maps:
        module_sets.append(set(m.keys()))
    modules = sorted(set.intersection(*module_sets)) if module_sets else []

    K = len(pres_list_maps)
    out: Dict[str, Dict[str, torch.Tensor]] = {}

    for mod in modules:
        # Targets -> flats
        t_flats = []
        S = HD = None
        for r in range(R):
            t_val, t_heads, t_batch = tgt_maps[r][mod]
            t_bhsd = _to_bh_sd(t_val, t_batch, t_heads)
            t_idx = 1 if t_bhsd.size(0) > 1 else 0
            t_hsd = t_bhsd[t_idx]
            t_flat = _hsd_to_flat_token(t_hsd).to(torch.float32)
            if S is None:
                S, HD = t_flat.shape
            t_flats.append(t_flat)

        # Preserves -> [S,HD,K]
        if K > 0:
            B_pad = torch.zeros(S, HD, K, device=device, dtype=torch.float32)
            for k_idx, m in enumerate(pres_list_maps):
                if mod not in m:
                    continue
                p_val, h_p, b_p = m[mod]
                p_bhsd = _to_bh_sd(p_val, b_p, h_p)
                p_idx = 1 if p_bhsd.size(0) > 1 else 0
                p_hsd = p_bhsd[p_idx]
                p_flat = _hsd_to_flat_token(p_hsd).to(torch.float32)
                B_pad[:, :, k_idx] = p_flat
            # per-token thin QR
            B_ortho = torch.zeros_like(B_pad)
            for j in range(S):
                pj = B_pad[j]  # [HD,K]
                if torch.allclose(pj.abs().sum(), torch.tensor(0.0, device=device)):
                    continue
                q, _ = torch.linalg.qr(pj, mode='reduced')
                r_rank = min(q.shape[1], K)
                B_ortho[j, :, :r_rank] = q[:, :r_rank]
            B_pad = B_ortho
        else:
            B_pad = torch.zeros(S, HD, 0, device=device, dtype=torch.float32)

        # Deflate helper
        def _deflate(vec_flat: torch.Tensor) -> torch.Tensor:
            if B_pad.size(2) == 0 or torch.count_nonzero(B_pad).item() == 0:
                return vec_flat
            c = torch.einsum('sh,shk->sk', vec_flat, B_pad)   # [S,K]
            proj = torch.einsum('shk,sk->sh', B_pad, c)       # [S,HD]
            return vec_flat - proj

        # U_pad from targets (orthonormal per token)
        U_list = []
        for r in range(R):
            u_def = _deflate(t_flats[r])
            u_hat = u_def / (torch.linalg.norm(u_def, dim=1, keepdim=True) + 1e-8)
            U_list.append(u_hat)
        U_stack = torch.stack(U_list, dim=2)  # [S,HD,R]
        U_ortho = torch.zeros_like(U_stack)
        for j in range(S):
            Uj = U_stack[j]  # [HD,R]
            if torch.allclose(Uj.abs().sum(), torch.tensor(0.0, device=device)):
                continue
            q, _ = torch.linalg.qr(Uj, mode='reduced')
            r_rank = min(q.shape[1], U_stack.size(2))
            U_ortho[j, :, :r_rank] = q[:, :r_rank]
        U_pad_fp32 = U_ortho

        # Score anchors per target for THIS module
        module_scores_per_target: List[List[float]] = []
        A_candidates_normed_per_target: List[List[torch.Tensor]] = []
        for r in range(R):
            scores_r, anchors_normed_r = [], []
            for a_map in pools[r]:
                a_val, a_heads, a_batch = a_map[mod]
                a_bhsd = _to_bh_sd(a_val, a_batch, a_heads)
                a_idx = 1 if a_bhsd.size(0) > 1 else 0
                a_hsd = a_bhsd[a_idx]
                a_flat = _hsd_to_flat_token(a_hsd).to(torch.float32)
                a_def = _deflate(a_flat)
                coeffs = torch.einsum('sh,shr->sr', a_def, U_pad_fp32)
                projU  = torch.einsum('sr,shr->sh', coeffs, U_pad_fp32)
                a_perp = a_def - projU
                e = (torch.linalg.norm(a_perp, dim=1) ** 2).mean().item()
                scores_r.append(e)
                a_hat = a_perp / (torch.linalg.norm(a_perp, dim=1, keepdim=True) + 1e-8)
                anchors_normed_r.append(a_hat)
            module_scores_per_target.append(scores_r)
            A_candidates_normed_per_target.append(anchors_normed_r)

        out.setdefault("__accum__", {"scores": [], "mods": []})
        out["__accum__"]["scores"].append(module_scores_per_target)
        out["__accum__"]["mods"].append(mod)

        out.setdefault("__stash__", {})
        out["__stash__"].setdefault(mod, {})
        out["__stash__"][mod].update({
            "B_pad_fp32": B_pad,
            "U_pad_fp32": U_pad_fp32,
            "A_cands_fp32": A_candidates_normed_per_target,
        })

    # Decide anchors globally (avg over modules)
    chosen_idxs, scores_all = [], []
    if modules:
        num_mods = len(out["__accum__"]["scores"])
        sums = [ [0.0]*len(pools[r]) for r in range(len(tgt_maps)) ]
        for mod_scores in out["__accum__"]["scores"]:
            for r in range(len(tgt_maps)):
                for m_idx in range(len(pools[r])):
                    sums[r][m_idx] += float(mod_scores[r][m_idx])
        for r in range(len(tgt_maps)):
            means_r = [s/num_mods for s in sums[r]]
            scores_all.append(means_r)
            chosen_idxs.append(int(max(range(len(pools[r])), key=lambda i: means_r[i])))
    else:
        chosen_idxs = [0]*len(tgt_maps)
        scores_all = [[0.0]*len(pools[r]) for r in range(len(tgt_maps))]

    # Finalize tensors per module
    final_params: Dict[str, Dict[str, torch.Tensor]] = {}
    for mod in modules:
        stash = out["__stash__"][mod]
        B_pad = stash["B_pad_fp32"].to(device=device, dtype=dtype).contiguous()
        U_pad = stash["U_pad_fp32"].to(device=device, dtype=dtype).contiguous()
        A_sel = []
        for r in range(len(tgt_maps)):
            a_hat = stash["A_cands_fp32"][r][chosen_idxs[r]].to(device=device, dtype=dtype).contiguous()
            A_sel.append(a_hat)
        A_pad = torch.stack(A_sel, dim=2)  # [S,HD,R]
        final_params[mod] = {"B_pad": B_pad, "U_pad": U_pad, "A_pad": A_pad}

    if "__accum__" in out: del out["__accum__"]
    if "__stash__" in out: del out["__stash__"]
    return final_params, chosen_idxs, scores_all

# ---------------------------- fast processor ----------------------------
class VisualAttentionProcess(nn.Module):
    def __init__(self, module_name=None, atten_type="original", params=None, record=False, beta=0.5, tau=0.1):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.params = params
        self.record = record
        self.beta = beta
        self.tau = tau
        self.records = {"values": {}} if record else {}

    def __call__(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor")
        attn.processor = AttnProcessor(
            module_name=self.module_name,
            atten_type=self.atten_type,
            params=self.params,
            record=self.record,
            beta=self.beta,
            tau=self.tau,
        )
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)

class AttnProcessor:
    def __init__(self, module_name=None, atten_type="original", params=None, record=False, beta=0.5, tau=0.1):
        self.module_name = module_name
        self.atten_type = atten_type
        self.params = params
        self.record = record
        self.records = {"values": {}} if record else {}
        self.beta = beta
        self.tau = tau

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

                B_pad = self.params["B_pad"]  # [S,HD,K]
                U_pad = self.params["U_pad"]  # [S,HD,R]
                A_pad = self.params["A_pad"]  # [S,HD,R]

                v_bhsd = value.view(batch_cur, heads_cur, seq_len, d_head)
                v_flat = v_bhsd.permute(0, 2, 1, 3).contiguous().view(batch_cur, seq_len, heads_cur * d_head)

                # preserve projection
                if B_pad.size(2) > 0 and torch.count_nonzero(B_pad).item() > 0:
                    coeffs = torch.einsum('bsh,shk->bsk', v_flat, B_pad)
                    v_pres = torch.einsum('bsk,shk->bsh', coeffs, B_pad)
                else:
                    v_pres = torch.zeros_like(v_flat)

                v_free = v_flat - v_pres
                t = torch.einsum('bsh,shr->bsr', v_free, U_pad)
                den = torch.linalg.norm(v_free, dim=2, keepdim=True) + 1e-8
                gate = (t.abs().amax(dim=2, keepdim=True) / den) >= self.tau

                erase = torch.einsum('bsr,shr->bsh', t, U_pad)

                if isinstance(self.beta, (list, tuple)):
                    beta_vec = torch.as_tensor(self.beta, device=v_free.device, dtype=v_free.dtype)
                elif torch.is_tensor(self.beta):
                    beta_vec = self.beta.to(device=v_free.device, dtype=v_free.dtype)
                else:
                    beta_vec = torch.tensor([float(self.beta)], device=v_free.device, dtype=v_free.dtype).repeat(U_pad.size(2))
                rep_scale = torch.einsum('r,bsr->bsr', beta_vec, t)
                rep = torch.einsum('bsr,shr->bsh', rep_scale, A_pad)

                v_free_new = v_free - erase + rep
                v_free_out = torch.where(gate, v_free_new, v_free)
                v_new_flat = v_pres + v_free_out

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

# ---------------------------- install processors ----------------------------
def set_attenprocessor(unet, atten_type="original", params=None, record=False, beta=0.5, tau=0.1, only_cross: bool = True):
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
                )
            )
    return unet

# ---------------------------- diffusion loop ----------------------------
@torch.no_grad()
def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0, guidance_scale=7.5, record=False, desc=None):
    visualize_map = {"values": {}} if record else {}
    scheduler.set_timesteps(total_timesteps)

    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
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

# ---------------------------- decode ----------------------------
@torch.no_grad()
def decode_latents_batch(vae, latents_list: List[torch.Tensor]) -> List[Image.Image]:
    if len(latents_list) == 0:
        return []
    latents = torch.stack(latents_list, dim=0)
    imgs = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    out = [process_img(img) for img in imgs]
    return out

# ---------------------------- helpers ----------------------------
def _parse_targets(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if s.strip()]

def _parse_anchor_pools(s: str, R_expected: int) -> List[List[str]]:
    groups = [g.strip() for g in s.split("|")] if s.strip() else []
    if len(groups) == 0:
        return [[] for _ in range(R_expected)]
    if len(groups) != R_expected:
        raise ValueError(f"--anchor_pools expects {R_expected} groups separated by '|', got {len(groups)}")
    pools = []
    for g in groups:
        cand = [x.strip() for x in g.split(",") if x.strip()]
        pools.append(cand)
    return pools

def _parse_betas(s: str, R_expected: int, default_beta: float) -> List[float]:
    if not s or ("," not in s and "|" not in s):
        return [default_beta]*R_expected
    parts = [p.strip() for p in re.split(r"[,\|]", s) if p.strip()]
    if len(parts) == 1:
        return [float(parts[0])]*R_expected
    if len(parts) != R_expected:
        raise ValueError(f"--beta as list must have {R_expected} values; got {len(parts)}")
    return [float(x) for x in parts]

def load_concepts_csv(path: str) -> Tuple[List[str], List[str], List[str]]:
    """CSV columns: type,name where type ∈ {target, non-target, preserve}.
    Returns (targets, non_targets, preserves), deduped in order of first appearance.
    """
    targets, non_targets, preserves = [], [], []
    seen = set()
    with open(path, newline='', encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            t = (row.get("type") or "").strip().lower()
            n = (row.get("name") or "").strip()
            if not n or n in seen:
                continue
            seen.add(n)
            if t == "target":
                targets.append(n)
            elif t == "non-target":
                non_targets.append(n)
            elif t == "preserve":
                preserves.append(n)
            else:
                # unknown type → ignore
                pass
    if len(targets) == 0:
        raise ValueError("CSV contains no rows with type=='target'.")
    return targets, non_targets, preserves

# ---------------------------- main ----------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, required=True,
                        help="Single output directory; images saved as <save_path>/<mode>/<promptslug>_*.png")
    parser.add_argument("--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="original,erase")  # original, erase
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--total_timesteps", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)
    parser.add_argument("--erase_type", type=str, default="", help="instance, style, celebrity")

    # legacy multi-target args
    parser.add_argument("--target_concepts", type=str, default="")
    parser.add_argument("--anchor_pools", type=str, default="")
    parser.add_argument("--preserve_concepts", type=str, default="")
    parser.add_argument("--beta", type=str, default="0.5")
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--contents", type=str, default="",
                        help="Comma list of concepts to generate. NOTE: any concepts that also appear in --preserve_concepts will be excluded from generation.")

    # perf toggles
    parser.add_argument("--xformers", action="store_true")
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--tf32", action="store_true")
    parser.add_argument("--fp32_linalg", action="store_true")

    # single-anchor CSV mode
    parser.add_argument("--single_anchor_mode", action="store_true",
                        help="Use CSV (type,name); erase all targets; sample ONLY targets + non-targets (preserves excluded from sampling); build preserves from CSV preserves.")
    parser.add_argument("--concepts_csv", type=str, default="")
    parser.add_argument("--single_anchor_text", type=str, default="an animal")

    args = parser.parse_args()

    assert args.num_samples >= args.batch_size
    bs = args.batch_size
    mode_list = args.mode.replace(" ", "").split(",")

    # --- Parse mode ---
    if args.single_anchor_mode:
        assert args.concepts_csv, "Provide --concepts_csv when --single_anchor_mode is set."
        targets, non_targets, preserves = load_concepts_csv(args.concepts_csv)

        # contents = ONLY targets + non-targets (preserves EXCLUDED from generation)
        concept_list = list(targets) + list(non_targets)

        # erase config: only targets matter; one shared anchor
        pools = [[args.single_anchor_text] for _ in targets]
        betas = _parse_betas(args.beta, len(targets), default_beta=0.5)

        # Preserve projector from CSV 'preserve' rows (not generated)
        preserve_list = list(preserves)

        if args.debug:
            print("[DEBUG] Single-anchor mode ON")
            print(f"  #targets={len(targets)}  #non-targets={len(non_targets)}  #preserves={len(preserves)}")
            print(f"  #contents(sampled)={len(concept_list)} (preserves excluded from sampling)")
            print("  First targets:", targets[:5])
            print("  First non-targets:", non_targets[:5])
            print("  First preserves:", preserves[:5])
            print(f"  Shared anchor: {args.single_anchor_text!r}")
    else:
        # legacy path
        targets = _parse_targets(args.target_concepts)
        assert len(targets) >= 1, "Provide --target_concepts in legacy mode."
        pools = _parse_anchor_pools(args.anchor_pools, len(targets))
        betas = _parse_betas(args.beta, len(targets), default_beta=0.5)
        preserve_list = [s.strip() for s in args.preserve_concepts.split(",") if s.strip()]
        concept_list = [s.strip() for s in args.contents.split(",") if s.strip()]
        # EXCLUDE preserves from generation in legacy mode too
        if preserve_list:
            preserve_set = set(preserve_list)
            concept_list = [c for c in concept_list if c not in preserve_set]

    if args.tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    # ---- Models ----
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

    # ---- Encodings ----
    uncond_encoding = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    # ---- Precompute CORA params ----
    if "erase" in mode_list:
        tgt_maps = [record_concept_maps(unet, pipe, t, args.guidance_scale)["values"] for t in targets]
        pres_maps = [record_concept_maps(unet, pipe, pc, args.guidance_scale)["values"] for pc in preserve_list]

        anchor_pools_maps: List[List[Dict[str, Tuple[torch.Tensor,int,int]]]] = []
        for r, pool in enumerate(pools):
            cand = pool if len(pool) > 0 else ["a man"]
            anchor_pools_maps.append([record_concept_maps(unet, pipe, a_txt, args.guidance_scale)["values"] for a_txt in cand])

        records_bundle = {"values": {"targets": tgt_maps, "anchor_pools": anchor_pools_maps, "preserve": pres_maps}}
        cora_params_allmods, chosen_idxs, scores_all = build_cora_params_multi(records_bundle, device, dtype)

        if args.debug:
            print("[DEBUG] Chosen anchors per target:")
            for r, t in enumerate(targets):
                pool_names = pools[r] if len(pools[r]) > 0 else ["a man"]
                sel = pool_names[chosen_idxs[r]]
                print(f"  Target[{r}] {t}: {sel}")
                print("   Scores:", {pool_names[i]: round(scores_all[r][i], 4) for i in range(len(pool_names))})

    # ---- Prepare save dirs (single root) ----
    os.makedirs(args.save_path, exist_ok=True)
    for mode in mode_list:
        os.makedirs(os.path.join(args.save_path, mode), exist_ok=True)
    if len(mode_list) > 1:
        os.makedirs(os.path.join(args.save_path, "combine"), exist_ok=True)

    # ---- Sampling ----
    seed_everything(args.seed, True)
    prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]

    # helper: slugify concept + prompt
    def _slug(s: str) -> str:
        s = re.sub(r"[^\w\s-]", "", s).strip()
        s = re.sub(r"[\s]+", "_", s)
        return s

    for i in range(int(args.num_samples // bs)):
        latents = torch.randn(bs, 4, 64, 64, device=device, dtype=dtype)
        for concept, prompts in zip(concept_list, prompt_list):
            for prompt in prompts:
                Images = {}
                encoding = get_textencoding(get_token(prompt, tokenizer), text_encoder).to(device, dtype=dtype)
                txt = torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0)

                if "original" in mode_list:
                    set_attenprocessor(unet, atten_type="original", params=None, record=False, only_cross=True)
                    Images["original"] = diffusion(
                        unet=unet, scheduler=pipe.scheduler, latents=latents, start_timesteps=0,
                        text_embeddings=txt, total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale, desc=f"{prompt} | original",
                    )

                if "erase" in mode_list:
                    set_attenprocessor(
                        unet, atten_type="erase", params=cora_params_allmods, record=False,
                        beta=[float(b) for b in betas], tau=args.tau, only_cross=True,
                    )
                    Images["erase"] = diffusion(
                        unet=unet, scheduler=pipe.scheduler, latents=latents, start_timesteps=0,
                        text_embeddings=txt, total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale, desc=f"{prompt} | MC-CORA erase",
                    )

                # ---- Decode & Save (single directory tree) ----
                decoded = {name: decode_latents_batch(vae, [img for img in img_list]) for name, img_list in Images.items()}

                def combine_h(ims: List[Image.Image]) -> Image.Image:
                    widths, heights = zip(*(im.size for im in ims))
                    canvas = Image.new("RGB", (sum(widths), max(heights)))
                    x = 0
                    for im in ims:
                        canvas.paste(im, (x, 0))
                        x += im.size[0]
                    return canvas

                # filename pattern: <promptslug>_<idx>.png
                pslug = _slug(prompt)
                for idx in range(len(decoded[mode_list[0]])):
                    base = f"{pslug}_{int(idx + bs * i)}.png"
                    row = []
                    for mode in mode_list:
                        out_p = os.path.join(args.save_path, mode, base)
                        decoded[mode][idx].save(out_p)
                        row.append(decoded[mode][idx])
                    if len(mode_list) > 1:
                        combine_h(row).save(os.path.join(args.save_path, "combine", base))

if __name__ == "__main__":
    main()
