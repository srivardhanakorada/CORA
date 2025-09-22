import os
import re
import copy
import argparse
from typing import Dict, List, Tuple

from PIL import Image  # type: ignore
from tqdm import tqdm  # type: ignore
import torch  # type: ignore
from torch import nn  # type: ignore
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler  # type: ignore

from template import template_dict
from utils import *  # seed_everything, get_token, get_textencoding, get_eot_idx, get_spread_embedding, process_img

# ------------------------------------------------------------
# CORA: FAST IMPLEMENTATION (vectorized; zero per-token Python loops)
# Key accelerations:
#   1) Precompute per-module, per-token CORA params (B_j, u_hat_j, a_hat_j) once.
#   2) Vectorized projection via einsum; no torch.linalg in the forward.
#   3) No deepcopy of UNet; reuse a single UNet and swap processors.
#   4) Restrict processors to cross-attn (attn2) only.
#   5) Batch VAE decode.
#   6) Avoid deepcopy(records) inside the processor.
#   7) Optional TF32 / xFormers / torch.compile toggles.
# ------------------------------------------------------------

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

    # Lightweight record: install a temporary recorder only on attn2
    rec_unet = set_attenprocessor(
        copy.deepcopy(unet), atten_type="original", record=True, only_cross=True
    )

    vis_map = {"values": {}}

    scheduler = copy.deepcopy(pipe.scheduler)
    scheduler.set_timesteps(1)

    latents = torch.zeros(1, 4, 64, 64, device=device, dtype=dtype)
    text_embeddings = torch.cat([uncond, spread], dim=0)

    # Single step, just to collect per-module value tensors
    latent_model_input = torch.cat([latents] * 2)
    latent_model_input = scheduler.scale_model_input(latent_model_input, scheduler.timesteps[0])
    noise_pred = rec_unet(latent_model_input, scheduler.timesteps[0], encoder_hidden_states=text_embeddings).sample

    # Scrape records from processors
    for proc in rec_unet.attn_processors.values():
        if hasattr(proc, "records"):
            for k, v in proc.records.get("values", {}).items():
                if k not in vis_map["values"]:
                    vis_map["values"][k] = v  # (value, heads, batch)

    return vis_map


# ===== Precompute fast CORA params per module =====
@torch.no_grad()
def build_cora_params(records_bundle: Dict[str, Dict], device, dtype) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    Input bundle format (as built in main):
    {
      "values": {
        "target":   {module: (value, heads, batch)},
        "anchor":   {module: (value, heads, batch)},
        "preserve": [{module: (value, heads, batch)}, ...]
      }
    }
    Output per-module params:
    {
      module_name: {
         "B_pad":  [S, HD, K],  (orthonormal basis columns, zero-padded to K)
         "u_hat":  [S, HD],
         "a_hat":  [S, HD],
      }
    }
    All tensors are moved to (device, dtype) and contiguous for fast einsum.
    """
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

        # Reshape to [B, H, S, D] and pick conditional stream idx (1 if present)
        tar_bhsd = _to_bh_sd(target_value, batch_t, heads_t)
        anc_bhsd = _to_bh_sd(anchor_value, batch_a, heads_a)
        tar_idx = 1 if tar_bhsd.size(0) > 1 else 0
        anc_idx = 1 if anc_bhsd.size(0) > 1 else 0
        tar_hsd = tar_bhsd[tar_idx]  # [H,S,D]
        anc_hsd = anc_bhsd[anc_idx]  # [H,S,D]

        tar_flat = _hsd_to_flat_token(tar_hsd)  # [S, HD]
        anc_flat = _hsd_to_flat_token(anc_hsd)  # [S, HD]
        S, HD = tar_flat.shape

        # Build preserve matrix per token j: P_j = [p1_j, ..., pK_j] in [HD, K]
        # Compute QR for orthonormal basis B_j (rank r <= K). Zero-pad to K for uniform shape.
        B_pad = torch.zeros(S, HD, K, device=device, dtype=torch.float32)
        if K > 0:
            for k_idx, m in enumerate(pres_list_maps):
                if mod not in m:
                    continue
                p_val, h_p, b_p = m[mod]
                p_bhsd = _to_bh_sd(p_val, b_p, h_p)
                p_idx = 1 if p_bhsd.size(0) > 1 else 0
                p_hsd = p_bhsd[p_idx]
                p_flat = _hsd_to_flat_token(p_hsd).to(torch.float32)  # [S, HD]
                B_pad[:, :, k_idx] = p_flat

            # Orthonormalize columns per token (thin QR). Handle rank-deficiency.
            # We QR the [HD, K] matrix for each token via torch.linalg.qr on a loop over S (S=77)
            # Cost is tiny and done once.
            B_ortho = torch.zeros_like(B_pad)
            for j in range(S):
                pj = B_pad[j]  # [HD, K]
                # Remove all-zero case quickly
                if torch.allclose(pj.abs().sum(), torch.tensor(0.0, device=device)):
                    continue
                # QR on [HD, K]
                q, r = torch.linalg.qr(pj, mode='reduced')  # q: [HD, r]
                r_rank = min(q.shape[1], K)
                B_ortho[j, :, :r_rank] = q[:, :r_rank]
            B_pad = B_ortho  # [S, HD, K]

        # Deflate u and a against preserves and orthogonalize a wrt u
        def _deflate_and_norm(vec_flat: torch.Tensor) -> torch.Tensor:
            # vec_flat: [S, HD] (float32)
            if K == 0:
                v_def = vec_flat
            else:
                coeffs = torch.einsum('shk,sk->sh', B_pad, torch.zeros(S, K, device=device))  # dummy to keep graph off
                # Compute v_def = v - B (B^T v)
                vt = vec_flat  # [S, HD]
                c = torch.einsum('shk,sk->sh', B_pad, torch.einsum('sh,shk->sk', vt, B_pad))  # [S, HD]
                v_def = vt - c
            # Normalize row-wise
            n = torch.linalg.norm(v_def, dim=1, keepdim=True) + 1e-8
            out = v_def / n
            return out

        # More efficient deflation using einsum directly
        def _deflate(vec_flat: torch.Tensor) -> torch.Tensor:
            if K == 0:
                return vec_flat
            # c = v^T B  => [S,K]
            c = torch.einsum('sh,shk->sk', vec_flat, B_pad)
            # proj = B c => [S,HD]
            proj = torch.einsum('shk,sk->sh', B_pad, c)
            return vec_flat - proj

        u_def = _deflate(tar_flat.to(torch.float32))
        # Normalize u
        u_norm = torch.linalg.norm(u_def, dim=1, keepdim=True) + 1e-8
        u_hat = u_def / u_norm  # [S,HD] float32

        a_def = _deflate(anc_flat.to(torch.float32))
        # Make a ⟂ u and normalize
        a_perp = a_def - (torch.einsum('sh,sh->s', u_hat, a_def).unsqueeze(1)) * u_hat
        a_norm = torch.linalg.norm(a_perp, dim=1, keepdim=True) + 1e-8
        a_hat = a_perp / a_norm  # [S,HD] float32
        # --- Diagnostics (build-time; no math change) ---
        a_def_norm = torch.linalg.norm(a_def, dim=1, keepdim=True) + 1e-8  # [S,1]
        cos_u_a_raw = torch.einsum('sh,sh->s', u_hat, a_def / a_def_norm)  # [S]
        a_perp_small_frac = (a_norm.squeeze(1) < 1e-3).float().mean().item()
        diag = {
            "cos_u_a_mean": float(cos_u_a_raw.mean().item()),   # alignment before making a ⟂ u
            "a_perp_small_frac": float(a_perp_small_frac),      # % tokens where anchor collapsed after ⟂
        }
        out[mod] = {
            "B_pad": B_pad.to(device=device, dtype=dtype).contiguous(),
            "u_hat": u_hat.to(device=device, dtype=dtype).contiguous(),
            "a_hat": a_hat.to(device=device, dtype=dtype).contiguous(),
            "diag": diag,
        }
    return out


# ===== Fast Processor =====
class VisualAttentionProcess(nn.Module):
    def __init__(self, module_name=None, atten_type="original", params=None, record=False, beta=0.5, tau=0.1):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type  # "original" | "erase"
        self.params = params  # dict from build_cora_params for this module
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
        self.params = params  # fast precomputed tensors for this module
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
            # self-attn or non-CLIP length: bypass edits
            hidden_states = torch.bmm(attn_probs, value)
        else:
            # --- Patch 1: erase path first; recording does NOT skip edits ---
            if self.params is not None and self.atten_type == "erase":
                # Optionally record raw values for visualization
                if self.record:
                    heads = attn.heads
                    batch = value.size(0) // heads
                    self.records["values"][self.module_name] = (value.detach(), heads, int(batch))

                # === FAST CORA (unchanged math) ===
                heads_cur = attn.heads
                d_head = value.size(2)
                batch_cur = max(1, value.size(0) // heads_cur)

                # Fetch precomputed params
                B_pad = self.params["B_pad"]  # [S,HD,K]
                u_hat = self.params["u_hat"]  # [S,HD]
                a_hat = self.params["a_hat"]  # [S,HD]

                # Reshape value to [B,H,S,D] -> [B,S,HD]
                v_bhsd = value.view(batch_cur, heads_cur, seq_len, d_head)
                v_flat = v_bhsd.permute(0, 2, 1, 3).contiguous().view(batch_cur, seq_len, heads_cur * d_head)

                # Project onto preserve subspace: v_pres = B (B^T v)
                if B_pad.size(2) > 0 and torch.count_nonzero(B_pad).item() > 0:
                    coeffs = torch.einsum('bsh,shk->bsk', v_flat, B_pad)
                    v_pres = torch.einsum('bsk,shk->bsh', coeffs, B_pad)
                else:
                    v_pres = torch.zeros_like(v_flat)

                v_free = v_flat - v_pres  # [B,S,HD]
                t = torch.einsum('bsh,sh->bs', v_free, u_hat)  # [B,S]
                denom = torch.linalg.norm(v_free, dim=2) + 1e-8
                mask = (t.abs() / denom) >= self.tau  # [B,S]

                # Lightweight runtime probes (only if recording)
                if self.record:
                    gate_frac = mask.float().mean()
                    avg_abs_t = t.abs().mean()
                    avg_beta_abs_t = (self.beta * t.abs()).mean()
                    v_flat_norm = torch.linalg.norm(v_flat, dim=2) + 1e-8  # [B,S]
                    pres_ratio = (torch.linalg.norm(v_pres, dim=2) / v_flat_norm).mean()
                    free_ratio = (torch.linalg.norm(v_free, dim=2) / v_flat_norm).mean()

                    entry = {
                        "module": self.module_name,
                        "gate_frac": float(gate_frac.detach().cpu().item()),
                        "avg_abs_t": float(avg_abs_t.detach().cpu().item()),
                        "avg_beta_abs_t": float(avg_beta_abs_t.detach().cpu().item()),
                        "pres_ratio": float(pres_ratio.detach().cpu().item()),
                        "free_ratio": float(free_ratio.detach().cpu().item()),
                    }
                    mask_sum = mask.float().sum().clamp_min(1.0)  # avoid divide-by-zero
                    avg_beta_abs_t_masked = (self.beta * t.abs() * mask.float()).sum() / mask_sum
                    entry["avg_beta_abs_t_masked"] = float(avg_beta_abs_t_masked.detach().cpu().item())

                    if "probe" not in self.records:
                        self.records["probe"] = []
                    self.records["probe"].append(entry)

                # Apply CORA edit
                v_free_new = v_free - t.unsqueeze(-1) * u_hat.unsqueeze(0) + self.beta * t.unsqueeze(-1) * a_hat.unsqueeze(0)
                v_free_out = torch.where(mask.unsqueeze(-1), v_free_new, v_free)
                v_new_flat = v_pres + v_free_out  # [B,S,HD]

                # Back to [B*H,S,D]
                v_new = v_new_flat.view(batch_cur, seq_len, heads_cur, d_head).permute(0, 2, 1, 3).contiguous()
                value = v_new.view(batch_cur * heads_cur, seq_len, d_head)

            else:
                # Non-erase path; optionally record for visualization
                if self.record:
                    heads = attn.heads
                    batch = value.size(0) // heads
                    self.records["values"][self.module_name] = (value.detach(), heads, int(batch))

            # common attention output
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
def set_attenprocessor(unet, atten_type="original", params=None, record=False, beta=0.5, tau=0.1, only_cross: bool = True):
    for name, m in unet.named_modules():
        if name.endswith("attn2") or (not only_cross and name.endswith("attn1")):
            cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
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


# ===== Diffusion loop (unchanged math, minimal overhead) =====
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
                    # existing path
                    for k, v in proc.records.get("values", {}).items():
                        if k not in visualize_map["values"]:
                            visualize_map["values"][k] = v
                    # new: collect probes
                    if "probe" in proc.records:
                        if "probe" not in visualize_map:
                            visualize_map["probe"] = {}
                        if proc.module_name not in visualize_map["probe"]:
                            visualize_map["probe"][proc.module_name] = []
                        visualize_map["probe"][proc.module_name].extend(proc.records["probe"])
                        proc.records["probe"].clear()  # avoid growth


        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map) if record else latents


# ===== Batch decode for speed =====
@torch.no_grad()
def decode_latents_batch(vae, latents_list: List[torch.Tensor]) -> List[Image.Image]:
    if len(latents_list) == 0:
        return []
    latents = torch.stack(latents_list, dim=0)
    imgs = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    # imgs: [B, 3, H, W] in [-1,1] or [0,1] depending on VAE; rely on process_img from utils
    out = [process_img(img) for img in imgs]
    return out


# ===== Main =====
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="")
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
    # Performance toggles
    parser.add_argument("--xformers", action="store_true", help="Enable xFormers mem-efficient attention if available")
    parser.add_argument("--compile", action="store_true", help="torch.compile the UNet (PyTorch 2.0+)")
    parser.add_argument("--tf32", action="store_true", help="Allow TF32 matmul on Ampere+")
    parser.add_argument("--fp32_linalg", action="store_true", help="Keep precompute in fp32 (default) but run edits in model dtype")
    parser.add_argument("--probe", action="store_true", help="Log lightweight diagnostics during erase")
    args = parser.parse_args()

    assert args.num_samples >= args.batch_size
    bs = args.batch_size
    mode_list = args.mode.replace(" ", "").split(",")
    concept_list = [s.strip() for s in args.contents.split(",") if s.strip()]
    assert args.target_concept.strip() != "", "Provide --target_concept"
    assert args.anchor_concept.strip() != "", "Provide --anchor_concept"

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
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    if args.compile:
        try:
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
        except Exception:
            pass

    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae

    # ---- Encodings ----
    device, dtype = pipe.device, pipe.unet.dtype
    uncond_encoding = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    # ---- Record maps (once) ----
    cora_params = None
    if "erase" in mode_list:
        tgt_map = record_concept_maps(unet, pipe, args.target_concept.strip(), args.guidance_scale)["values"]
        anc_map = record_concept_maps(unet, pipe, args.anchor_concept.strip(), args.guidance_scale)["values"]
        pres_maps = []
        preserve_list = [s.strip() for s in args.preserve_concepts.split(",") if s.strip()]
        for pc in preserve_list:
            pres_maps.append(record_concept_maps(unet, pipe, pc, args.guidance_scale)["values"])
        records_bundle = {"values": {"target": tgt_map, "anchor": anc_map, "preserve": pres_maps}}
        cora_params = build_cora_params(records_bundle, device, dtype)
        if args.probe:
            # One-time build-time sanity: any anchor collapse?
            cos_list, collapse_list = [], []
            for mod, p in cora_params.items():
                d = p.get("diag", {})
                if "cos_u_a_mean" in d: cos_list.append(d["cos_u_a_mean"])
                if "a_perp_small_frac" in d: collapse_list.append(d["a_perp_small_frac"])
            if cos_list:
                print(f"[CORA-PRECOMPUTE] mean cos(u,a_def)={sum(cos_list)/len(cos_list):.3f}  |  a_perp small frac={sum(collapse_list)/len(collapse_list):.2%}")


        if args.debug:
            print("[DEBUG] Modules prepared:", len(cora_params))

    # ---- Sampling ----
    seed_everything(args.seed, True)
    prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]

    # We'll toggle processors right before each diffusion call (no duplicate UNet refs).

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
                        unet, atten_type="erase", params=cora_params,
                        record=args.probe, beta=args.beta, tau=args.tau, only_cross=True
                    )
                    Images["erase"] = diffusion(
                        unet=unet,
                        scheduler=pipe.scheduler,
                        latents=latents,
                        start_timesteps=0,
                        text_embeddings=txt,
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        record=args.probe,
                        desc=f"{prompt} | CORA erase",
                    )
                    if args.probe and isinstance(Images["erase"], tuple):
                        _, viz = Images["erase"]
                        # Aggregate across modules
                        all_rows = []
                        for mod, rows in viz.get("probe", {}).items():
                            if not rows: 
                                continue
                            # average over steps for this module
                            gate = sum(r["gate_frac"] for r in rows) / len(rows)
                            inj  = sum(r["avg_beta_abs_t"] for r in rows) / len(rows)
                            pres = sum(r["pres_ratio"] for r in rows) / len(rows)
                            free = sum(r["free_ratio"] for r in rows) / len(rows)
                            all_rows.append((mod, gate, inj, pres, free))

                        # Global means (quick read)
                        if all_rows:
                            g_mean = sum(r[1] for r in all_rows) / len(all_rows)
                            inj_m  = sum(r[2] for r in all_rows) / len(all_rows)
                            pres_m = sum(r[3] for r in all_rows) / len(all_rows)
                            free_m = sum(r[4] for r in all_rows) / len(all_rows)
                            print(f"[CORA-PROBE] gate%={100*g_mean:.1f}  |  E[|β·t|]={inj_m:.4f}  |  E[‖v_pres‖/‖v_flat‖]={pres_m:.3f}  |  E[‖v_free‖/‖v_flat‖]={free_m:.3f}")
                            # --- Per-resolution probe (down/mid/up) ---
                            def _grp(modname: str) -> str:
                                # NOTE: match substrings without a leading dot
                                if "down_blocks" in modname: return "down"
                                if "mid_block"  in modname:  return "mid"
                                if "up_blocks"  in modname:  return "up"
                                return "other"

                            from collections import defaultdict
                            agg = defaultdict(lambda: {"gate":0.0,"inj":0.0,"pres":0.0,"free":0.0,"n":0})

                            for mod, gate, inj, pres, free in all_rows:
                                g = _grp(mod)
                                agg[g]["gate"] += gate; agg[g]["inj"]  += inj
                                agg[g]["pres"] += pres; agg[g]["free"] += free
                                agg[g]["n"]    += 1

                            def _fmt(g):
                                d = agg[g]; n = max(d["n"],1)
                                return f"{100*d['gate']/n:.1f}% | inj={d['inj']/n:.3f} | pres={d['pres']/n:.3f} | free={d['free']/n:.3f}"

                            print("[CORA-PROBE-BY-RES] down:", _fmt("down"), " | mid:", _fmt("mid"), " | up:", _fmt("up"))
                            print("[CORA-PROBE-BY-RES] counts: down", agg["down"]["n"], "| mid", agg["mid"]["n"], "| up", agg["up"]["n"], "| other", agg["other"]["n"])


                            # --- Top modules (helpful to see if only early layers fire) ---
                            top_by_gate = sorted(all_rows, key=lambda r: r[1], reverse=True)[:5]
                            top_by_inj  = sorted(all_rows, key=lambda r: r[2], reverse=True)[:5]
                            print("[TOP gate]")
                            for r in top_by_gate:
                                print("  ", r[0], f"{100*r[1]:.1f}%")
                            print("[TOP inj ]")
                            for r in top_by_inj:
                                print("  ", r[0], f"{r[2]:.3f}")

                            # Optional: flag suspicious situations once
                            if g_mean < 0.05:
                                print("  ↳ Low gating: τ likely too high OR u_hat misaligned for these prompts.")
                            if inj_m < 1e-3:
                                print("  ↳ Anchor injection weak: |t| small against u_hat or β too low relative to scale.")
                            if pres_m > 0.7:
                                print("  ↳ Preserve projector is dominating; anchor/target being deflated too much.")


                # ---- Save ----
                save_path = os.path.join(args.save_root, args.target_concept.replace(", ", "_"), concept)
                for mode in mode_list:
                    os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1:
                    os.makedirs(os.path.join(save_path, "combine"), exist_ok=True)

                # Unwrap (latents, viz) tuples into just latents before decoding
                for k, v in list(Images.items()):
                    if isinstance(v, tuple):
                        Images[k] = v[0]

                # Batch decode for each mode
                decoded = {name: decode_latents_batch(vae, [img for img in img_list]) for name, img_list in Images.items()}

                def combine_h(ims: List[Image.Image]) -> Image.Image:
                    widths, heights = zip(*(im.size for im in ims))
                    canvas = Image.new("RGB", (sum(widths), max(heights)))
                    x = 0
                    for im in ims:
                        canvas.paste(im, (x, 0))
                        x += im.size[0]
                    return canvas

                for idx in range(len(decoded[mode_list[0]])):
                    fname = re.sub(r"[^\w\s]", "", prompt).replace(", ", "_") + f"_{int(idx + bs * i)}.png"
                    row = []
                    for mode in mode_list:
                        decoded[mode][idx].save(os.path.join(save_path, mode, fname))
                        row.append(decoded[mode][idx])
                    if len(mode_list) > 1:
                        combine_h(row).save(os.path.join(save_path, "combine", fname))


if __name__ == "__main__":
    main()
