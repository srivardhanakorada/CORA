########################################### main_cora.py #################################################
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

        out[mod] = {
            "B_pad": B_pad.to(device=device, dtype=dtype).contiguous(),
            "u_hat": u_hat.to(device=device, dtype=dtype).contiguous(),
            "a_hat": a_hat.to(device=device, dtype=dtype).contiguous(),
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
            if self.record:
                heads = attn.heads
                batch = value.size(0) // heads
                self.records["values"][self.module_name] = (value.detach(), heads, int(batch))
            elif self.params is not None and self.atten_type == "erase":
                # === FAST CORA ===
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

                v_free_new = v_free - t.unsqueeze(-1) * u_hat.unsqueeze(0) + self.beta * t.unsqueeze(-1) * a_hat.unsqueeze(0)
                v_free_out = torch.where(mask.unsqueeze(-1), v_free_new, v_free)
                v_new_flat = v_pres + v_free_out  # [B,S,HD]

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
                    for k, v in proc.records.get("values", {}).items():
                        if k not in visualize_map["values"]:
                            visualize_map["values"][k] = v

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
                    set_attenprocessor(unet, atten_type="erase", params=cora_params, record=False, beta=args.beta, tau=args.tau, only_cross=True)
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

                # ---- Save ----
                save_path = os.path.join(args.save_root, args.target_concept.replace(", ", "_"), concept)
                for mode in mode_list:
                    os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1:
                    os.makedirs(os.path.join(save_path, "combine"), exist_ok=True)

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
