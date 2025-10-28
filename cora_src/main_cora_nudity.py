# ======================================= #
#  CORA NSFW Eraser - CSV (I2P-style)    #
#  Single-Concept, Row-wise Processing    #
#  Flat saves: original/erase/…/{idx}.png #
#  CSV columns (examples):                 #
#    prompt,categories,...,               #
#    sd_seed,sd_guidance_scale,           #
#    sd_image_width,sd_image_height,      #
#    sd_model                             #
# ======================================= #
import os, re, csv, json, copy, argparse
from typing import Dict, List, Tuple, Optional

from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

# utils.py must provide:
# - seed_everything, get_token, get_textencoding, get_eot_idx, get_spread_embedding, process_img
from utils import *

# ---------------- Helpers ---------------- #
def _to_bh_sd(x: torch.Tensor, batch: int, heads: int) -> torch.Tensor:
    return x.view(batch, heads, x.size(1), x.size(2))

def _hsd_to_flat_token(x_hsd: torch.Tensor) -> torch.Tensor:
    H, S, D = x_hsd.shape
    return x_hsd.permute(1, 0, 2).contiguous().view(S, H * D)

def _sanitize_filename(s: str) -> str:
    s = re.sub(r"[^\w\s\-.,()]", "", s, flags=re.UNICODE)
    s = re.sub(r"\s+", "_", s.strip())
    return s[:180]

# ---- HF repo resolution (aliases + local paths) ---- #
_KNOWN_ALIASES = {
    "stable-diffusion-v1-4": "CompVis/stable-diffusion-v1-4",
    "sd-v1-4": "CompVis/stable-diffusion-v1-4",
    "sd14": "CompVis/stable-diffusion-v1-4",
    "stable-diffusion-v1-5": "runwayml/stable-diffusion-v1-5",
    "sd-v1-5": "runwayml/stable-diffusion-v1-5",
    "sd15": "runwayml/stable-diffusion-v1-5",
    "sd-turbo": "stabilityai/sd-turbo",
}
def _resolve_repo_or_path(model_or_repo: str, fallback: str) -> str:
    m = (model_or_repo or "").strip()
    if m and os.path.isdir(m):
        return m
    if "/" in m and len(m.split("/")) == 2:
        return m
    if m in _KNOWN_ALIASES:
        return _KNOWN_ALIASES[m]
    return fallback or m

# ------------- Recording --------------- #
@torch.no_grad()
def record_concept_maps(unet, pipe, concept_text: str, guidance_scale: float):
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    device, dtype = pipe.device, pipe.unet.dtype

    enc   = get_textencoding(get_token(concept_text, tokenizer), text_encoder)
    idx   = get_eot_idx(get_token(concept_text, tokenizer))
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

# --------- Build CORA (anchor selection) --------- #
@torch.no_grad()
def build_cora_params_select_anchor(records_bundle: Dict[str, Dict], device, dtype):
    vals = records_bundle["values"]
    tgt_map = vals["target"]
    anc_list_maps = vals["anchors"]
    pres_list_maps = vals.get("preserve", [])
    M = len(anc_list_maps)
    assert M >= 1

    module_sets = [set(a.keys()) for a in anc_list_maps]
    modules = sorted(set(tgt_map.keys()).intersection(*module_sets))
    K = len(pres_list_maps)

    per_anchor_score_accum = torch.zeros(M, dtype=torch.float32, device=device)
    per_anchor_count = torch.zeros(M, dtype=torch.float32, device=device)
    stash_by_mod = {}

    for mod in modules:
        target_value, heads_t, batch_t = tgt_map[mod]
        tar_bhsd = _to_bh_sd(target_value, batch_t, heads_t)
        tar_idx = 1 if tar_bhsd.size(0) > 1 else 0
        tar_hsd = tar_bhsd[tar_idx]
        tar_flat = _hsd_to_flat_token(tar_hsd).to(torch.float32)
        S, HD = tar_flat.shape

        B_pad = torch.zeros(S, HD, K, device=device, dtype=torch.float32)
        if K > 0:
            for k_idx, m in enumerate(pres_list_maps):
                if mod in m:
                    p_val, h_p, b_p = m[mod]
                    p_bhsd = _to_bh_sd(p_val, b_p, h_p)
                    p_idx = 1 if p_bhsd.size(0) > 1 else 0
                    p_hsd = p_bhsd[p_idx]
                    B_pad[:, :, k_idx] = _hsd_to_flat_token(p_hsd).to(torch.float32)
            B_ortho = torch.zeros_like(B_pad)
            for j in range(S):
                pj = B_pad[j]
                if not torch.allclose(pj.abs().sum(), torch.tensor(0.0, device=device)):
                    q, _ = torch.linalg.qr(pj, mode='reduced')
                    r_rank = min(q.shape[1], K)
                    B_ortho[j, :, :r_rank] = q[:, :r_rank]
            B_pad = B_ortho

        def _deflate(vec_flat):
            if K == 0: return vec_flat
            c = torch.einsum('sh,shk->sk', vec_flat, B_pad)
            proj = torch.einsum('shk,sk->sh', B_pad, c)
            return vec_flat - proj

        u_def = _deflate(tar_flat)
        u_hat = u_def / (torch.linalg.norm(u_def, dim=1, keepdim=True) + 1e-8)

        module_scores = []
        module_ahats = []
        for a_map in anc_list_maps:
            anc_value, heads_a, batch_a = a_map[mod]
            anc_bhsd = _to_bh_sd(anc_value, batch_a, heads_a)
            anc_idx = 1 if anc_bhsd.size(0) > 1 else 0
            anc_hsd = anc_bhsd[anc_idx]
            anc_flat = _hsd_to_flat_token(anc_hsd).to(torch.float32)

            a_def = _deflate(anc_flat)
            a_perp = a_def - (torch.einsum('sh,sh->s', u_hat, a_def).unsqueeze(1)) * u_hat
            e = (torch.linalg.norm(a_perp, dim=1) ** 2).mean()
            a_hat = a_perp / (torch.linalg.norm(a_perp, dim=1, keepdim=True) + 1e-8)
            module_scores.append(e)
            module_ahats.append(a_hat)

        per_anchor_score_accum += torch.stack(module_scores)
        per_anchor_count += 1.0

        stash_by_mod.setdefault(mod, {})
        stash_by_mod[mod]["B_pad_fp32"] = B_pad
        stash_by_mod[mod]["u_hat_fp32"] = u_hat
        stash_by_mod[mod]["a_hat_candidates_fp32"] = module_ahats

    mean_scores = per_anchor_score_accum / per_anchor_count.clamp_min(1.0)
    chosen_idx = int(torch.argmax(mean_scores).item())
    scores = [float(x) for x in mean_scores.tolist()]

    final_params = {}
    for mod, stash in stash_by_mod.items():
        final_params[mod] = {
            "B_pad": stash["B_pad_fp32"].to(device=device, dtype=dtype).contiguous(),
            "u_hat": stash["u_hat_fp32"].to(device=device, dtype=dtype).contiguous(),
            "a_hat": stash["a_hat_candidates_fp32"][chosen_idx].to(device=device, dtype=dtype).contiguous(),
        }
    return final_params, chosen_idx, scores

# -------------- Processor -------------- #
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
            hidden_states = hidden_states.view(b, c, h*w).transpose(1, 2)

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

                B_pad = self.params["B_pad"]
                u_hat = self.params["u_hat"]
                a_hat = self.params["a_hat"]

                v_bhsd = value.view(batch_cur, heads_cur, seq_len, d_head)
                v_flat = v_bhsd.permute(0, 2, 1, 3).contiguous().view(batch_cur, seq_len, heads_cur * d_head)

                # preserve projection
                if B_pad.size(2) > 0 and torch.count_nonzero(B_pad).item() > 0:
                    coeffs = torch.einsum('bsh,shk->bsk', v_flat, B_pad)
                    v_pres = torch.einsum('bsk,shk->bsh', coeffs, B_pad)
                else:
                    v_pres = torch.zeros_like(v_flat)

                v_free = v_flat - v_pres
                t = torch.einsum('bsh,sh->bs', v_free, u_hat)
                denom = torch.linalg.norm(v_free, dim=2) + 1e-8
                mask = (t.abs() / denom) >= self.tau

                v_free_new = v_free - t.unsqueeze(-1) * u_hat.unsqueeze(0) + self.beta * t.unsqueeze(-1) * a_hat.unsqueeze(0)
                v_free_out = torch.where(mask.unsqueeze(-1), v_free_new, v_free)
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

# ------------ Install processors ------------ #
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

# ----------------- Diffusion ----------------- #
@torch.no_grad()
def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0, guidance_scale=7.5, record=False, desc=None):
    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample
    return latents

@torch.no_grad()
def decode_latents_batch(vae, latents_tensor: torch.Tensor) -> List[Image.Image]:
    imgs = vae.decode(latents_tensor / vae.config.scaling_factor, return_dict=False)[0]
    return [process_img(img) for img in imgs]

def combine_h(ims: List[Image.Image]) -> Image.Image:
    widths, heights = zip(*(im.size for im in ims))
    canvas = Image.new("RGB", (sum(widths), max(heights)))
    x = 0
    for im in ims:
        canvas.paste(im, (x, 0))
        x += im.size[0]
    return canvas

# ---------------- CSV Loader ---------------- #
def read_concepts_csv(csv_path: str, limit: Optional[int] = None) -> List[dict]:
    rows: List[dict] = []
    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if limit is not None and len(rows) >= limit:
                break
            norm = {k.strip(): (v.strip() if isinstance(v, str) else v) for k, v in r.items()}
            if not norm.get("prompt"):
                continue
            rows.append(norm)
    return rows

# -------------------- Main ------------------- #
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save_path", type=str, required=True)
    ap.add_argument("--concepts_csv", type=str, required=True)

    ap.add_argument("--mode", type=str, default="original,erase")
    ap.add_argument("--erase_type", type=str, default="nsfw")
    ap.add_argument("--num_samples", type=int, default=0, help="Cap total rows processed (0 = all)")
    ap.add_argument("--batch_size", type=int, default=1)  # row-wise pipeline; unused for batching rows
    ap.add_argument("--total_timesteps", type=int, default=30)

    # CORA knobs (single-concept)
    ap.add_argument("--target_concept", type=str, required=True)
    ap.add_argument("--preserve_concepts", type=str, default="")
    ap.add_argument("--single_anchor_mode", action="store_true")
    ap.add_argument("--single_anchor_text", type=str, default="a person")
    ap.add_argument("--anchor_pool", type=str, default="")
    ap.add_argument("--beta", type=float, default=0.5)
    ap.add_argument("--tau", type=float, default=0.1)

    # Defaults (row can override)
    ap.add_argument("--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    ap.add_argument("--guidance_scale", type=float, default=7.5)
    ap.add_argument("--width", type=int, default=512)
    ap.add_argument("--height", type=int, default=512)
    ap.add_argument("--seed", type=int, default=0)

    # Loader behavior
    ap.add_argument("--local_only", action="store_true", help="Load models only from local cache/paths (no Hub)")

    # Performance toggles
    ap.add_argument("--xformers", action="store_true")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--tf32", action="store_true")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    mode_list = args.mode.replace(" ", "").split(",")
    os.makedirs(args.save_path, exist_ok=True)
    # Flat folders
    for m in mode_list:
        os.makedirs(os.path.join(args.save_path, m), exist_ok=True)
    if len(mode_list) > 1:
        os.makedirs(os.path.join(args.save_path, "combine"), exist_ok=True)

    manifest_path = os.path.join(args.save_path, "manifest.jsonl")
    manifest_f = open(manifest_path, "a", encoding="utf-8")

    if args.tf32:
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.set_float32_matmul_precision("medium")
        except Exception:
            pass

    # Pipeline cache
    pipe_cache: Dict[str, DiffusionPipeline] = {}

    def get_pipe(model_or_repo: str) -> DiffusionPipeline:
        resolved = _resolve_repo_or_path(model_or_repo, args.sd_ckpt)
        if resolved not in pipe_cache:
            try:
                pipe = DiffusionPipeline.from_pretrained(
                    resolved, safety_checker=None, torch_dtype=torch.float16,
                    local_files_only=args.local_only
                ).to("cuda")
            except Exception as e:
                if not args.local_only:
                    pipe = DiffusionPipeline.from_pretrained(
                        resolved, safety_checker=None, torch_dtype=torch.float16,
                        local_files_only=True
                    ).to("cuda")
                else:
                    raise RuntimeError(f"Failed to load model '{resolved}' with local_only={args.local_only}: {repr(e)}")
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            if args.xformers:
                try: pipe.enable_xformers_memory_efficient_attention()
                except Exception: pass
            if args.compile:
                try: pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=False)
                except Exception: pass
            pipe_cache[resolved] = pipe
        return pipe_cache[resolved]

    # Load CSV
    rows = read_concepts_csv(args.concepts_csv, None if args.num_samples <= 0 else args.num_samples)
    assert len(rows) > 0, "CSV has no usable rows."

    # Precompute CORA params per active pipe (recompute on model swap)
    current_model_for_cora = None
    cora_params = None
    chosen_anchor = None
    anchor_names: List[str] = []

    def ensure_cora_for(pipe: DiffusionPipeline):
        nonlocal current_model_for_cora, cora_params, chosen_anchor, anchor_names
        if current_model_for_cora is pipe:
            return

        device, dtype = pipe.device, pipe.unet.dtype
        unet = pipe.unet

        tgt_map = record_concept_maps(unet, pipe, args.target_concept.strip(), args.guidance_scale)["values"]
        preserve_list = [s.strip() for s in args.preserve_concepts.split(",") if s.strip()]
        pres_maps = [record_concept_maps(unet, pipe, pc, args.guidance_scale)["values"] for pc in preserve_list]

        if args.single_anchor_mode:
            anc_map = record_concept_maps(unet, pipe, args.single_anchor_text.strip(), args.guidance_scale)["values"]
            bundle = {"values": {"target": tgt_map, "anchors": [anc_map], "preserve": pres_maps}}
            cora_params, chosen_idx, scores = build_cora_params_select_anchor(bundle, device, dtype)
            chosen_anchor = args.single_anchor_text.strip()
            anchor_names = [chosen_anchor]
        else:
            pool = [s.strip() for s in args.anchor_pool.split(",") if s.strip()]
            if len(pool) == 0:
                pool = [args.single_anchor_text.strip()]  # fallback
            anc_maps = [record_concept_maps(unet, pipe, a_txt, args.guidance_scale)["values"] for a_txt in pool]
            bundle = {"values": {"target": tgt_map, "anchors": anc_maps, "preserve": pres_maps}}
            cora_params, chosen_idx, scores = build_cora_params_select_anchor(bundle, device, dtype)
            chosen_anchor = pool[chosen_idx]
            anchor_names = pool

        current_model_for_cora = pipe

    # ===== Row-wise loop =====
    for idx, r in enumerate(rows):
        # per-row overrides (safe fallbacks)
        model  = r.get("sd_model") or args.sd_ckpt
        width  = int(r.get("sd_image_width") or args.width)
        height = int(r.get("sd_image_height") or args.height)
        gscale = float(r.get("sd_guidance_scale") or args.guidance_scale)
        seed   = int(r.get("sd_seed") or args.seed)
        prompt = r["prompt"]

        pipe = get_pipe(model)
        ensure_cora_for(pipe)

        tokenizer, text_encoder, vae = pipe.tokenizer, pipe.text_encoder, pipe.vae
        device, dtype = pipe.device, pipe.unet.dtype
        unet = pipe.unet

        # paired latents per row
        seed_everything(seed, True)
        h_lat, w_lat = height // 8, width // 8
        latents = torch.randn(1, 4, h_lat, w_lat, device=device, dtype=dtype)

        # text encodings
        uncond = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)
        cond   = get_textencoding(get_token(prompt, tokenizer), text_encoder).to(device, dtype=dtype)
        txt    = torch.cat([uncond, cond], dim=0)

        decoded_by_mode: Dict[str, Image.Image] = {}
        # original
        if "original" in mode_list:
            set_attenprocessor(unet, atten_type="original", params=None, record=False, only_cross=True)
            lat_out = diffusion(unet, pipe.scheduler, latents.clone(), txt, args.total_timesteps,
                                guidance_scale=gscale, desc=f"Row {idx} | original")
            decoded_by_mode["original"] = decode_latents_batch(vae, lat_out)[0]

        # erase
        if "erase" in mode_list:
            set_attenprocessor(unet, atten_type="erase", params=cora_params, record=False,
                               beta=args.beta, tau=args.tau, only_cross=True)
            lat_out = diffusion(unet, pipe.scheduler, latents.clone(), txt, args.total_timesteps,
                                guidance_scale=gscale, desc=f"Row {idx} | erase")
            decoded_by_mode["erase"] = decode_latents_batch(vae, lat_out)[0]

        # ---- Flat saves: {idx}.png in each mode ----
        file_stubs = {}
        row_imgs   = []
        for m in mode_list:
            out_path = os.path.join(args.save_path, m, f"{idx}.png")
            decoded_by_mode[m].save(out_path)
            file_stubs[m] = out_path
            row_imgs.append(decoded_by_mode[m])

        if len(mode_list) > 1:
            comb = combine_h(row_imgs)
            cpath = os.path.join(args.save_path, "combine", f"{idx}.png")
            comb.save(cpath)
            file_stubs["combine"] = cpath

        # manifest entry
        rec = {
            "index": idx,
            "prompt": prompt,
            "row": {
                "hard": r.get("hard"),
                "inappropriate_percentage": r.get("inappropriate_percentage"),
                "nudity_percentage": r.get("nudity_percentage"),
                "q16_percentage": r.get("q16_percentage"),
                "sd_safety_percentage": r.get("sd_safety_percentage"),
                "prompt_toxicity": r.get("prompt_toxicity"),
                "lexica_url": r.get("lexica_url"),
                "categories_raw": r.get("categories"),
            },
            "sd": {
                "model": (r.get("sd_model") or args.sd_ckpt),
                "seed": seed,
                "guidance_scale": gscale,
                "width": width,
                "height": height,
            },
            "cora": {
                "target_concept": args.target_concept,
                "chosen_anchor": (args.single_anchor_text if args.single_anchor_mode else (chosen_anchor or args.single_anchor_text)),
                "preserve_concepts": [s for s in args.preserve_concepts.split(",") if s.strip()],
                "beta": args.beta,
                "tau": args.tau,
            },
            "paths": file_stubs,
        }
        manifest_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        manifest_f.flush()

    manifest_f.close()
    print(f"[CORA] Done. Images -> {args.save_path} | Manifest -> {manifest_path}")

if __name__ == "__main__":
    main()
