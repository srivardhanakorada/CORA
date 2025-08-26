# main_cora_single.py
import os
import re
import copy
import argparse
from PIL import Image
from tqdm import tqdm
import torch
from torch import nn
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler  # type: ignore

from template import template_dict
from utils import *  # get_token, get_textencoding, get_spread_embedding, process_img, seed_everything

# ==============================
# CORA Attention Processor
# ==============================

class VisualAttentionProcess(nn.Module):
    def __init__(
        self,
        module_name=None,
        atten_type="original",          # "original" | "erase"
        records_bundle=None,            # {"values":{"target":{module:(val,heads,batch)}, "anchor":{...}, "preserve":[{...}, ...]}}
        record=False,
        beta=0.5,
        tau=0.1,
        **kwargs,
    ):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.records_bundle = records_bundle
        self.record = record
        self.beta = beta
        self.tau = tau

    def __call__(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor")
        attn.processor = AttnProcessor(
            module_name=self.module_name,
            atten_type=self.atten_type,
            records_bundle=self.records_bundle,
            record=self.record,
            beta=self.beta,
            tau=self.tau,
        )
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)


class AttnProcessor:
    def __init__(
        self,
        module_name=None,
        atten_type="original",
        records_bundle=None,
        record=False,
        beta=0.5,
        tau=0.1,
    ):
        self.module_name = module_name
        self.atten_type = atten_type
        self.records_bundle = copy.deepcopy(records_bundle) if records_bundle else None
        self.record = record
        self.records = {"values": {}} if record else {}
        self.beta = beta
        self.tau = tau

    # ---------- shape helpers ----------
    @staticmethod
    def _to_bh_sd(x, batch, heads):
        # [B*H, S, D] -> [B, H, S, D]
        return x.view(batch, heads, x.size(1), x.size(2))

    @staticmethod
    def _bh_sd_to_flat_hsd(x_bhsd):
        # [B,H,S,D] -> list of B tensors [H,S,D]
        return [x_bhsd[b] for b in range(x_bhsd.size(0))]

    @staticmethod
    def _hsd_to_flat_token(x_hsd):
        # [H,S,D] -> [S, H*D]
        H, S, D = x_hsd.shape
        return x_hsd.permute(1, 0, 2).contiguous().view(S, H * D)

    @staticmethod
    def _flat_token_to_hsd(x_flat, heads, d_head):
        # [S, H*D] -> [H, S, D]
        S, HD = x_flat.shape
        assert HD % heads == 0
        D = d_head if d_head is not None else HD // heads
        return x_flat.view(S, heads, D).permute(1, 0, 2).contiguous()

    # ---------- linear algebra ----------
    @staticmethod
    def _project_onto_span(preserve_vecs, v):
        """
        preserve_vecs: list of [D] vectors (can be empty)
        v: [D] vector
        returns P(v) in column span(preserve_vecs)
        """
        if len(preserve_vecs) == 0:
            return torch.zeros_like(v)
        # Build V: [D, K]
        V = torch.stack(preserve_vecs, dim=1)  # [D, K]
        V32 = V.to(torch.float32)
        v32 = v.to(torch.float32)
        # Least squares: V c ≈ v  => projection = V c
        # Handles rank-deficiency robustly via lstsq/pinv
        try:
            c = torch.linalg.lstsq(V32, v32).solution  # [K]
            proj = (V32 @ c).to(v.dtype)
        except RuntimeError:
            # Fallback to pseudo-inverse
            proj = (V32 @ torch.linalg.pinv(V32) @ v32).to(v.dtype)
        return proj

    @staticmethod
    def _normalize(x, eps=1e-8):
        n = x.norm() + eps
        if torch.isfinite(n) and n > 0:
            return x / n
        return torch.zeros_like(x)

    # ---------- CORA math per token ----------
    def _cora_per_token(self, v, u, a, preserve_list):
        """
        v: [D] current value (flattened across heads)
        u: [D] target (flattened)
        a: [D] anchor (flattened)
        preserve_list: list of [D] preserve vectors for this token (flattened)
        Implements:
          P = projector onto span(preserve_list)
          u~=(I-P)u, a~=(I-P)a, v_pres=P v, v_free=v-v_pres
          u = u~/||u~||, a = (a~ - (u·a~)u); a = a/||a||
          t = u·v_free
          if |t|/||v_free|| < tau: return v
          v' = v_pres + (v_free - t u + beta t a)
        """
        dtype_in = v.dtype

        # Project onto preserve subspace (in float32 for numerical stability)
        v_pres = self._project_onto_span(preserve_list, v)

        u_def = u - self._project_onto_span(preserve_list, u)
        a_def = a - self._project_onto_span(preserve_list, a)

        u_hat = self._normalize(u_def.to(torch.float32)).to(dtype_in)
        a_tmp = (a_def.to(torch.float32) - (u_hat.to(torch.float32) @ a_def.to(torch.float32)) * u_hat.to(torch.float32))
        a_hat = self._normalize(a_tmp).to(dtype_in)

        v_free = (v - v_pres).to(torch.float32)
        t = (u_hat.to(torch.float32) @ v_free)  # scalar

        denom = v_free.norm() + 1e-8
        if torch.abs(t) / denom < self.tau:
            return v  # gate off

        v_free_new = (v_free - t * u_hat.to(torch.float32) + self.beta * t * a_hat.to(torch.float32)).to(dtype_in)
        return v_pres + v_free_new

    # ---------- main call ----------
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

        query = attn.head_to_batch_dim(query)   # [B*H, S, D]
        key   = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        attn_probs = attn.get_attention_scores(query, key, attention_mask)

        if encoder_hidden_states.shape[1] != 77:
            hidden_states = torch.bmm(attn_probs, value)
        else:
            if self.record:
                heads = attn.heads
                batch = value.size(0) // heads
                # store (value, heads, batch) keyed by exact module name
                self.records["values"][self.module_name] = (value.detach(), heads, batch)
            elif self.records_bundle is not None and self.atten_type == "erase":
                vals = self.records_bundle.get("values", {})
                tgt_map = vals.get("target", {})
                anc_map = vals.get("anchor", {})
                pres_list_maps = vals.get("preserve", [])

                rec_tgt = tgt_map.get(self.module_name, None)
                rec_anc = anc_map.get(self.module_name, None)

                # If we don't have both target and anchor for this module, leave it untouched
                if rec_tgt is not None and rec_anc is not None:
                    target_value, heads_t, batch_t = rec_tgt
                    anchor_value, heads_a, batch_a = rec_anc

                    # shape current
                    heads_cur = attn.heads
                    batch_cur = max(1, value.size(0) // heads_cur)
                    d_head = value.size(2)

                    # reshape recorded to [B,H,S,D] and pick the conditional sample (idx 1) if present
                    tar_bhsd = self._to_bh_sd(target_value, batch_t, heads_t)
                    anc_bhsd = self._to_bh_sd(anchor_value, batch_a, heads_a)
                    tar_idx = 1 if tar_bhsd.size(0) > 1 else 0
                    anc_idx = 1 if anc_bhsd.size(0) > 1 else 0
                    tar_hsd = tar_bhsd[tar_idx]  # [H,S,D]
                    anc_hsd = anc_bhsd[anc_idx]  # [H,S,D]

                    tar_flat_tokens = self._hsd_to_flat_token(tar_hsd)  # [S, H*D]
                    anc_flat_tokens = self._hsd_to_flat_token(anc_hsd)  # [S, H*D]

                    # prepare preserve list per token (each entry is list of vectors)
                    preserve_flat_per_token = [[] for _ in range(seq_len)]
                    for m in pres_list_maps:
                        rec_p = m.get(self.module_name, None)
                        if rec_p is None:
                            continue
                        p_val, h_p, b_p = rec_p
                        p_bhsd = self._to_bh_sd(p_val, b_p, h_p)
                        p_idx = 1 if p_bhsd.size(0) > 1 else 0
                        p_hsd = p_bhsd[p_idx]
                        p_flat_tokens = self._hsd_to_flat_token(p_hsd)  # [S, H*D]
                        for j in range(seq_len):
                            preserve_flat_per_token[j].append(p_flat_tokens[j].to(value.dtype))

                    # apply CORA per batch sample and per token
                    cur_bhsd = self._to_bh_sd(value, batch_cur, heads_cur)  # [B,H,S,D]
                    new_list = []
                    for bidx in range(batch_cur):
                        v_hsd = cur_bhsd[bidx]                              # [H,S,D]
                        v_flat_tokens = self._hsd_to_flat_token(v_hsd)      # [S,H*D]
                        out_tokens = []
                        for j in range(seq_len):
                            vj = v_flat_tokens[j]
                            uj = tar_flat_tokens[j]
                            aj = anc_flat_tokens[j]
                            pres_j = preserve_flat_per_token[j]
                            vj_new = self._cora_per_token(vj, uj, aj, pres_j)
                            out_tokens.append(vj_new.unsqueeze(0))
                        out_tokens = torch.cat(out_tokens, dim=0)           # [S,H*D]
                        v_new_hsd = self._flat_token_to_hsd(out_tokens, heads_cur, d_head)
                        new_list.append(v_new_hsd.unsqueeze(0))

                    new_bhsd = torch.cat(new_list, dim=0)                   # [B,H,S,D]
                    value = new_bhsd.view(batch_cur * heads_cur, seq_len, d_head)

            hidden_states = torch.bmm(attn_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual
        return hidden_states / attn.rescale_output_factor


def set_attenprocessor(
    unet,
    atten_type="original",
    records_bundle=None,
    record=False,
    beta=0.5,
    tau=0.1,
):
    for name, m in unet.named_modules():
        if name.endswith("attn2") or name.endswith("attn1"):
            cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
            m.set_processor(
                VisualAttentionProcess(
                    module_name=name,
                    atten_type=atten_type,
                    records_bundle=records_bundle,
                    record=record,
                    cross_attention_dim=cross_attention_dim,
                    beta=beta,
                    tau=tau,
                )
            )
    return unet

# ==============================
# Diffusion & Recording
# ==============================

def diffusion(
    unet,
    scheduler,
    latents,
    text_embeddings,
    total_timesteps,
    start_timesteps=0,
    guidance_scale=7.5,
    record=False,
    desc=None,
):
    visualize_map = {"values": {}} if record else {}
    scheduler.set_timesteps(total_timesteps)

    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)
        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if record:
            # store once per module; (value, heads, batch)
            for proc in unet.attn_processors.values():
                for k, v in proc.records["values"].items():
                    if k not in visualize_map["values"]:
                        visualize_map["values"][k] = v

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map) if record else latents


@torch.no_grad()
def record_concept_maps(unet, pipe, concept_text, guidance_scale):
    """
    Records attention value tensors once for a given concept (CFG with uncond+concept),
    returning {"values": {module_name: (value, heads, batch)}}.
    """
    tokenizer, text_encoder = pipe.tokenizer, pipe.text_encoder
    device, dtype = pipe.device, pipe.unet.dtype

    enc = get_textencoding(get_token(concept_text, tokenizer), text_encoder)
    idx = get_eot_idx(get_token(concept_text, tokenizer))
    spread = get_spread_embedding(enc, idx).to(device, dtype=dtype)

    uncond = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    rec_unet = set_attenprocessor(copy.deepcopy(unet), atten_type="original", record=True)
    _, rec_map = diffusion(
        unet=rec_unet,
        scheduler=pipe.scheduler,
        latents=torch.zeros(1, 4, 64, 64, device=device, dtype=dtype),
        text_embeddings=torch.cat([uncond, spread], dim=0),
        total_timesteps=1,
        start_timesteps=0,
        guidance_scale=guidance_scale,
        record=True,
        desc=f"Recording {concept_text}",
    )
    return rec_map


# ==============================
# Main
# ==============================

@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_root", type=str, default="")
    parser.add_argument("--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--mode", type=str, default="original,erase")  # original, erase (CORA)
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--total_timesteps", type=int, default=30)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)

    parser.add_argument("--erase_type", type=str, default="", help="instance, style, celebrity")
    parser.add_argument("--target_concept", type=str, default="")
    parser.add_argument("--anchor_concept", type=str, default="a man")
    parser.add_argument("--preserve_concepts", type=str, default="")  # comma-separated
    parser.add_argument("--contents", type=str, default="")

    parser.add_argument("--beta", type=float, default=0.5)  # CORA replacement strength
    parser.add_argument("--tau", type=float, default=0.1)   # CORA gate threshold
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    assert args.num_samples >= args.batch_size
    bs = args.batch_size
    mode_list = args.mode.replace(" ", "").split(",")
    concept_list = [s.strip() for s in args.contents.split(",") if s.strip()]
    assert args.target_concept.strip() != "", "Provide --target_concept"
    assert args.anchor_concept.strip() != "", "Provide --anchor_concept"

    # ---- Models ----
    pipe = DiffusionPipeline.from_pretrained(
        args.sd_ckpt, safety_checker=None, torch_dtype=torch.float16
    ).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae

    if "original" in mode_list: unet_original = copy.deepcopy(unet)
    if "erase" in mode_list:    unet_erase    = copy.deepcopy(unet)

    # ---- Encodings ----
    device, dtype = pipe.device, pipe.unet.dtype
    uncond_encoding = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    # ---- Record maps (once) ----
    records_bundle = None
    if "erase" in mode_list:
        tgt_map = record_concept_maps(unet, pipe, args.target_concept.strip(), args.guidance_scale)["values"]
        anc_map = record_concept_maps(unet, pipe, args.anchor_concept.strip(), args.guidance_scale)["values"]
        pres_maps = []
        preserve_list = [s.strip() for s in args.preserve_concepts.split(",") if s.strip()]
        for pc in preserve_list:
            pres_maps.append(record_concept_maps(unet, pipe, pc, args.guidance_scale)["values"])

        if args.debug:
            print("[DEBUG] Recorded modules:")
            print("  target :", sorted(list(tgt_map.keys()))[:3], "...", len(tgt_map))
            print("  anchor :", sorted(list(anc_map.keys()))[:3], "...", len(anc_map))
            for idx, pm in enumerate(pres_maps):
                print(f"  preserve[{idx}] :", sorted(list(pm.keys()))[:3], "...", len(pm))

        records_bundle = {"values": {"target": tgt_map, "anchor": anc_map, "preserve": pres_maps}}

    # ---- Sampling ----
    seed_everything(args.seed, True)
    prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]

    for i in range(int(args.num_samples // bs)):
        latents = torch.randn(bs, 4, 64, 64, device=device, dtype=dtype)

        for concept, prompts in zip(concept_list, prompt_list):
            for prompt in prompts:
                Images = {}
                encoding = get_textencoding(get_token(prompt, tokenizer), text_encoder).to(device, dtype=dtype)

                if "original" in mode_list:
                    Images["original"] = diffusion(
                        unet=unet_original,
                        scheduler=pipe.scheduler,
                        latents=latents,
                        start_timesteps=0,
                        text_embeddings=torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0),
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        desc=f"{prompt} | original",
                    )

                if "erase" in mode_list:
                    unet_erase = set_attenprocessor(
                        unet_erase,
                        atten_type="erase",
                        records_bundle=copy.deepcopy(records_bundle),
                        beta=args.beta,
                        tau=args.tau,
                    )
                    Images["erase"] = diffusion(
                        unet=unet_erase,
                        scheduler=pipe.scheduler,
                        latents=latents,
                        start_timesteps=0,
                        text_embeddings=torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0),
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        desc=f"{prompt} | CORA erase",
                    )

                # ---- Save ----
                save_path = os.path.join(args.save_root, args.target_concept.replace(", ", "_"), concept)
                for mode in mode_list: os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1: os.makedirs(os.path.join(save_path, "combine"), exist_ok=True)

                decoded = {
                    name: [
                        process_img(vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0])
                        for img in img_list
                    ]
                    for name, img_list in Images.items()
                }

                def combine_h(ims):
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


if __name__ == "__main__":
    main()
