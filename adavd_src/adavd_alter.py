# adavd_src/adavd_alter.py
# Deterministic AdaVD, aligned with CORA's execution so images match seed-for-seed.

import os
import re
import copy
import argparse
from typing import Dict, List, Tuple

from PIL import Image
from tqdm import tqdm
from einops import rearrange

import torch
from torch import nn
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from template import template_dict
from utils import *  # seed_everything, get_token, get_textencoding, get_eot_idx, get_spread_embedding, process_img


# --------------------------
# Determinism (very important)
# --------------------------
# torch.use_deterministic_algorithms(True, warn_only=False)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
# try:
#     # Disable flash/mem-efficient kernels => use math attention (deterministic)
#     torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True)
# except Exception:
#     pass  # Older torch versions


# ======================
# AdaVD Attention Blocks
# ======================
ORTHO_DECOMP_STORAGE: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}


class VisualAttentionProcess(nn.Module):
    def __init__(
        self,
        module_name=None,
        atten_type="original",
        target_records=None,
        record=False,
        record_type=None,
        sigmoid_setting=None,
        decomp_timestep=0,
        **kwargs,
    ):
        super().__init__()
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = target_records
        self.record = record
        self.record_type = record_type
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep

    def __call__(self, attn, hidden_states, encoder_hidden_states, *args, **kwargs):
        attn._modules.pop("processor")
        attn.processor = AttnProcessor(
            module_name=self.module_name,
            atten_type=self.atten_type,
            target_records=self.target_records,
            record=self.record,
            record_type=self.record_type,
            sigmoid_setting=self.sigmoid_setting,
            decomp_timestep=self.decomp_timestep,
        )
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)


class AttnProcessor:
    """
    Implements AdaVD edits in the attention processor with a callable __call__,
    matching Diffusers' expected interface.
    """

    def __init__(
        self,
        module_name=None,
        atten_type="original",
        target_records=None,
        record=False,
        record_type=None,
        sigmoid_setting=None,
        decomp_timestep=0,
    ) -> None:
        self.module_name = module_name
        self.atten_type = atten_type  # "original" | "erase" | "retain"
        self.target_records = copy.copy(target_records)
        self.record = record
        self.record_type = record_type.strip().split(",") if record_type is not None else []
        self.records = {key: {} for key in self.record_type} if record_type is not None else {}
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep

    # ---- math helpers ----
    def sigmoid(self, x, setting):
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def cal_gram_schmidt(self, target_value):
        target_value = target_value.view((2, int(len(target_value) // 16), -1) + target_value.size()[-2:])
        target_value = (
            target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
        )
        target_value_ = rearrange(target_value, "b h l d -> b l (h d)")
        results = [self.gram_schmidt(target_value_[:, i, :]) for i in range(target_value_.size()[1])]
        project_matrix = torch.stack([result[0] for result in results], dim=0)  # [77, 2, 2]
        basis_ortho = torch.stack([result[1] for result in results], dim=0)  # [77, 2, 640]
        return project_matrix, basis_ortho

    def gram_schmidt(self, V):  # V: [n, d]
        n = len(V)
        project_matrix = torch.zeros((n, n), dtype=V.dtype, device=V.device) + torch.diag(
            torch.ones(n, dtype=V.dtype, device=V.device)
        )
        for i in range(1, n):
            vi = V[i : i + 1, :]
            for j in range(i):
                qj = V[j : j + 1, :]
                project_matrix[i][j] = -torch.dot(qj.view(-1), vi.view(-1)) / torch.dot(qj.view(-1), qj.view(-1))
        ortho_basis = torch.matmul(project_matrix, V)  # n x d
        return project_matrix, ortho_basis

    def cal_ortho_decomp(self, target_value, pro_record, ortho_basis=None, project_matrix=None):
        # Returns "erase component" era_record (same shape as pro_record)
        if ortho_basis is None and project_matrix is None:
            # single-concept
            tar_record_ = target_value[0].permute(1, 0, 2).reshape(77, -1)  # [77, 640]
            pro_record_ = pro_record.permute(1, 0, 2).reshape(77, -1)  # [77, 640]

            dot1 = (tar_record_ * pro_record_).sum(-1)
            dot2 = (tar_record_ * tar_record_).sum(-1).clamp_min(1e-12)
            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)
            if self.sigmoid_setting is not None:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)

            weight = torch.nan_to_num(cos_sim * (dot1 / dot2), nan=0.0)
            if weight.numel() > 0:
                # keep BOS/EOS unedited
                weight[0].fill_(0)

            era_record = weight.unsqueeze(0).unsqueeze(-1) * tar_record_.view((77, 16, -1)).permute(1, 0, 2)
            return era_record
        else:
            # multi-concept
            tar_record_ = rearrange(target_value, "b h l d -> l b (h d)")  # [77, num_concepts, 640]
            pro_record_ = rearrange(pro_record, "h l d -> l (h d)").unsqueeze(1)  # [77, 1, 640]

            dot1 = (ortho_basis * pro_record_).sum(-1)
            dot2 = (ortho_basis * ortho_basis).sum(-1).clamp_min(1e-12)
            weight = torch.nan_to_num((dot1 / dot2).unsqueeze(1), nan=0.0)
            if weight.numel() > 0:
                weight[0].fill_(0)

            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)  # [77, n_concepts]
            if self.sigmoid_setting is not None:
                cos_sim = self.sigmoid(cos_sim, self.sigmoid_setting)

            projected_basis = torch.bmm(project_matrix, cos_sim.unsqueeze(-1) * tar_record_)
            era_record = torch.bmm(weight, projected_basis).view((77, 16, -1)).permute(1, 0, 2)
            return era_record

    def record_ortho_decomp(self, target_record, current_record):
        # Find one record matching this module name; pop it so each gets used once.
        current_name = next(k for k in target_record if k.endswith(self.module_name))
        current_timestep, current_block = current_name.split(".", 1)
        (target_value, project_matrix, ortho_basis) = target_record.pop(current_name)

        if int(current_timestep) <= self.decomp_timestep:
            return current_record, current_record

        if current_block in ORTHO_DECOMP_STORAGE:
            # already decomposed for this prompt
            return ORTHO_DECOMP_STORAGE[current_block]

        # Reshape to align heads/batches across concepts
        target_value = target_value.view((2, int(len(target_value) // 16), -1) + target_value.size()[-2:])
        target_value = (
            target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
        )
        current_record = current_record.view((2, int(len(current_record) // 16), -1) + target_value.size()[-2:])
        current_record = (
            current_record.permute(1, 0, 2, 3, 4)
            .contiguous()
            .view((current_record.size()[1], -1) + target_value.size()[-2:])
        )

        erase_record, retain_record = [], []
        for pro_record in current_record:
            era = self.cal_ortho_decomp(target_value, pro_record, ortho_basis, project_matrix)
            ret = pro_record - era
            erase_record.append(era.view((2, -1) + era.size()[-2:]))
            retain_record.append(ret.view((2, -1) + ret.size()[-2:]))

        retain_record = rearrange(torch.stack(retain_record, dim=0), "b n c l d -> (n b c) l d")
        erase_record = rearrange(torch.stack(erase_record, dim=0), "b n c l d -> (n b c) l d")
        ORTHO_DECOMP_STORAGE[current_block] = (erase_record, retain_record)
        return ORTHO_DECOMP_STORAGE[current_block]

    # ---- the required callable ----
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

        # QKV
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)

        # AdaVD edit points (queries/keys/attn_maps) happen only when CLIP length (77)
        if not self.record and encoder_hidden_states.shape[1] == 77:
            if self.target_records is not None:
                if "queries" in self.target_records:
                    erase_q, retain_q = self.record_ortho_decomp(self.target_records["queries"], query)
                    query = retain_q if self.atten_type == "retain" else erase_q if self.atten_type == "erase" else query
                if "keys" in self.target_records:
                    erase_k, retain_k = self.record_ortho_decomp(self.target_records["keys"], key)
                    key = retain_k if self.atten_type == "retain" else erase_k if self.atten_type == "erase" else key

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if not self.record and encoder_hidden_states.shape[1] == 77:
            if self.target_records is not None and "attn_maps" in self.target_records:
                erase_p, retain_p = self.record_ortho_decomp(self.target_records["attn_maps"], attention_probs)
                attention_probs = (
                    retain_p if self.atten_type == "retain" else erase_p if self.atten_type == "erase" else attention_probs
                )

        # Self vs cross
        if encoder_hidden_states.shape[1] != 77:
            hidden_states = torch.bmm(attention_probs, value)
        else:
            # Cross attention path
            if self.record:
                # Save the raw tensors for later target_records construction
                for kk, vv in {"queries": query, "keys": key, "values": value, "attn_maps": attention_probs}.items():
                    if kk in self.record_type:
                        if vv.shape[0] // 16 == 1:  # single-concept
                            self.records[kk][self.module_name] = [vv] + [None, None]
                        else:  # multi-concept
                            self.records[kk][self.module_name] = [vv] + list(self.cal_gram_schmidt(vv))
            elif self.target_records is not None and "values" in self.target_records:
                erase_v, retain_v = self.record_ortho_decomp(self.target_records["values"], value)
                value = retain_v if self.atten_type == "retain" else erase_v if self.atten_type == "erase" else value
            hidden_states = torch.bmm(attention_probs, value)

        # Out proj
        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)  # linear
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(b, c, h, w)
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def set_attenprocessor(
    unet,
    atten_type="original",
    target_records=None,
    record=False,
    record_type=None,
    sigmoid_setting=None,
    decomp_timestep=0,
    only_cross=True,
):
    for name, m in unet.named_modules():
        if name.endswith("attn2") or (not only_cross and name.endswith("attn1")):
            m.set_processor(
                VisualAttentionProcess(
                    module_name=name,
                    atten_type=atten_type,
                    target_records=(target_records if target_records is not None else None),
                    record=record,
                    record_type=record_type,
                    sigmoid_setting=sigmoid_setting,
                    decomp_timestep=decomp_timestep,
                )
            )
    return unet


# ======================
# Diffusion & Decoding
# ======================
@torch.no_grad()
def diffusion(
    unet,
    scheduler,
    latents,
    text_embeddings,
    total_timesteps,
    start_timesteps=0,
    guidance_scale=7.5,
    record=False,
    record_type=None,
    desc=None,
):
    visualize_map = {key: {} for key in record_type.strip().split(",")} if record else {}
    scheduler.set_timesteps(total_timesteps)

    for timestep in tqdm(scheduler.timesteps[start_timesteps:total_timesteps], desc=desc):
        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        if record:
            for tname in record_type.strip().split(","):
                for proc in unet.attn_processors.values():
                    if hasattr(proc, "records"):
                        for k, v in proc.records[tname].items():
                            if k not in visualize_map[tname]:
                                visualize_map[tname][f"{timestep.item()}.{k.split('.',1)[1]}"] = v

        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map) if record else latents


@torch.no_grad()
def decode_latents_batch(vae, latents_list: List[torch.Tensor]) -> List[Image.Image]:
    if len(latents_list) == 0:
        return []
    latents = torch.stack(latents_list, dim=0)
    imgs = vae.decode(latents / vae.config.scaling_factor, return_dict=False)[0]
    out = [process_img(img) for img in imgs]
    return out


# =============
#      Main
# =============
@torch.no_grad()
def main():
    global ORTHO_DECOMP_STORAGE

    parser = argparse.ArgumentParser()
    parser.add_argument("--sd_ckpt", type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--mode", type=str, default="original,erase")  # original, erase, retain
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--total_timesteps", type=int, default=30)
    parser.add_argument("--decomp_timestep", type=int, default=0)
    parser.add_argument("--num_samples", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=10)

    # Erasing config
    parser.add_argument("--erase_type", type=str, default="", help="instance, style, celebrity, ...")
    parser.add_argument("--target_concept", type=str, default="")
    parser.add_argument("--contents", type=str, default="")
    parser.add_argument("--sigmoid_a", type=float, default=100)
    parser.add_argument("--sigmoid_b", type=float, default=0.93)
    parser.add_argument("--sigmoid_c", type=float, default=2)
    parser.add_argument("--record_type", type=str, default="values", help="keys, values, attn_maps")
    parser.add_argument("--save_path", type=str, required=True)
    parser.add_argument("--only_cross", action="store_true", help="Limit to attn2 (recommended to match CORA)")
    args = parser.parse_args()

    assert args.num_samples >= args.batch_size
    bs = args.batch_size
    mode_list = args.mode.replace(" ", "").split(",")
    concept_list = [s.strip() for s in args.contents.split(",") if s.strip()]
    assert args.target_concept.strip() != "", "Provide --target_concept"
    only_cross = True if args.only_cross or True else False  # default True to match CORA

    # ---- Pipeline ----
    pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, safety_checker=None, torch_dtype=torch.float16).to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)

    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    device, dtype = pipe.device, pipe.unet.dtype

    # Encodings
    uncond_encoding = get_textencoding(get_token("", tokenizer), text_encoder).to(device, dtype=dtype)

    # ---- Record target tensors once (1 step), then expand to all timesteps ----
    target_records = {}
    if any(m in ("erase", "retain") for m in mode_list):
        # Temporary recorder UNet (no need to deepcopy the big UNet; we use same weights)
        set_attenprocessor(unet, atten_type="original", record=True, record_type=args.record_type, only_cross=only_cross)

        # dedicated 1-step scheduler to avoid messing with the main one
        rec_sched = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        rec_sched.set_timesteps(1)

        target_concepts = [s.strip() for s in args.target_concept.split(",") if s.strip()]
        encs = [get_textencoding(get_token(tc, tokenizer), text_encoder) for tc in target_concepts]
        eots = [get_eot_idx(get_token(tc, tokenizer)) for tc in target_concepts]
        spreads = [get_spread_embedding(enc, idx) for (enc, idx) in zip(encs, eots)]
        target_concept_encoding = torch.concat(spreads).to(device, dtype=dtype)

        lat0 = torch.zeros(len(target_concept_encoding), 4, 64, 64, device=device, dtype=dtype)
        txt = torch.cat([uncond_encoding] * len(target_concept_encoding) + [target_concept_encoding], dim=0)

        latent_model_input = torch.cat([lat0] * 2)
        latent_model_input = rec_sched.scale_model_input(latent_model_input, rec_sched.timesteps[0])
        _ = unet(latent_model_input, rec_sched.timesteps[0], encoder_hidden_states=txt).sample

        # Collect and expand
        target_records = {args.record_type: {}}
        for proc in unet.attn_processors.values():
            if hasattr(proc, "records"):
                for k, v in proc.records.get(args.record_type, {}).items():
                    target_records[args.record_type][f"0.{k}"] = v  # seed key with timestep=0

        # Expand across main sampling timesteps
        pipe.scheduler.set_timesteps(args.total_timesteps)
        original_items = list(target_records[args.record_type].items())
        expanded = {}
        for t in pipe.scheduler.timesteps:
            t_int = int(t)
            for key, val in original_items:
                suffix = ".".join(key.split(".")[1:])
                expanded[f"{t_int}.{suffix}"] = val
        target_records[args.record_type].update(expanded)

        # Remove recorder
        set_attenprocessor(unet, atten_type="original", record=False, record_type=None, only_cross=only_cross)

    # ---- Sampling (match CORA exactly) ----
    seed_everything(args.seed, True)
    prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]

    os.makedirs(args.save_path, exist_ok=True)
    for mode in mode_list:
        os.makedirs(os.path.join(args.save_path, mode), exist_ok=True)
    if len(mode_list) > 1:
        os.makedirs(os.path.join(args.save_path, "combine"), exist_ok=True)

    for i in range(int(args.num_samples // bs)):
        # one latents tensor reused for all modes & prompts in this batch index
        latents = torch.randn(bs, 4, 64, 64, device=device, dtype=dtype)

        for concept, prompts in zip(concept_list, prompt_list):
            for prompt in prompts:
                Images: Dict[str, List[torch.Tensor]] = {}

                encoding = get_textencoding(get_token(prompt, tokenizer), text_encoder).to(device, dtype=dtype)
                txt = torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0)

                # ORIGINAL
                if "original" in mode_list:
                    set_attenprocessor(unet, atten_type="original", record=False, target_records=None, only_cross=only_cross)
                    Images["original"] = diffusion(
                        unet=unet,
                        scheduler=pipe.scheduler,
                        latents=latents,
                        start_timesteps=0,
                        text_embeddings=txt,
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        record=False,
                        desc=f"{prompt} | original",
                    )

                # ERASE (AdaVD)
                if "erase" in mode_list:
                    ORTHO_DECOMP_STORAGE.clear()  # reset per-prompt cache
                    set_attenprocessor(
                        unet,
                        atten_type="erase",
                        target_records=copy.deepcopy(target_records),
                        record=False,
                        record_type=None,
                        sigmoid_setting=(args.sigmoid_a, args.sigmoid_b, args.sigmoid_c),
                        decomp_timestep=args.decomp_timestep,
                        only_cross=only_cross,
                    )
                    Images["erase"] = diffusion(
                        unet=unet,
                        scheduler=pipe.scheduler,
                        latents=latents,
                        start_timesteps=0,
                        text_embeddings=txt,
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        record=False,
                        desc=f"{prompt} | AdaVD erase",
                    )

                # RETAIN (optional)
                if "retain" in mode_list:
                    ORTHO_DECOMP_STORAGE.clear()
                    set_attenprocessor(
                        unet,
                        atten_type="retain",
                        target_records=copy.deepcopy(target_records),
                        record=False,
                        record_type=None,
                        sigmoid_setting=(args.sigmoid_a, args.sigmoid_b, args.sigmoid_c),
                        decomp_timestep=args.decomp_timestep,
                        only_cross=only_cross,
                    )
                    Images["retain"] = diffusion(
                        unet=unet,
                        scheduler=pipe.scheduler,
                        latents=latents,
                        start_timesteps=0,
                        text_embeddings=txt,
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        record=False,
                        desc=f"{prompt} | AdaVD retain",
                    )

                # -------- Save decoded images (reuse batch decoder) --------
                decoded = {
                    name: decode_latents_batch(vae, [img for img in img_list]) for name, img_list in Images.items()
                }

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
                        decoded[mode][idx].save(os.path.join(args.save_path, mode, fname))
                        row.append(decoded[mode][idx])
                    if len(mode_list) > 1:
                        combine_h(row).save(os.path.join(args.save_path, "combine", fname))


if __name__ == "__main__":
    main()
