import os, sys, pdb
import re
import copy
import random
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
from einops import rearrange

import torch
from torch import nn
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

from template import template_dict
from utils import *

# === BEGIN: instrumentation globals ===
CURRENT_TIMESTEP = None
CURRENT_PROMPT_TOKENS = None  # list[str] for the current prompt
CURRENT_PROMPT_STR = ""       # full prompt string
EFFECTIVE_LOG = []            # list of dict rows
# === END: instrumentation globals ===

def _ping(msg: str):
    print(msg)
    sys.stdout.flush()

class VisualAttentionProcess(nn.Module):
    def __init__(self, module_name=None, atten_type='original', target_records=None, record=False,
                 record_type=None, sigmoid_setting=None, decomp_timestep=0,  **kwargs):
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
        attn.processor = AttnProcessor(self.module_name, self.atten_type, self.target_records,
                                      self.record, self.record_type, self.sigmoid_setting,
                                      self.decomp_timestep)
        return attn.processor(attn, hidden_states, encoder_hidden_states, *args, **kwargs)


class AttnProcessor():
    def __init__(self, module_name=None, atten_type='original', target_records=None, record=False,
                 record_type=None, sigmoid_setting=None, decomp_timestep=0) -> None:
        self.module_name = module_name
        self.atten_type = atten_type
        self.target_records = copy.copy(target_records)
        self.record = record
        self.record_type = record_type.strip().split(',') if record_type is not None else []
        self.records = {key: {} for key in self.record_type} if record_type is not None else {}
        self.sigmoid_setting = sigmoid_setting
        self.decomp_timestep = decomp_timestep

    def sigmoid(self, x, setting):
        a, b, c = setting
        return c / (1 + torch.exp(-a * (x - b)))

    def cal_ortho_decomp(self, target_value, pro_record, ortho_basis=None, project_matrix=None):
        # returns era_record and a dict of metrics
        if ortho_basis is None and project_matrix is None:
            tar_record_ = target_value[0].permute(1, 0, 2).reshape(77, -1)   # [77, 640]
            pro_record_ = pro_record.permute(1, 0, 2).reshape(77, -1)        # [77, 640]

            # --- raw cosine BEFORE gating ---
            raw_cos = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)  # [77]

            # --- gate ---
            if self.sigmoid_setting is not None:
                a, b, c = self.sigmoid_setting
                gate = c / (1 + torch.exp(-a * (raw_cos - b)))
            else:
                gate = torch.ones_like(raw_cos)

            # --- dot ratio ---
            dot1 = (tar_record_ * pro_record_).sum(-1)                       # [77]
            dot2 = (tar_record_ * tar_record_).sum(-1).clamp_min(1e-12)      # [77]
            dot_ratio = dot1 / dot2                                          # [77]

            weight = torch.nan_to_num(gate * dot_ratio, nan=0.0)             # [77]
            weight[0].fill_(0)  # keep BOS/EOS behavior identical to original

            era_record = weight.unsqueeze(0).unsqueeze(-1) * tar_record_.view((77, 16, -1)).permute(1, 0, 2)
            metrics = {"raw_cos": raw_cos.detach(), "gate": gate.detach(),
                       "dot_ratio": dot_ratio.detach(), "weight": weight.detach()}
            return era_record, metrics

        else:
            # (multi-concept) – log a scalar per position
            tar_record_ = rearrange(target_value, 'b h l d -> l b (h d)')     # [77, num_concepts, 640]
            pro_record_ = rearrange(pro_record, 'h l d -> l (h d)').unsqueeze(1)  # [77, 1, 640]

            dot1 = (ortho_basis * pro_record_).sum(-1)                        # [77, num_concepts]
            dot2 = (ortho_basis * ortho_basis).sum(-1).clamp_min(1e-12)       # [77, num_concepts]
            dot_ratio = (dot1 / dot2).mean(-1)                                # [77]

            raw_cos = torch.cosine_similarity(tar_record_.mean(1), pro_record_.squeeze(1), dim=-1)  # [77]
            if self.sigmoid_setting is not None:
                a, b, c = self.sigmoid_setting
                gate = c / (1 + torch.exp(-a * (raw_cos - b)))
            else:
                gate = torch.ones_like(raw_cos)

            weight = torch.nan_to_num(dot_ratio * gate, nan=0.0)              # [77]
            weight[0].fill_(0)

            cos_sim = torch.cosine_similarity(tar_record_, pro_record_, dim=-1)  # [77, num_concepts]
            if self.sigmoid_setting is not None:
                cos_sim = c / (1 + torch.exp(-a * (cos_sim - b)))
            projected_basis = torch.bmm(project_matrix, cos_sim.unsqueeze(-1) * tar_record_)
            weight_bc = dot_ratio.unsqueeze(0).unsqueeze(-1)                  # [1,77,1]
            era_record = torch.bmm(weight_bc, projected_basis).view((77, 16, -1)).permute(1, 0, 2)

            metrics = {"raw_cos": raw_cos.detach(), "gate": gate.detach(),
                       "dot_ratio": dot_ratio.detach(), "weight": weight.detach()}
            return era_record, metrics

    def _push_metrics(self, block_name, raw_cos, gate, dot_ratio, weight):
        tokens = CURRENT_PROMPT_TOKENS if CURRENT_PROMPT_TOKENS is not None else []
        ts = int(CURRENT_TIMESTEP.item()) if CURRENT_TIMESTEP is not None else -1
        for i in range(raw_cos.shape[0]):
            EFFECTIVE_LOG.append({
                "timestep": ts,
                "layer": block_name,
                "token_idx": int(i),
                "token_text": tokens[i] if i < len(tokens) else "",
                "prompt": CURRENT_PROMPT_STR,
                "raw_cos": float(raw_cos[i]),
                "gate": float(gate[i]),
                "dot_ratio": float(dot_ratio[i]),
                "weight": float(weight[i]),
            })

    def record_ortho_decomp(self, target_record, current_record):
        # Find a record for this module name; pop it so each record is used once.
        try:
            current_name = next(k for k in target_record if k.endswith(self.module_name))
        except StopIteration:
            # No matching record (debug ping once per block)
            return current_record, current_record

        current_timestep, current_block = current_name.split('.', 1)
        (target_value, project_matrix, ortho_basis) = target_record.pop(current_name)

        if int(current_timestep) <= self.decomp_timestep:
            return current_record, current_record

        if current_block in ORTHO_DECOMP_STORAGE:
            # already decomposed this block for this prompt; reuse
            return ORTHO_DECOMP_STORAGE[current_block]

        # First time for this block in this prompt: compute and log
        target_value = target_value.view((2, int(len(target_value)//16), -1) + target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
        current_record = current_record.view((2, int(len(current_record)//16), -1) + target_value.size()[-2:])
        current_record = current_record.permute(1, 0, 2, 3, 4).contiguous().view((current_record.size()[1], -1) + target_value.size()[-2:])
        erase_record, retain_record = [], []

        for pro_record in current_record:
            era_record, metrics = self.cal_ortho_decomp(target_value, pro_record, ortho_basis, project_matrix)
            self._push_metrics(current_block, metrics["raw_cos"], metrics["gate"], metrics["dot_ratio"], metrics["weight"])
            ret_record = pro_record - era_record
            erase_record.append(era_record.view((2, -1) + era_record.size()[-2:]))
            retain_record.append(ret_record.view((2, -1) + ret_record.size()[-2:]))

        retain_record = rearrange(torch.stack(retain_record, dim=0), 'b n c l d -> (n b c) l d')
        erase_record = rearrange(torch.stack(erase_record, dim=0), 'b n c l d -> (n b c) l d')
        ORTHO_DECOMP_STORAGE[current_block] = (erase_record, retain_record)
        return ORTHO_DECOMP_STORAGE[current_block]

    def cal_gram_schmidt(self, target_value):
        target_value = target_value.view((2, int(len(target_value)//16), -1) + target_value.size()[-2:])
        target_value = target_value.permute(1, 0, 2, 3, 4).contiguous().view((target_value.size()[1], -1) + target_value.size()[-2:])
        target_value_ = rearrange(target_value, 'b h l d -> b l (h d)')
        results = [self.gram_schmidt(target_value_[:, i, :]) for i in range(target_value_.size()[1])]
        project_matrix = torch.stack([result[0] for result in results], dim=0)  # [77, 2, 2]
        basis_ortho = torch.stack([result[1] for result in results], dim=0)    # [77, 2, 640]
        return project_matrix, basis_ortho

    def gram_schmidt(self, V):  # [n, 1, d]
        n = len(V)
        project_matrix = torch.zeros((n, n), dtype=V.dtype).to(V.device) + torch.diag(torch.ones(n, dtype=V.dtype)).to(V.device)
        for i in range(1, n):
            vi = V[i:i+1, :]
            for j in range(i):
                qj = V[j:j+1, :]
                project_matrix[i][j] = -torch.dot(qj.view(-1), vi.view(-1)) / torch.dot(qj.view(-1), qj.view(-1))
        ortho_basis = torch.matmul(project_matrix.to(V.device), V)  # n d
        return project_matrix.to(V.device), ortho_basis

    def __call__(self, attn, hidden_states, encoder_hidden_states=None, attention_mask=None, temb=None):
        residual = hidden_states

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim
        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )
        attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)

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
        value = attn.head_to_batch_dim(value)  # [batch, head, len, dim//head]

        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'queries' in self.target_records:
                erase_query, retain_query = self.record_ortho_decomp(
                    target_record=self.target_records['queries'],
                    current_record=query,
                )
                query = retain_query if self.atten_type == 'retain' else erase_query if self.atten_type == 'erase' else query
            if 'keys' in self.target_records:
                erase_key, retain_key = self.record_ortho_decomp(
                    target_record=self.target_records['keys'],
                    current_record=key,
                )
                key = retain_key if self.atten_type == 'retain' else erase_key if self.atten_type == 'erase' else key

        attention_probs = attn.get_attention_scores(query, key, attention_mask)

        if not self.record and encoder_hidden_states.shape[1] == 77:
            if 'attn_maps' in self.target_records:
                erase_attention_probs, retain_attention_probs = self.record_ortho_decomp(
                    target_record=self.target_records['attn_maps'],
                    current_record=attention_probs,
                )
                attention_probs = retain_attention_probs if self.atten_type == 'retain' else erase_attention_probs if self.atten_type == 'erase' else attention_probs

        if encoder_hidden_states.shape[1] != 77:
            # self-attention
            hidden_states = torch.bmm(attention_probs, value)
        else:
            # cross-attention
            if self.record:
                for kk, vv in {'queries': query, 'keys': key, 'values': value, 'attn_maps': attention_probs}.items():
                    if kk in self.record_type:
                        if vv.shape[0] // 16 == 1:  # single-concept
                            self.records[kk][self.module_name] = [vv] + [None, None]
                        else:  # multi-concept
                            self.records[kk][self.module_name] = [vv] + list(self.cal_gram_schmidt(vv))
            elif 'values' in self.target_records:
                erase_value, retain_value = self.record_ortho_decomp(
                    target_record=self.target_records['values'],
                    current_record=value,
                )
                value = retain_value if self.atten_type == 'retain' else erase_value if self.atten_type == 'erase' else value
            hidden_states = torch.bmm(attention_probs, value)

        hidden_states = attn.batch_to_head_dim(hidden_states)
        hidden_states = attn.to_out[0](hidden_states)  # linear proj
        hidden_states = attn.to_out[1](hidden_states)  # dropout

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)
        if attn.residual_connection:
            hidden_states = hidden_states + residual

        return hidden_states / attn.rescale_output_factor


def set_attenprocessor(unet, atten_type='original', target_records=None, record=False, record_type=None,
                       sigmoid_setting=None, decomp_timestep=0):
    for name, m in unet.named_modules():
        if name.endswith('attn2') or name.endswith('attn1'):
            cross_attention_dim = None if name.endswith("attn1") else unet.config.cross_attention_dim
            if name.startswith("mid_block"):
                hidden_size = unet.config.block_out_channels[-1]
            elif name.startswith("up_blocks"):
                block_id = int(name[len("up_blocks.")])
                hidden_size = list(reversed(unet.config.block_out_channels))[block_id]
            elif name.startswith("down_blocks"):
                block_id = int(name[len("down_blocks.")])
                hidden_size = unet.config.block_out_channels[block_id]

            m.set_processor(VisualAttentionProcess(
                module_name=name,
                atten_type=atten_type,
                target_records=target_records,
                record=record,
                record_type=record_type,
                cross_attention_dim=cross_attention_dim,
                sigmoid_setting=sigmoid_setting,
                decomp_timestep=decomp_timestep
            ))
    return unet


def diffusion(unet, scheduler, latents, text_embeddings, total_timesteps, start_timesteps=0,
              guidance_scale=7.5, record=False, record_type=None, desc=None, **kwargs):

    visualize_map_withstep = {key: {} for key in record_type.strip().split(',')} if record_type is not None else {}

    scheduler.set_timesteps(total_timesteps)
    for timestep in tqdm(scheduler.timesteps[start_timesteps: total_timesteps], desc=desc):

        latent_model_input = torch.cat([latents] * 2)
        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep)

        # record timestep for logging
        global CURRENT_TIMESTEP
        CURRENT_TIMESTEP = timestep

        # predict the noise residual
        noise_pred = unet(
            latent_model_input,
            timestep,
            encoder_hidden_states=text_embeddings,
        ).sample

        if record:
            for type in record_type.strip().split(','):
                for value in unet.attn_processors.values():  # This 'value' differs from CA/SA tensors
                    for k, v in value.records[type].items():
                        visualize_map_withstep[type][f'{timestep.item()}.{k}'] = v

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)  # epsilon(t, x_t, c)

        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    return (latents, visualize_map_withstep) if record else latents


ORTHO_DECOMP_STORAGE = {}

@torch.no_grad()
def main():
    global ORTHO_DECOMP_STORAGE

    parser = argparse.ArgumentParser()
    # Base Config
    parser.add_argument('--save_root', type=str, default='')
    parser.add_argument('--sd_ckpt', type=str, default="CompVis/stable-diffusion-v1-4")
    parser.add_argument('--seed', type=int, default=0)
    # Sampling Config
    parser.add_argument('--mode', type=str, default='original', help='original, erase, retain')
    parser.add_argument('--guidance_scale', type=float, default=7.5)
    parser.add_argument('--total_timesteps', type=int, default=30, help='total sampling steps')
    parser.add_argument('--decomp_timestep', type=int, default=0, help='skip decomp/logging for t <= this')
    parser.add_argument('--num_samples', type=int, default=10, help='samples per prompt')
    parser.add_argument('--batch_size', type=int, default=10, help='batch size')
    # Erasing Config
    parser.add_argument('--erase_type', type=str, default='', help='instance, style, celebrity, dogs_combined, ...')
    parser.add_argument('--target_concept', type=str, default='')
    parser.add_argument('--use_softmax', type=bool, default=False)
    parser.add_argument('--contents', type=str, default='')
    parser.add_argument('--sigmoid_a', type=float, default=100)
    parser.add_argument('--sigmoid_b', type=float, default=0.93)
    parser.add_argument('--sigmoid_c', type=float, default=2)
    parser.add_argument('--record_type', type=str, default='values', help="keys, values")
    # Rounding + plotting
    parser.add_argument('--round_decimals', type=int, default=2,
                        help='number of decimals for CSV output')
    parser.add_argument('--plot', action='store_true',
                        help='save quick diagnostic plots from the aggregated CSV')
    parser.add_argument('--plot_tokens', type=str, default='',
                        help='comma-separated tokens to visualize; if empty, auto top-k by mean weight')
    parser.add_argument('--plot_topk', type=int, default=12,
                        help='when --plot_tokens empty, number of top tokens to plot by mean weight')
    args = parser.parse_args()
    assert args.num_samples >= args.batch_size

    # normalize save_root path and make sure it exists
    args.save_root = os.path.abspath(args.save_root)
    os.makedirs(args.save_root, exist_ok=True)
    _ping(f"[AdaVD] save_root = {args.save_root}")

    bs = args.batch_size
    mode_list = args.mode.replace(' ', '').split(',')

    # region [If certain concept is already sampled, then skip it.]
    concept_list, concept_list_tmp = [], [item.strip() for item in args.contents.split(',')]
    if 'retain' in mode_list:
        for concept in concept_list_tmp:
            check_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept, 'retain')
            os.makedirs(check_path, exist_ok=True)
            if len(os.listdir(check_path)) != len(template_dict[args.erase_type]) * 10:
                concept_list.append(concept)
    else:
        concept_list = concept_list_tmp
    if len(concept_list) == 0:
        _ping("[AdaVD] No concepts to run — exiting.")
        sys.exit()
    _ping(f"[AdaVD] concepts = {concept_list}")
    # endregion

    # region [Prepare Models]
    pipe = DiffusionPipeline.from_pretrained(args.sd_ckpt, safety_checker=None, torch_dtype=torch.float16).to('cuda')
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    unet, tokenizer, text_encoder, vae = pipe.unet, pipe.tokenizer, pipe.text_encoder, pipe.vae
    if 'original' in mode_list: unet_original = copy.deepcopy(unet)
    if 'erase' in mode_list: unet_erase = copy.deepcopy(unet)
    if 'retain' in mode_list: unet_retain = copy.deepcopy(unet)
    # endregion

    # region [Prepare embeddings]
    target_concepts = [item.strip() for item in args.target_concept.split(',')]
    target_concept_encodings_ = [get_textencoding(get_token(concept, tokenizer), text_encoder) for concept in target_concepts]
    target_eot_idxs = [get_eot_idx(get_token(concept, tokenizer)) for concept in target_concepts]
    target_concept_encoding = [get_spread_embedding(target_concept_encoding_, idx)
                               for (target_concept_encoding_, idx) in zip(target_concept_encodings_, target_eot_idxs)]
    target_concept_encoding = torch.concat(target_concept_encoding)
    uncond_encoding = get_textencoding(get_token('', tokenizer), text_encoder)
    # endregion

    # region [Record target records once, then expand across timesteps]
    target_records = {}
    if 'erase' in mode_list or 'retain' in mode_list:
        unet_rec = set_attenprocessor(unet, atten_type='original', record=True, record_type=args.record_type)
        _, target_records = diffusion(
            unet=unet_rec, scheduler=pipe.scheduler,
            latents=torch.zeros(len(target_concept_encoding), 4, 64, 64).to(pipe.device, dtype=target_concept_encoding.dtype),
            text_embeddings=torch.cat([uncond_encoding] * len(target_concept_encoding) + [target_concept_encoding], dim=0),
            total_timesteps=1, start_timesteps=0, guidance_scale=args.guidance_scale,
            record=True, record_type=args.record_type, desc="Calculating target records",
        )
        # Expand safely across all sampling timesteps
        pipe.scheduler.set_timesteps(args.total_timesteps)
        original_items = list(target_records[args.record_type].items())
        expanded = {}
        for timestep in pipe.scheduler.timesteps:
            t = int(timestep)
            for key, val in original_items:
                suffix = '.'.join(key.split('.')[1:])
                expanded[f"{t}.{suffix}"] = val
        target_records[args.record_type].update(expanded)
        _ping(f"[AdaVD] target_records['{args.record_type}'] entries: {len(target_records[args.record_type])}")
        del unet_rec
    # endregion

    # Sampling process
    seed_everything(args.seed, True)
    prompt_list = [[x.format(concept) for x in template_dict[args.erase_type]] for concept in concept_list]
    for i in range(int(args.num_samples // bs)):
        latent = torch.randn(bs, 4, 64, 64).to(pipe.device, dtype=target_concept_encoding.dtype)
        for concept, prompts in zip(concept_list, prompt_list):
            for prompt in prompts:

                # reset per-prompt globals
                global CURRENT_PROMPT_TOKENS, CURRENT_PROMPT_STR, ORTHO_DECOMP_STORAGE
                ORTHO_DECOMP_STORAGE, Images = {}, {}
                CURRENT_PROMPT_STR = prompt

                encoding = get_textencoding(get_token(prompt, tokenizer), text_encoder)

                # record decoded tokens for this prompt
                tok_ids = tokenizer(prompt, return_tensors="pt").input_ids[0].tolist()
                CURRENT_PROMPT_TOKENS = [tokenizer.decode([t]) for t in tok_ids]

                if 'original' in mode_list:
                    Images['original'] = diffusion(
                        unet=unet_original, scheduler=pipe.scheduler,
                        latents=latent, start_timesteps=0,
                        text_embeddings=torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0),
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        desc=f"{prompt} | original"
                    )
                if 'erase' in mode_list:
                    unet_erase_proc = set_attenprocessor(
                        unet_erase, atten_type='erase', target_records=copy.deepcopy(target_records),
                        sigmoid_setting=(args.sigmoid_a, args.sigmoid_b, args.sigmoid_c),
                        decomp_timestep=args.decomp_timestep,
                    )
                    Images['erase'] = diffusion(
                        unet=unet_erase_proc, scheduler=pipe.scheduler,
                        latents=latent, start_timesteps=0,
                        text_embeddings=torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0),
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        desc=f"{prompt} | erase"
                    )

                if 'retain' in mode_list:
                    unet_retain_proc = set_attenprocessor(
                        unet_retain, atten_type='retain', target_records=copy.deepcopy(target_records),
                        sigmoid_setting=(args.sigmoid_a, args.sigmoid_b, args.sigmoid_c),
                        decomp_timestep=args.decomp_timestep,
                    )
                    Images['retain'] = diffusion(
                        unet=unet_retain_proc, scheduler=pipe.scheduler,
                        latents=latent, start_timesteps=0,
                        text_embeddings=torch.cat([uncond_encoding] * bs + [encoding] * bs, dim=0),
                        total_timesteps=args.total_timesteps,
                        guidance_scale=args.guidance_scale,
                        desc=f"{prompt} | retain"
                    )

                save_path = os.path.join(args.save_root, args.target_concept.replace(', ', '_'), concept)
                for mode in mode_list:
                    os.makedirs(os.path.join(save_path, mode), exist_ok=True)
                if len(mode_list) > 1:
                    os.makedirs(os.path.join(save_path, 'combine'), exist_ok=True)

                # Decode and process images
                decoded_imgs = {
                    name: [process_img(vae.decode(img.unsqueeze(0) / vae.config.scaling_factor, return_dict=False)[0]) for img in img_list]
                    for name, img_list in Images.items()
                }

                # Save images
                def combine_images_horizontally(Images_list):
                    widths, heights = zip(*(img.size for img in Images_list))
                    new_img = Image.new('RGB', (sum(widths), max(heights)))
                    for j, img in enumerate(Images_list):
                        new_img.paste(img, (sum(widths[:j]), 0))
                    return new_img

                for idx in range(len(decoded_imgs[mode_list[0]])):
                    safe_prompt = re.sub(r'[^\w\s]', '', prompt).replace(', ', '_')
                    save_filename = f"{safe_prompt}_{int(idx + bs * i)}.png"
                    images_to_combine = []
                    for mode in mode_list:
                        decoded_imgs[mode][idx].save(os.path.join(save_path, mode, save_filename))
                        images_to_combine.append(decoded_imgs[mode][idx])
                    if len(mode_list) > 1:
                        img_combined = combine_images_horizontally(images_to_combine)
                        img_combined.save(os.path.join(save_path, 'combine', save_filename))

    # === BEGIN: write apples-to-apples logs (always write, even if empty) ===
    import pandas as pd
    os.makedirs(args.save_root, exist_ok=True)
    raw_path = os.path.join(args.save_root, "adavd_effective_weight_raw.csv")
    agg_path = os.path.join(args.save_root, "adavd_effective_weight_mean.csv")

    # Always create a DataFrame so files are created even if log is empty
    if EFFECTIVE_LOG:
        df = pd.DataFrame(EFFECTIVE_LOG)
    else:
        df = pd.DataFrame(columns=[
            "timestep","layer","token_idx","token_text","prompt","raw_cos","gate","dot_ratio","weight"
        ])

    # Aggregate: mean over timesteps, per (layer, token_text)
    if len(df) > 0:
        agg = (df.groupby(["layer", "token_text"], dropna=False)
                [["raw_cos", "gate", "dot_ratio", "weight"]]
                .mean().reset_index())
    else:
        agg = pd.DataFrame(columns=["layer","token_text","raw_cos","gate","dot_ratio","weight"])

    # ---- ROUND TO N DECIMALS (output only) ----
    num_cols = ["raw_cos","gate","dot_ratio","weight"]
    df_out = df.copy()
    for col in num_cols:
        if col in df_out.columns:
            df_out[col] = pd.to_numeric(df_out[col], errors="coerce").round(args.round_decimals)
    df_out.to_csv(raw_path, index=False)

    agg_out = agg.copy()
    for col in num_cols:
        if col in agg_out.columns:
            agg_out[col] = pd.to_numeric(agg_out[col], errors="coerce").round(args.round_decimals)
    agg_out.to_csv(agg_path, index=False)

    _ping(f"[AdaVD] EFFECTIVE_LOG rows: {len(EFFECTIVE_LOG)}")
    _ping(f"[AdaVD] Wrote:\n  {raw_path}\n  {agg_path}")
    # === END ===
    # === OPTIONAL PLOTS ===
    if args.plot:
        def _make_plots(agg_df, out_dir, tokens_csv, topk, round_dec):
            # Local import + non-interactive backend
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from matplotlib.ticker import FormatStrFormatter
            import numpy as np
            import pandas as pd

            sub = agg_df.copy()
            sub["token_text"] = sub["token_text"].astype(str).str.lower()

            # pick tokens
            if tokens_csv.strip():
                tok_list = [t.strip().lower() for t in tokens_csv.split(",") if t.strip()]
            else:
                # top-k by global mean weight
                topk_idx = (sub.groupby("token_text")["weight"]
                            .mean().sort_values(ascending=False)
                            .head(topk).index.tolist())
                tok_list = topk_idx

            sub = sub[sub["token_text"].isin(tok_list)]
            if sub.empty:
                _ping("[AdaVD] Plotting skipped: token filter produced empty data.")
                return

            # token × layer pivot of mean weight
            piv = (sub.pivot_table(index="token_text", columns="layer", values="weight", aggfunc="mean")
                    .fillna(0.0))
            # round for readability (plot uses these values)
            piv = piv.applymap(lambda x: round(float(x), round_dec))

            # Keep current column order; map to numeric indices 0..L-1
            layers = list(piv.columns)
            x = np.arange(len(layers))

            # Save a mapping so you know which index is which layer string
            map_df = pd.DataFrame({"layer_idx": x, "layer": layers})
            map_path = os.path.join(out_dir, "adavd_layer_index_map.csv")
            map_df.to_csv(map_path, index=False)

            # Line plot: different color per token, x = layer index
            plt.figure(figsize=(max(8, len(layers)*0.5), max(4, len(piv.index)*0.6)))
            for token in piv.index:
                y = piv.loc[token, layers].values.astype(float)
                plt.plot(x, y, marker="o", label=token)  # default matplotlib colors => different line colors

            ax = plt.gca()
            ax.yaxis.set_major_formatter(FormatStrFormatter(f'%.{round_dec}f'))  # round y tick labels
            plt.xticks(x, [str(i) for i in x])   # show 0,1,2,... only
            plt.xlabel("layer index")
            plt.ylabel("mean weight")
            plt.title("AdaVD mean weight across layers")
            plt.grid(True, linestyle="--", alpha=0.3)
            plt.legend(loc="best", fontsize=9)
            plt.tight_layout()

            out_path = os.path.join(out_dir, "adavd_line_weight.png")
            plt.savefig(out_path, dpi=220)
            plt.close()

            _ping(f"[AdaVD] Plots:\n  {out_path}\n  {map_path}")

        _make_plots(agg_out, args.save_root, args.plot_tokens, args.plot_topk, args.round_decimals)



if __name__ == '__main__':
    main()
