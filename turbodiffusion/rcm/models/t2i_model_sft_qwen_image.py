# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Text-to-Image SFT model using QwenImageTransformer2DModel.

Inherits from T2VModel_SFT and overrides the parts that differ for image generation:
- No video noise multiplier (T=1 always).
- Passes crossattn_mask from the data batch to the net.
- LoRA default target modules match diffusers naming (to_q, to_k, to_v, to_out.0).
- generate_samples_from_batch decodes latents back to pixel images.
"""

from __future__ import annotations

import collections
from typing import Any, Dict, List, Mapping, Optional, Tuple

import attrs
import torch
from einops import rearrange, repeat
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor
from torch.nn.modules.module import _IncompatibleKeys

from imaginaire.lazy_config import LazyDict
from imaginaire.utils import log, misc
from rcm.conditioner import DataType
from rcm.models.t2v_model_sft import T2VConfig_SFT, T2VModel_SFT
import numpy as np
from diffusers import FlowMatchEulerDiscreteScheduler
from rcm.samplers.euler import FlowEulerSampler
from rcm.samplers.unipc import FlowUniPCMultistepSampler
from rcm.utils.checkpointer import non_strict_load_model

IS_PREPROCESSED_KEY = "is_preprocessed"
IS_PROCESSED_KEY = "is_processed"

# Default LoRA target modules for QwenImageTransformer (diffusers naming)
DEFAULT_LORA_TARGET_MODULES_QWEN = ["to_q", "to_k", "to_v", "to_out.0"]


@attrs.define(slots=False)
class T2IConfig_SFT_QwenImage(T2VConfig_SFT):
    """Config for Text-to-Image SFT with QwenImageTransformer2DModel.

    Key differences from T2VConfig_SFT:
    - state_t=1: images have a single frame.
    - adjust_video_noise=False: no sqrt(T) noise multiplier.
    - lora_target_modules defaults to diffusers naming convention.
    - text_encoder_class/path: unused (embeddings pre-extracted offline).
    """

    state_ch: int = 16
    state_t: int = 1
    adjust_video_noise: bool = False
    lora_target_modules: Optional[List[str]] = None  # default: DEFAULT_LORA_TARGET_MODULES_QWEN
    text_encoder_class: str = "qwen"
    text_encoder_path: str = ""


class T2IModel_SFT_QwenImage(T2VModel_SFT):
    """SFT model for Text-to-Image generation with QwenImageTransformer2DModel.

    Inherits from T2VModel_SFT and overrides:
    - _inject_lora: uses diffusers-style LoRA target module names.
    - draw_training_time: no video noise multiplier.
    - training_step: propagates crossattn_mask to the net.
    - generate_samples_from_batch: CFG denoising loop for images.
    """

    def __init__(self, config: T2IConfig_SFT_QwenImage) -> None:
        super().__init__(config)
        # Cache the official scheduler (loaded once at init) for use during sampling.
        _model_path = config.net.get("model_path", "") if hasattr(config.net, "get") else getattr(config.net, "model_path", "")
        if _model_path:
            self._qwen_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                _model_path, subfolder="scheduler"
            )
        else:
            self._qwen_scheduler = None

    # ------------------------------------------------------------------
    # LoRA injection with diffusers-style target modules
    # ------------------------------------------------------------------

    def _inject_lora(self, net: torch.nn.Module) -> torch.nn.Module:
        target_modules = self.config.lora_target_modules
        if not target_modules:
            log.warning(
                "lora_target_modules is empty, using default %s",
                DEFAULT_LORA_TARGET_MODULES_QWEN,
            )
            target_modules = DEFAULT_LORA_TARGET_MODULES_QWEN

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            init_lora_weights=True,  # B=0 init: delta_W = B@A = 0 at start, preserving pretrained model output
        )
        return get_peft_model(net, lora_config)

    # ------------------------------------------------------------------
    # Timestep sampling — no video noise multiplier
    # ------------------------------------------------------------------

    def draw_training_time(self, x0_size: int, condition: Any) -> torch.Tensor:
        batch_size = x0_size[0]
        sigma_B = self.p_t(batch_size).to(device="cuda")
        sigma_B_1 = rearrange(sigma_B, "b -> b 1")
        # Images: no video noise multiplier (T=1)
        time_B_1 = sigma_B_1 / (sigma_B_1 + 1)
        return time_B_1

    # ------------------------------------------------------------------
    # Training step — propagates crossattn_mask
    # ------------------------------------------------------------------

    def training_step(
        self,
        data_batch: dict[str, Tensor],
        iteration: int,
    ) -> Tuple[dict[str, Tensor], Tensor]:
        _, x0_B_C_T_H_W, condition, _uncondition = self.get_data_and_condition(data_batch)

        time_B_T = self.draw_training_time(x0_B_C_T_H_W.size(), condition)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        x0_B_C_T_H_W, time_B_T, epsilon_B_C_T_H_W, condition = self.sync(
            x0_B_C_T_H_W, time_B_T, epsilon_B_C_T_H_W, condition
        )

        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        xt_B_C_T_H_W = (1 - time_B_1_T_1_1) * x0_B_C_T_H_W + time_B_1_T_1_1 * epsilon_B_C_T_H_W

        target_B_C_T_H_W = epsilon_B_C_T_H_W - x0_B_C_T_H_W  # rectified flow velocity

        # timesteps_B_T: (B,) normalized sigma in [0, 1].
        # QwenTimestepProjEmbeddings uses scale=1000 internally, so we pass sigma directly
        # (matching the official QwenImagePipeline which passes timestep / 1000).
        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=time_B_1_T_1_1.reshape(x0_B_C_T_H_W.size(0)).to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()

        loss = self.config.loss_scale * (
            (net_output_B_C_T_H_W - target_B_C_T_H_W) ** 2
        ).mean(dim=(1, 2, 3, 4)).mean()

        output_batch = {
            "x0": x0_B_C_T_H_W.detach().cpu(),
            "xt": xt_B_C_T_H_W.detach().cpu(),
            "F_pred": net_output_B_C_T_H_W.detach().cpu(),
            "target": target_B_C_T_H_W.detach().cpu(),
        }
        return output_batch, loss

    # ------------------------------------------------------------------
    # get_data_and_condition — force IMAGE data type
    # ------------------------------------------------------------------

    def get_data_and_condition(self, data_batch: dict[str, Tensor]):
        if IS_PROCESSED_KEY not in data_batch or not data_batch[IS_PROCESSED_KEY]:
            if self.config.input_latent_key in data_batch:
                self._normalize_latent_inplace(data_batch)
                data_batch[self.config.input_data_key] = (
                    self.decode(data_batch[self.config.input_latent_key]).contiguous().float().clamp(-1, 1)
                )
                data_batch[IS_PREPROCESSED_KEY] = True
            self._normalize_video_inplace(data_batch)
            data_batch[self.config.input_latent_key] = self.encode(
                data_batch[self.config.input_data_key]
            ).contiguous().float()
            data_batch[IS_PROCESSED_KEY] = True

        raw_state = data_batch[self.config.input_data_key]
        latent_state = data_batch[self.config.input_latent_key]
        if self.neg_embed is not None:
            B = data_batch["embed"].shape[0]
            if isinstance(self.neg_embed, dict):
                neg_emb = self.neg_embed["embed"]  # (1, L, D) or (L, D)
                neg_msk = self.neg_embed["mask"]    # (1, L) or (L,)
                if neg_emb.ndim == 2:
                    neg_emb = neg_emb.unsqueeze(0)
                if neg_msk.ndim == 1:
                    neg_msk = neg_msk.unsqueeze(0)
            else:
                neg_emb = self.neg_embed  # (L, D) bare tensor from old format
                if neg_emb.ndim == 2:
                    neg_emb = neg_emb.unsqueeze(0)
                L = neg_emb.shape[1]
                neg_msk = torch.ones(1, L, dtype=torch.int8)

            data_batch["neg_embed"] = repeat(neg_emb.to(**self.tensor_kwargs), "1 l d -> b l d", b=B)
            data_batch["neg_mask"] = repeat(neg_msk.to(device=self.tensor_kwargs["device"]), "1 l -> b l", b=B)
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(DataType.IMAGE)
        uncondition = uncondition.edit_data_type(DataType.IMAGE)
        return raw_state, latent_state, condition, uncondition

    # ------------------------------------------------------------------
    # Inference — CFG denoising loop for images
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        teacher: bool = False,
        state_shape: Optional[Tuple] = None,
        n_sample: Optional[int] = None,
        init_noise: Optional[Tensor] = None,
        num_steps: int = 50,
        sampler: str = "UniPC",
        timestep_shift: Optional[float] = None,
    ) -> Tensor:
        _, _, condition, uncondition = self.get_data_and_condition(data_batch)
        input_key = self.config.input_latent_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _latent = data_batch[input_key]
            # latent shape: (B, C, T, H, W) — T=1
            state_shape = list(_latent.shape[1:])  # [C, 1, H, W]

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        net = self.net

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )
        init_noise, condition, uncondition = self.sync(init_noise, condition, uncondition)
        x = init_noise.to(torch.float64)

        # Use the official QwenImagePipeline scheduler (cached in __init__).
        # This ensures correct use_dynamic_shifting, max_shift, time_shift_type etc.
        # sigmas = linspace(1.0, 1/num_steps, num_steps) matches the official pipeline.
        scheduler = self._qwen_scheduler
        # image_seq_len: number of packed image tokens = (H_lat/patch_size) * (W_lat/patch_size)
        # QwenImage always uses patch_size=2
        patch_size = 2
        H_lat = x.shape[-2]
        W_lat = x.shape[-1]
        image_seq_len = (H_lat // patch_size) * (W_lat // patch_size)
        # mu: interpolation factor for dynamic shift (same formula as pipeline_qwenimage.py)
        base_seq_len = scheduler.config.get("base_image_seq_len", 256)
        max_seq_len = scheduler.config.get("max_image_seq_len", 8192)
        base_shift = scheduler.config.get("base_shift", 0.5)
        max_shift = scheduler.config.get("max_shift", 0.9)
        m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
        mu = image_seq_len * m + (base_shift - m * base_seq_len)
        sigmas = np.linspace(1.0, 1.0 / num_steps, num_steps)
        scheduler.set_timesteps(sigmas=sigmas, device=self.tensor_kwargs["device"], mu=mu)

        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)

        for step_i, t in enumerate(scheduler.timesteps):
            # QwenTimestepProjEmbeddings uses scale=1000 internally, expects sigma in [0, 1]
            timesteps = (t / scheduler.config.num_train_timesteps) * ones
            # NOTE: no torch.no_grad() here — FSDP2 requires grad context for all_gather.
            # Memory is not a concern since we explicitly detach results below.
            v_cond = net(
                x_B_C_T_H_W=x.to(**self.tensor_kwargs),
                timesteps_B_T=timesteps.to(**self.tensor_kwargs),
                **condition.to_dict(),
            ).float().detach()
            v_uncond = net(
                x_B_C_T_H_W=x.to(**self.tensor_kwargs),
                timesteps_B_T=timesteps.to(**self.tensor_kwargs),
                **uncondition.to_dict(),
            ).float().detach()
            v_pred = v_uncond + self.config.guidance_scale * (v_cond - v_uncond)
            # scheduler.step expects packed format; we work in 5D so use direct Euler step
            sigma_t = float(t) / scheduler.config.num_train_timesteps  # current sigma in [0,1]
            # get next sigma
            if step_i + 1 < len(scheduler.timesteps):
                sigma_next = float(scheduler.timesteps[step_i + 1]) / scheduler.config.num_train_timesteps
            else:
                sigma_next = 0.0
            dt = sigma_next - sigma_t
            x = x + dt * v_pred.to(x.dtype)
        return torch.nan_to_num(x.float())

    # ------------------------------------------------------------------
    # state_dict / load_state_dict — LoRA-only save/load
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        if self.config.lora_enabled:
            lora_state = get_peft_model_state_dict(self.net, adapter_name="default")
            return {"net." + k: v for k, v in lora_state.items()}
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        return net_state_dict

    def load_state_dict(
        self,
        state_dict: Mapping[str, Any],
        strict: bool = True,
        assign: bool = False,
    ):
        if self.config.lora_enabled:
            lora_dict = collections.OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("net."):
                    lora_dict[k.replace("net.", "", 1)] = v
            if lora_dict:
                set_peft_model_state_dict(self.net, lora_dict, adapter_name="default")
            return

        _reg_state_dict = collections.OrderedDict()
        _ema_state_dict = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net."):
                _reg_state_dict[k.replace("net.", "")] = v
            elif k.startswith("net_ema."):
                _ema_state_dict[k.replace("net_ema.", "")] = v

        if strict:
            reg_results: _IncompatibleKeys = self.net.load_state_dict(
                _reg_state_dict, strict=strict, assign=assign
            )
            if self.config.ema.enabled:
                ema_results: _IncompatibleKeys = self.net_ema.load_state_dict(
                    _ema_state_dict, strict=strict, assign=assign
                )
                return _IncompatibleKeys(
                    missing_keys=reg_results.missing_keys + ema_results.missing_keys,
                    unexpected_keys=reg_results.unexpected_keys + ema_results.unexpected_keys,
                )
            return reg_results
        log.critical("load model in non-strict mode")
        log.critical(non_strict_load_model(self.net, _reg_state_dict), rank0_only=False)
        if self.config.ema.enabled:
            log.critical("load ema model in non-strict mode")
            log.critical(non_strict_load_model(self.net_ema, _ema_state_dict), rank0_only=False)
