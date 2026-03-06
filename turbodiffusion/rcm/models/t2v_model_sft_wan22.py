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

"""Wan2.2 I2V SFT with dual high/low noise models, each with independent LoRA and optimizer.

At each training step, one model is randomly selected based on boundary_ratio:
  - high noise model (net_high): trained on timesteps t ∈ [boundary_ratio, 1]
  - low noise model (net_low):   trained on timesteps t ∈ [0, boundary_ratio]

The framework only tracks one optimizer (optimizer_high). optimizer_low is managed
internally in on_before_zero_grad and on_after_backward.
"""

from __future__ import annotations

import collections
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional, Tuple

import attrs
import numpy as np
import scipy.stats
import torch
import torch._dynamo
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from einops import rearrange
from megatron.core import parallel_state
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor
from torch.distributed._composable.fsdp import FSDPModule, MixedPrecisionPolicy, fully_shard
from torch.distributed._tensor.api import DTensor
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict, set_model_state_dict
from torch.nn.modules.module import _IncompatibleKeys

from imaginaire.config import ObjectStoreConfig
from imaginaire.lazy_config import LazyCall as L
from imaginaire.lazy_config import LazyDict
from imaginaire.lazy_config import instantiate as lazy_instantiate
from imaginaire.model import ImaginaireModel
from imaginaire.utils import log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.ema import FastEmaModelUpdater
from rcm.conditioner import DataType, TextCondition
from rcm.configs.defaults.ema import EMAConfig
from rcm.utils.checkpointer import non_strict_load_model
from rcm.utils.context_parallel import broadcast
from rcm.utils.dtensor_helper import DTensorFastEmaModelUpdater, broadcast_dtensor_model_states
from rcm.utils.fsdp_helper import hsdp_device_mesh
from rcm.utils.timestep_utils import LogNormal
from rcm.utils.misc import count_params
from rcm.utils.optim_instantiate_dtensor import get_base_scheduler, get_regular_param_group
from rcm.utils.torch_future import clip_grad_norm_
from rcm.samplers.euler import FlowEulerSampler
from rcm.samplers.unipc import FlowUniPCMultistepSampler

torch._dynamo.config.suppress_errors = True

IS_PREPROCESSED_KEY = "is_preprocessed"
IS_PROCESSED_KEY = "is_processed"

DEFAULT_LORA_TARGET_MODULES = ["q", "k", "v", "o"]


@attrs.define(slots=False)
class T2VConfig_SFT_Wan22:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None

    # low noise model network (also used as the framework-visible "net")
    net: LazyDict = None
    # high noise model network; if None, uses same architecture as net
    net_high: LazyDict = None

    grad_clip: bool = False
    sigma_max: float = 80

    ema: EMAConfig = EMAConfig()
    checkpoint: ObjectStoreConfig = ObjectStoreConfig()
    p_t: LazyDict = L(LogNormal)(p_mean=0.0, p_std=1.6)

    fsdp_shard_size: int = 1
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "videos"
    input_latent_key: str = "latents"
    input_caption_key: str = "prompts"
    input_image_cond_key: str = "image_cond_latents"  # i2v reference frame latents

    loss_scale: float = 1.0
    neg_embed_path: str = ""
    timestep_shift: float = 5.0
    guidance_scale: float = 5.0

    adjust_video_noise: bool = True
    state_ch: int = 16
    state_t: int = 21
    resolution: str = "720p"
    rectified_flow_t_scaling_factor: float = 1000.0
    text_encoder_class: str = "umT5"
    text_encoder_path: str = ""

    # boundary between high and low noise regimes (same as FastVideo)
    boundary_ratio: float = 0.9

    # pretrained DCP checkpoints to load before LoRA injection
    pretrained_ckpt_high: str = ""
    pretrained_ckpt_low: str = ""

    # LoRA – applied to both models independently
    lora_enabled: bool = False
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None  # defaults to DEFAULT_LORA_TARGET_MODULES


class T2VModel_SFT_Wan22(ImaginaireModel):
    """Wan2.2 I2V SFT trainer with dual high/low noise models."""

    def __init__(self, config: T2VConfig_SFT_Wan22):
        super().__init__()
        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.warning(f"T2VModel_SFT_Wan22: precision {self.precision}")

        self.p_t = lazy_instantiate(config.p_t)

        if config.neg_embed_path:
            self.neg_embed = easy_io.load(config.neg_embed_path)
        else:
            self.neg_embed = None

        import math

        if config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        with misc.timer("T2VModel_SFT_Wan22: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == config.state_ch, (
                f"latent_ch {self.tokenizer.latent_ch} != state_ch {config.state_ch}"
            )

        if config.fsdp_shard_size > 1:
            log.info(f"FSDP size: {config.fsdp_shard_size}")
            self.fsdp_device_mesh = hsdp_device_mesh(sharding_group_size=config.fsdp_shard_size)
        else:
            self.fsdp_device_mesh = None

        # flag used during optimizer step to know which model was trained
        self._train_high: bool = True
        # will be set in init_optimizer_scheduler
        self.optimizer_low: Optional[torch.optim.Optimizer] = None
        self.scheduler_low: Optional[torch.optim.lr_scheduler.LRScheduler] = None

        self.set_up_model()

        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

    # ------------------------------------------------------------------
    # Checkpoint helpers
    # ------------------------------------------------------------------

    def load_ckpt_to_net(self, net: torch.nn.Module, ckpt_path: str, prefix: str = "net") -> None:
        storage_reader = FileSystemReader(ckpt_path)
        _state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = set(metadata.state_dict_metadata.keys())
        model_keys = set(_state_dict.keys())
        prefixed_model_keys = {f"{prefix}.{k}" for k in model_keys}

        missing_keys = prefixed_model_keys - checkpoint_keys
        unexpected_keys = checkpoint_keys - prefixed_model_keys

        if missing_keys:
            log.warning(f"Missing keys in checkpoint: {missing_keys}")
        if unexpected_keys:
            log.warning(f"Unexpected keys in checkpoint: {unexpected_keys}")
        if not missing_keys and not unexpected_keys:
            log.info("All keys matched successfully.")

        _new_state_dict = collections.OrderedDict()
        for k in _state_dict.keys():
            if "_extra_state" in k:
                log.warning(k)
            _new_state_dict[f"{prefix}.{k}"] = _state_dict[k]
        dcp.load(_new_state_dict, storage_reader=storage_reader, planner=DefaultLoadPlanner(allow_partial_load=True))
        for k in _state_dict.keys():
            _state_dict[k] = _new_state_dict[f"{prefix}.{k}"]

        log.info(set_model_state_dict(net, _state_dict, options=StateDictOptions(strict=False)))
        del _state_dict, _new_state_dict

    # ------------------------------------------------------------------
    # Model building helpers
    # ------------------------------------------------------------------

    def build_net(self, net_dict: LazyDict, apply_fsdp: bool = True) -> torch.nn.Module:
        with misc.timer("Creating PyTorch model"):
            with torch.device("meta"):
                net = lazy_instantiate(net_dict)
            with misc.timer("meta to cuda and broadcast model states"):
                net.to_empty(device="cuda")
                net.init_weights()

        if apply_fsdp and self.fsdp_device_mesh:
            mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
            net.fully_shard(mesh=self.fsdp_device_mesh, mp_policy=mp_policy)
            net = fully_shard(net, mesh=self.fsdp_device_mesh, mp_policy=mp_policy, reshard_after_forward=True)
            broadcast_dtensor_model_states(net, self.fsdp_device_mesh)
            for name, param in net.named_parameters():
                assert isinstance(param, DTensor), f"param should be DTensor, {name} got {type(param)}"
        return net

    def _inject_lora(self, net: torch.nn.Module) -> torch.nn.Module:
        target_modules = self.config.lora_target_modules
        if not target_modules:
            log.warning("lora_target_modules is empty, using default %s", DEFAULT_LORA_TARGET_MODULES)
            target_modules = DEFAULT_LORA_TARGET_MODULES

        lora_config = LoraConfig(
            r=self.config.lora_r,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            target_modules=target_modules,
            init_lora_weights="gaussian",
        )
        return get_peft_model(net, lora_config)

    def _build_and_prepare_net(
        self,
        net_dict: LazyDict,
        pretrained_ckpt: str,
        ckpt_prefix: str,
    ) -> torch.nn.Module:
        """Build network, optionally load pretrained checkpoint, then inject LoRA."""
        config = self.config
        apply_fsdp = not config.lora_enabled
        net = self.build_net(net_dict, apply_fsdp=apply_fsdp)

        if pretrained_ckpt:
            log.info(f"Loading pretrained checkpoint from {pretrained_ckpt} (prefix={ckpt_prefix})")
            self.load_ckpt_to_net(net, pretrained_ckpt, prefix=ckpt_prefix)

        if config.lora_enabled:
            log.info(f"Injecting LoRA into {ckpt_prefix}")
            net = self._inject_lora(net)
            if self.fsdp_device_mesh:
                mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
                net.fully_shard(mesh=self.fsdp_device_mesh, mp_policy=mp_policy)
                net = fully_shard(net, mesh=self.fsdp_device_mesh, mp_policy=mp_policy, reshard_after_forward=True)
                broadcast_dtensor_model_states(net, self.fsdp_device_mesh)

        return net

    def _enable_cp_for_net(self, net: torch.nn.Module) -> None:
        cp_group = self.get_context_parallel_group()
        base = getattr(net, "base_model", None)
        net_inner = getattr(base, "model", base) if base is not None else net
        if cp_group is not None and cp_group.size() > 1:
            net_inner.enable_context_parallel(cp_group)
        else:
            net_inner.disable_context_parallel()

    @misc.timer("T2VModel_SFT_Wan22: set_up_model")
    def set_up_model(self) -> None:
        config = self.config
        with misc.timer("Creating dual networks"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )

            net_high_dict = config.net_high if config.net_high is not None else config.net
            self.net_high = self._build_and_prepare_net(net_high_dict, config.pretrained_ckpt_high, "net")
            self.net_low = self._build_and_prepare_net(config.net, config.pretrained_ckpt_low, "net")

            self._enable_cp_for_net(self.net_high)
            self._enable_cp_for_net(self.net_low)

            self._param_count_high = count_params(self.net_high, verbose=False)
            self._param_count_low = count_params(self.net_low, verbose=False)

            # EMA not supported for dual-model setup (can be added later)
            if config.ema.enabled:
                log.warning("EMA is not supported for T2VModel_SFT_Wan22 and will be disabled.")
                config.ema.enabled = False

        torch.cuda.empty_cache()

    # ------------------------------------------------------------------
    # Optimizer / scheduler
    # ------------------------------------------------------------------

    def init_optimizer_scheduler(self, optimizer_config: LazyDict, scheduler_config: LazyDict):
        # high noise model: framework-managed optimizer
        optimizer_high = lazy_instantiate(optimizer_config, model=self.net_high)
        scheduler_high = get_base_scheduler(optimizer_high, self, scheduler_config)

        # low noise model: internally managed
        optimizer_low = lazy_instantiate(optimizer_config, model=self.net_low)
        scheduler_low = get_base_scheduler(optimizer_low, self, scheduler_config)

        self.optimizer_low = optimizer_low
        self.scheduler_low = scheduler_low

        return optimizer_high, scheduler_high

    # ------------------------------------------------------------------
    # Training hooks
    # ------------------------------------------------------------------

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net_high = self.net_high.to(memory_format=memory_format, **self.tensor_kwargs)
        self.net_low = self.net_low.to(memory_format=memory_format, **self.tensor_kwargs)

    def on_after_backward(self) -> None:
        """Step the internally managed optimizer for the idle model's zero_grad."""
        pass

    def on_before_zero_grad(
        self,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler,
        iteration: int,
    ) -> None:
        """After the framework steps optimizer_high (or optimizer_low depending on which was active),
        we manually step/zero_grad the other optimizer."""
        if self.optimizer_low is None:
            return

        if self._train_high:
            # framework already stepped optimizer_high; we step optimizer_low here
            self.optimizer_low.step()
            self.scheduler_low.step()
            self.optimizer_low.zero_grad(set_to_none=True)
        else:
            # framework stepped optimizer_high (the framework's optimizer), but we were training
            # the low model - we need to undo the high optimizer step and do the low one instead.
            # Since framework called grad_scaler.step(optimizer_high), that already happened.
            # We step low instead (gradients are on net_low).
            self.optimizer_low.step()
            self.scheduler_low.step()
            self.optimizer_low.zero_grad(set_to_none=True)

    # ------------------------------------------------------------------
    # Timestep sampling
    # ------------------------------------------------------------------

    def _draw_training_time_bounded(self, batch_size: int, train_high: bool) -> torch.Tensor:
        """Sample timesteps from a truncated log-normal restricted to [boundary_ratio, 1] or [0, boundary_ratio].

        The log-normal is defined by p_t (p_mean, p_std in log-sigma space).  We truncate it to
        the target interval by converting the t-bounds to sigma-bounds, then computing the
        corresponding z-score bounds for scipy.stats.truncnorm.

        t <-> sigma via:  sigma = t / (1 - t),  t = sigma / (sigma + 1)
        """
        boundary = self.config.boundary_ratio
        p_mean = self.p_t.p_mean
        p_std = self.p_t.p_std
        eps = 1e-7

        if train_high:
            lo, hi = boundary, 1.0 - eps
        else:
            lo, hi = eps, boundary

        lo_sigma = lo / (1.0 - lo)
        hi_sigma = hi / (1.0 - hi)

        a = (np.log(lo_sigma) - p_mean) / p_std
        b = (np.log(hi_sigma) - p_mean) / p_std

        log_sigma = scipy.stats.truncnorm.rvs(a, b, loc=p_mean, scale=p_std, size=batch_size)
        sigma = np.exp(log_sigma)
        t = sigma / (sigma + 1.0)
        return torch.tensor(t, dtype=torch.float32, device="cuda")

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    def _normalize_video_inplace(self, data_batch: dict[str, Tensor]) -> None:
        input_key = self.config.input_data_key
        if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
            assert torch.is_floating_point(data_batch[input_key])
        else:
            assert data_batch[input_key].dtype == torch.uint8
            data_batch[input_key] = data_batch[input_key].to(**self.tensor_kwargs) / 127.5 - 1.0
            data_batch[IS_PREPROCESSED_KEY] = True

        from torchvision.transforms.v2 import UniformTemporalSubsample

        expected_length = self.tokenizer.get_pixel_num_frames(self.config.state_t)
        original_length = data_batch[input_key].shape[2]
        if original_length != expected_length:
            video = rearrange(data_batch[input_key], "b c t h w -> b t c h w")
            video = UniformTemporalSubsample(expected_length)(video)
            data_batch[input_key] = rearrange(video, "b t c h w -> b c t h w")

    def _normalize_latent_inplace(self, data_batch: dict[str, Tensor]) -> None:
        latents = data_batch[self.config.input_latent_key]
        assert latents.shape[2] >= self.config.state_t
        data_batch[self.config.input_latent_key] = latents[:, :, : self.config.state_t, :, :]

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]):
        if IS_PROCESSED_KEY not in data_batch or not data_batch[IS_PROCESSED_KEY]:
            if self.config.input_latent_key in data_batch:
                self._normalize_latent_inplace(data_batch)
                data_batch[self.config.input_data_key] = (
                    self.decode(data_batch[self.config.input_latent_key]).contiguous().float().clamp(-1, 1)
                )
                data_batch[IS_PREPROCESSED_KEY] = True
            self._normalize_video_inplace(data_batch)
            data_batch[self.config.input_latent_key] = (
                self.encode(data_batch[self.config.input_data_key]).contiguous().float()
            )
            data_batch[IS_PROCESSED_KEY] = True

        raw_state = data_batch[self.config.input_data_key]
        latent_state = data_batch[self.config.input_latent_key]
        if self.neg_embed is not None:
            from einops import repeat

            data_batch["neg_t5_text_embeddings"] = repeat(
                self.neg_embed.to(**self.tensor_kwargs),
                "l d -> b l d",
                b=data_batch["t5_text_embeddings"].shape[0],
            )
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.VIDEO)
        return raw_state, latent_state, condition, uncondition

    def _get_image_cond(self, data_batch: dict[str, Tensor]) -> Optional[Tensor]:
        """Return the image conditioning latent for i2v, or None."""
        key = self.config.input_image_cond_key
        if key in data_batch:
            return data_batch[key].to(**self.tensor_kwargs)
        return None

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _, x0_B_C_T_H_W, condition, _ = self.get_data_and_condition(data_batch)

        # All ranks make the same decision using rank-0's random draw.
        decision = torch.tensor(
            1.0 if torch.rand(1).item() <= self.config.boundary_ratio else 0.0,
            device="cuda",
        )
        dist.broadcast(decision, src=0)
        train_high = decision.item() == 1.0
        self._train_high = train_high

        net = self.net_high if train_high else self.net_low

        B = x0_B_C_T_H_W.shape[0]
        t_B = self._draw_training_time_bounded(B, train_high)  # [B]

        epsilon_B_C_T_H_W = torch.randn_like(x0_B_C_T_H_W)

        # Sync across context-parallel ranks
        cp_group = self.get_context_parallel_group()
        if cp_group is not None and cp_group.size() > 1:
            x0_B_C_T_H_W = broadcast(x0_B_C_T_H_W, cp_group)
            t_B = broadcast(t_B, cp_group)
            epsilon_B_C_T_H_W = broadcast(epsilon_B_C_T_H_W, cp_group)
            condition = condition.broadcast(cp_group)

        # wan2pt2 forward expects timesteps_B_T with shape [B, 1]
        t_B_1 = t_B.unsqueeze(1)  # [B, 1]

        # Noisy input following rectified flow: xt = (1-t)*x0 + t*eps
        t_B_1_1_1_1 = rearrange(t_B, "b -> b 1 1 1 1")
        xt_B_C_T_H_W = (1 - t_B_1_1_1_1) * x0_B_C_T_H_W + t_B_1_1_1_1 * epsilon_B_C_T_H_W

        target_B_C_T_H_W = epsilon_B_C_T_H_W - x0_B_C_T_H_W  # rectified flow velocity

        # i2v image conditioning
        y_B_C_T_H_W = self._get_image_cond(data_batch)

        net_output_B_C_T_H_W = net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(t_B_1 * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
            y_B_C_T_H_W=y_B_C_T_H_W,
            **condition.to_dict(),
        ).float()

        loss = self.config.loss_scale * ((net_output_B_C_T_H_W - target_B_C_T_H_W) ** 2).mean(dim=(1, 2, 3, 4)).mean()

        output_batch = {
            "x0": x0_B_C_T_H_W.detach().cpu(),
            "xt": xt_B_C_T_H_W.detach().cpu(),
            "F_pred": net_output_B_C_T_H_W.detach().cpu(),
            "target": target_B_C_T_H_W.detach().cpu(),
            "train_high": train_high,
        }
        return output_batch, loss

    # ------------------------------------------------------------------
    # Checkpoint state_dict / load_state_dict
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        if self.config.lora_enabled:
            lora_high = get_peft_model_state_dict(self.net_high, adapter_name="default")
            lora_low = get_peft_model_state_dict(self.net_low, adapter_name="default")
            result = {}
            result.update({f"net_high.{k}": v for k, v in lora_high.items()})
            result.update({f"net_low.{k}": v for k, v in lora_low.items()})
            return result
        high_sd = {f"net_high.{k}": v for k, v in self.net_high.state_dict().items()}
        low_sd = {f"net_low.{k}": v for k, v in self.net_low.state_dict().items()}
        high_sd.update(low_sd)
        return high_sd

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        if self.config.lora_enabled:
            lora_high = collections.OrderedDict()
            lora_low = collections.OrderedDict()
            for k, v in state_dict.items():
                if k.startswith("net_high."):
                    lora_high[k.replace("net_high.", "", 1)] = v
                elif k.startswith("net_low."):
                    lora_low[k.replace("net_low.", "", 1)] = v
            if lora_high:
                set_peft_model_state_dict(self.net_high, lora_high, adapter_name="default")
            if lora_low:
                set_peft_model_state_dict(self.net_low, lora_low, adapter_name="default")
            return

        high_sd = collections.OrderedDict()
        low_sd = collections.OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("net_high."):
                high_sd[k.replace("net_high.", "", 1)] = v
            elif k.startswith("net_low."):
                low_sd[k.replace("net_low.", "", 1)] = v
        self.net_high.load_state_dict(high_sd, strict=strict, assign=assign)
        self.net_low.load_state_dict(low_sd, strict=strict, assign=assign)

    # ------------------------------------------------------------------
    # Misc model interface
    # ------------------------------------------------------------------

    def model_dict(self) -> Dict[str, Any]:
        return {"net_high": self.net_high, "net_low": self.net_low}

    def model_param_stats(self) -> Dict[str, int]:
        return {
            "learnable_high": self._param_count_high,
            "learnable_low": self._param_count_low,
        }

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(state) * self.config.sigma_data

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.decode(latent / self.config.sigma_data)

    def get_num_video_latent_frames(self) -> int:
        return self.config.state_t

    @property
    def text_encoder_class(self) -> str:
        return self.config.text_encoder_class

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        return False

    @torch.no_grad()
    def forward(self, xt, t, condition: TextCondition):
        pass

    @torch.no_grad()
    def validation_step(
        self, data: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        pass

    def clip_grad_norm_(
        self,
        max_norm: float,
        norm_type: float = 2.0,
        error_if_nonfinite: bool = False,
        foreach: Optional[bool] = None,
        iteration: int = 0,
    ):
        if not self.config.grad_clip:
            max_norm = 1e12
        # clip both nets
        params = list(self.net_high.parameters()) + list(self.net_low.parameters())
        return clip_grad_norm_(
            params,
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        ).cpu()

    @staticmethod
    def get_context_parallel_group():
        if parallel_state.is_initialized():
            return parallel_state.get_context_parallel_group()
        return None

    def sync(self, *args):
        cp_group = self.get_context_parallel_group()
        cp_size = 1 if cp_group is None else cp_group.size()
        if cp_size > 1:
            out = tuple(
                broadcast(arg, cp_group) if isinstance(arg, torch.Tensor) else arg.broadcast(cp_group)
                for arg in args
            )
        else:
            out = args
        return out[0] if len(out) == 1 else out

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        yield None

    @torch.no_grad()
    def generate_samples_from_batch(
        self,
        data_batch: Dict,
        seed: int = 1,
        teacher: bool = False,
        state_shape: Optional[Tuple] = None,
        n_sample: Optional[int] = None,
        init_noise: Optional[torch.Tensor] = None,
        num_steps: int = 50,
        sampler: str = "UniPC",
        timestep_shift: Optional[float] = None,
    ) -> torch.Tensor:
        """Generate samples by switching from net_high to net_low at boundary_ratio.

        Mirrors the wan2.2_i2v_infer.py logic: start with the high-noise model,
        then switch to the low-noise model once the current timestep t < boundary_ratio.
        """
        _, _, condition, uncondition = self.get_data_and_condition(data_batch)
        input_key = self.config.input_data_key

        if n_sample is None:
            n_sample = data_batch[input_key].shape[0]
        if state_shape is None:
            _T, _H, _W = data_batch[input_key].shape[-3:]
            state_shape = [
                self.config.state_ch,
                self.tokenizer.get_latent_num_frames(_T),
                _H // self.tokenizer.spatial_compression_factor,
                _W // self.tokenizer.spatial_compression_factor,
            ]

        generator = torch.Generator(device=self.tensor_kwargs["device"])
        generator.manual_seed(seed)

        if init_noise is None:
            init_noise = torch.randn(
                n_sample,
                *state_shape,
                dtype=torch.float32,
                device=self.tensor_kwargs["device"],
                generator=generator,
            )

        condition, uncondition = self.sync(condition, uncondition)
        x = init_noise.to(torch.float64)

        if timestep_shift is None:
            timestep_shift = self.config.timestep_shift
        sigma_max = self.config.sigma_max / (self.config.sigma_max + 1)
        unshifted_sigma_max = sigma_max / (timestep_shift - (timestep_shift - 1) * sigma_max)
        samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
        sampler_cls = samplers[sampler](num_train_timesteps=1000, sigma_max=unshifted_sigma_max, sigma_min=0.0)
        sampler_cls.set_timesteps(num_inference_steps=num_steps, device=self.tensor_kwargs["device"], shift=timestep_shift)

        y_B_C_T_H_W = self._get_image_cond(data_batch)
        ones_B_1 = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        boundary = self.config.boundary_ratio

        # Start with high-noise model; switch to low-noise model once t < boundary_ratio
        net = self.net_high
        switched = False

        for _, t in enumerate(sampler_cls.timesteps):
            t_val = t.item() if hasattr(t, "item") else float(t)
            if t_val < boundary and not switched:
                net = self.net_low
                switched = True
                log.info(f"generate_samples_from_batch: switched to net_low at t={t_val:.4f}")

            timesteps = t * ones_B_1
            with torch.no_grad():
                v_cond = net(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs),
                    timesteps_B_T=(timesteps * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                    y_B_C_T_H_W=y_B_C_T_H_W,
                    **condition.to_dict(),
                ).float()
                v_uncond = net(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs),
                    timesteps_B_T=(timesteps * self.config.rectified_flow_t_scaling_factor).to(**self.tensor_kwargs),
                    y_B_C_T_H_W=y_B_C_T_H_W,
                    **uncondition.to_dict(),
                ).float()
            v_pred = v_uncond + self.config.guidance_scale * (v_cond - v_uncond)
            x = sampler_cls.step(v_pred, t, x)

        return torch.nan_to_num(x.float())
