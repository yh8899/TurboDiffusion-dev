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

from __future__ import annotations

import collections
import math
from contextlib import contextmanager
from typing import Any, Dict, List, Mapping, Optional, Tuple

import attrs
import numpy as np
import torch
import torch._dynamo
import torch.distributed.checkpoint as dcp
from einops import rearrange, repeat
from megatron.core import parallel_state
from peft import LoraConfig, get_peft_model
from peft.utils import get_peft_model_state_dict, set_peft_model_state_dict
from torch import Tensor
from torch.distributed._composable.fsdp import FSDPModule, fully_shard, MixedPrecisionPolicy
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
from rcm.utils.optim_instantiate_dtensor import get_base_scheduler
from rcm.utils.timestep_utils import LogNormal
from rcm.utils.checkpointer import non_strict_load_model
from rcm.utils.context_parallel import broadcast
from rcm.utils.dtensor_helper import DTensorFastEmaModelUpdater, broadcast_dtensor_model_states
from rcm.utils.fsdp_helper import hsdp_device_mesh
from rcm.utils.misc import count_params
from rcm.utils.torch_future import clip_grad_norm_
from rcm.configs.defaults.ema import EMAConfig
from rcm.samplers.euler import FlowEulerSampler
from rcm.samplers.unipc import FlowUniPCMultistepSampler

torch._dynamo.config.suppress_errors = True

IS_PREPROCESSED_KEY = "is_preprocessed"
IS_PROCESSED_KEY = "is_processed"

# Default LoRA target modules for Wan (WanSelfAttention / WanT2VCrossAttention: q, k, v, o)
DEFAULT_LORA_TARGET_MODULES = ["q", "k", "v", "o"]


@attrs.define(slots=False)
class T2VConfig_SFT:
    tokenizer: LazyDict = None
    conditioner: LazyDict = None
    net: LazyDict = None
    grad_clip: bool = False
    sigma_max: float = 80

    ema: EMAConfig = EMAConfig()
    checkpoint: ObjectStoreConfig = ObjectStoreConfig()
    p_t: LazyDict = L(LogNormal)(
        p_mean=0.0,
        p_std=1.6,
    )
    fsdp_shard_size: int = 1
    sigma_data: float = 1.0
    precision: str = "bfloat16"
    input_data_key: str = "videos"
    input_latent_key: str = "latents"
    input_caption_key: str = "prompts"
    loss_scale: float = 1.0
    neg_embed_path: str = ""
    timestep_shift: float = 5
    guidance_scale: float = 5.0  # for sampling (classifier-free guidance)

    adjust_video_noise: bool = True
    state_ch: int = 16
    state_t: int = 21
    resolution: str = "480p"
    rectified_flow_t_scaling_factor: float = 1000.0
    text_encoder_class: str = "umT5"
    text_encoder_path: str = ""

    # Pretrained DCP checkpoint to load into net before (optional) LoRA injection.
    pretrained_ckpt: str = ""

    # LoRA (only when lora_enabled=True)
    lora_enabled: bool = False
    lora_r: int = 128
    lora_alpha: int = 128
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None  # default DEFAULT_LORA_TARGET_MODULES


class T2VModel_SFT(ImaginaireModel):
    def __init__(self, config: T2VConfig_SFT):
        super().__init__()
        self.config = config

        self.precision = {
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[config.precision]
        self.tensor_kwargs = {"device": "cuda", "dtype": self.precision}
        log.warning(f"DiffusionModel: precision {self.precision}")

        self.p_t = lazy_instantiate(config.p_t)
        if config.neg_embed_path:
            self.neg_embed = easy_io.load(config.neg_embed_path)
        else:
            self.neg_embed = None

        if config.adjust_video_noise:
            self.video_noise_multiplier = math.sqrt(config.state_t)
        else:
            self.video_noise_multiplier = 1.0

        with misc.timer("DiffusionModel: set_up_tokenizer"):
            self.tokenizer = lazy_instantiate(config.tokenizer)
            assert self.tokenizer.latent_ch == config.state_ch, (
                f"latent_ch {self.tokenizer.latent_ch} != state_shape {config.state_ch}"
            )

        if config.fsdp_shard_size > 1:
            log.info(f"FSDP size: {config.fsdp_shard_size}")
            self.fsdp_device_mesh = hsdp_device_mesh(sharding_group_size=config.fsdp_shard_size)
        else:
            self.fsdp_device_mesh = None

        self.set_up_model()

        if parallel_state.is_initialized():
            self.data_parallel_size = parallel_state.get_data_parallel_world_size()
        else:
            self.data_parallel_size = 1

    def load_ckpt_to_net(self, net, ckpt_path, prefix="net"):
        storage_reader = FileSystemReader(ckpt_path)
        _state_dict = get_model_state_dict(net)

        metadata = storage_reader.read_metadata()
        checkpoint_keys = metadata.state_dict_metadata.keys()

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

    def build_net(self, net_dict: LazyDict, apply_fsdp: bool = True) -> torch.nn.Module:
        init_device = "meta"
        with misc.timer("Creating PyTorch model"):
            with torch.device(init_device):
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
        # PEFT 已在 inject adapter 时通过 _mark_only_adapters_as_trainable 将 base 置为 requires_grad=False，
        # 仅 adapter 参数为 True，无需再手动设置。
        net = get_peft_model(net, lora_config)
        return net

    @misc.timer("DiffusionModel: set_up_model")
    def set_up_model(self):
        config = self.config
        with misc.timer("Creating PyTorch model and ema if enabled"):
            self.conditioner = lazy_instantiate(config.conditioner)
            assert sum(p.numel() for p in self.conditioner.parameters() if p.requires_grad) == 0, (
                "conditioner should not have learnable parameters"
            )

            apply_fsdp = not config.lora_enabled
            self.net = self.build_net(config.net, apply_fsdp=apply_fsdp)

            if config.pretrained_ckpt:
                log.info(f"Loading pretrained checkpoint from {config.pretrained_ckpt}")
                self.load_ckpt_to_net(self.net, config.pretrained_ckpt)

            if config.lora_enabled:
                log.info("Injecting LoRA and freezing base")
                self.net = self._inject_lora(self.net)
                if self.fsdp_device_mesh:
                    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
                    self.net.fully_shard(mesh=self.fsdp_device_mesh, mp_policy=mp_policy)
                    self.net = fully_shard(
                        self.net, mesh=self.fsdp_device_mesh, mp_policy=mp_policy, reshard_after_forward=True
                    )
                    broadcast_dtensor_model_states(self.net, self.fsdp_device_mesh)

            self._param_count = count_params(self.net, verbose=False)

            cp_group = self.get_context_parallel_group()
            net_for_cp = getattr(self.net, "base_model", None)
            net_for_cp = getattr(net_for_cp, "model", net_for_cp) if net_for_cp is not None else self.net
            if cp_group is not None and cp_group.size() > 1:
                net_for_cp.enable_context_parallel(cp_group)
            else:
                net_for_cp.disable_context_parallel()

            if config.ema.enabled:
                self.net_ema = self.build_net(config.net, apply_fsdp=False)
                if config.lora_enabled:
                    self.net_ema = self._inject_lora(self.net_ema)
                if self.fsdp_device_mesh:
                    mp_policy = MixedPrecisionPolicy(reduce_dtype=torch.float32)
                    self.net_ema.fully_shard(mesh=self.fsdp_device_mesh, mp_policy=mp_policy)
                    self.net_ema = fully_shard(
                        self.net_ema, mesh=self.fsdp_device_mesh, mp_policy=mp_policy, reshard_after_forward=True
                    )
                    broadcast_dtensor_model_states(self.net_ema, self.fsdp_device_mesh)
                self.net_ema.requires_grad_(False)
                if self.fsdp_device_mesh:
                    self.net_ema_worker = DTensorFastEmaModelUpdater()
                else:
                    self.net_ema_worker = FastEmaModelUpdater()
                s = config.ema.rate
                self.ema_exp_coefficient = np.roots([1, 7, 16 - s**-2, 12 - s**-2]).real.max()
                self.net_ema_worker.copy_to(src_model=self.net, tgt_model=self.net_ema)
        torch.cuda.empty_cache()

    def init_optimizer_scheduler(self, optimizer_config: LazyDict, scheduler_config: LazyDict):
        net_optimizer = lazy_instantiate(optimizer_config, model=self.net)
        net_scheduler = get_base_scheduler(net_optimizer, self, scheduler_config)
        return net_optimizer, net_scheduler

    def on_before_zero_grad(
        self, optimizer: torch.optim.Optimizer, scheduler: torch.optim.lr_scheduler.LRScheduler, iteration: int
    ) -> None:
        del scheduler, optimizer
        if self.config.ema.enabled:
            ema_beta = self.ema_beta(iteration)
            self.net_ema_worker.update_average(self.net, self.net_ema, beta=ema_beta)

    def on_train_start(self, memory_format: torch.memory_format = torch.preserve_format) -> None:
        if self.config.ema.enabled:
            self.net_ema.to(dtype=torch.float32)
        if hasattr(self.tokenizer, "reset_dtype"):
            self.tokenizer.reset_dtype()
        self.net = self.net.to(memory_format=memory_format, **self.tensor_kwargs)

    def _sample_rf_time(self, sampler, time_shape: Any) -> torch.Tensor:
        assert isinstance(time_shape, (int, tuple, list, torch.Size)), f"Unsupported time shape type: {type(time_shape)}"
        sampled = sampler(shape=time_shape, device="cuda", dtype=torch.float64)
        domain = getattr(sampler, "output_domain", "rf")
        assert domain == "rf", f"Expected RF-domain timestep sampler, got {domain}"
        return sampled.clamp(min=0.0, max=1.0)

    def draw_training_time(self, time_shape: Any, condition: Any = None) -> torch.Tensor:
        time_B_1 = self._sample_rf_time(self.p_t, time_shape)
        if condition is not None:
            is_video_batch = condition.data_type == DataType.VIDEO
            multiplier = self.video_noise_multiplier if is_video_batch else 1
            if multiplier != 1:
                # convert to sigma, scale, convert back
                sigma = time_B_1 / (1.0 - time_B_1)
                sigma = sigma * multiplier
                time_B_1 = sigma / (sigma + 1.0)
        return time_B_1.float()

    def training_step(
        self, data_batch: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        _, x0_B_C_T_H_W, condition, _uncondition = self.get_data_and_condition(data_batch)

        time_B_T = self.draw_training_time((x0_B_C_T_H_W.shape[0], 1), condition)
        epsilon_B_C_T_H_W = torch.randn(x0_B_C_T_H_W.size(), device="cuda")
        x0_B_C_T_H_W, time_B_T, epsilon_B_C_T_H_W, condition = self.sync(
            x0_B_C_T_H_W, time_B_T, epsilon_B_C_T_H_W, condition
        )

        time_B_1_T_1_1 = rearrange(time_B_T, "b t -> b 1 t 1 1")
        xt_B_C_T_H_W = (1 - time_B_1_T_1_1) * x0_B_C_T_H_W + time_B_1_T_1_1 * epsilon_B_C_T_H_W

        target_B_C_T_H_W = epsilon_B_C_T_H_W - x0_B_C_T_H_W  # rectified flow velocity

        net_output_B_C_T_H_W = self.net(
            x_B_C_T_H_W=xt_B_C_T_H_W.to(**self.tensor_kwargs),
            timesteps_B_T=(time_B_1_T_1_1 * self.config.rectified_flow_t_scaling_factor)
            .squeeze(dim=[1, 3, 4])
            .to(**self.tensor_kwargs),
            **condition.to_dict(),
        ).float()

        loss = self.config.loss_scale * ((net_output_B_C_T_H_W - target_B_C_T_H_W) ** 2).mean(dim=(1, 2, 3, 4)).mean()

        output_batch = {
            "x0": x0_B_C_T_H_W.detach().cpu(),
            "xt": xt_B_C_T_H_W.detach().cpu(),
            "F_pred": net_output_B_C_T_H_W.detach().cpu(),
            "target": target_B_C_T_H_W.detach().cpu(),
        }
        return output_batch, loss

    @torch.no_grad()
    def forward(self, xt, t, condition: TextCondition):
        pass

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

        if timestep_shift is None:
            timestep_shift = self.config.timestep_shift
        sigma_max = self.config.sigma_max / (self.config.sigma_max + 1)
        unshifted_sigma_max = sigma_max / (timestep_shift - (timestep_shift - 1) * sigma_max)
        samplers = {"Euler": FlowEulerSampler, "UniPC": FlowUniPCMultistepSampler}
        sampler_cls = samplers[sampler](num_train_timesteps=1000, sigma_max=unshifted_sigma_max, sigma_min=0.0)
        sampler_cls.set_timesteps(
            num_inference_steps=num_steps, device=self.tensor_kwargs["device"], shift=timestep_shift
        )
        ones = torch.ones(x.size(0), 1, device=x.device, dtype=x.dtype)
        for _, t in enumerate(sampler_cls.timesteps):
            timesteps = t * ones
            with torch.no_grad():
                v_cond = net(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs),
                    timesteps_B_T=timesteps.to(**self.tensor_kwargs),
                    **condition.to_dict(),
                ).float()
                v_uncond = net(
                    x_B_C_T_H_W=x.to(**self.tensor_kwargs),
                    timesteps_B_T=timesteps.to(**self.tensor_kwargs),
                    **uncondition.to_dict(),
                ).float()
            v_pred = v_uncond + self.config.guidance_scale * (v_cond - v_uncond)
            x = sampler_cls.step(v_pred, t, x)
        samples = x.float()
        return torch.nan_to_num(samples)

    @torch.no_grad()
    def validation_step(
        self, data: dict[str, torch.Tensor], iteration: int
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
        pass

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
                broadcast(arg, cp_group) if isinstance(arg, torch.Tensor) else arg.broadcast(cp_group) for arg in args
            )
        else:
            out = args
        return out[0] if len(out) == 1 else out

    def _normalize_video_inplace(self, data_batch: dict[str, Tensor]) -> None:
        input_key = self.config.input_data_key
        if IS_PREPROCESSED_KEY in data_batch and data_batch[IS_PREPROCESSED_KEY] is True:
            assert torch.is_floating_point(data_batch[input_key])
            assert torch.all(
                (data_batch[input_key] >= -1.0001) & (data_batch[input_key] <= 1.0001)
            ), f"Video data range [{data_batch[input_key].min()}, {data_batch[input_key].max()}]"
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

    def get_data_and_condition(self, data_batch: dict[str, torch.Tensor]) -> Tuple[Tensor, Tensor, Any, Any]:
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
            data_batch["neg_t5_text_embeddings"] = repeat(
                self.neg_embed.to(**self.tensor_kwargs), "l d -> b l d", b=data_batch["t5_text_embeddings"].shape[0]
            )
            condition, uncondition = self.conditioner.get_condition_with_negative_prompt(data_batch)
        else:
            condition, uncondition = self.conditioner.get_condition_uncondition(data_batch)
        condition = condition.edit_data_type(DataType.VIDEO)
        uncondition = uncondition.edit_data_type(DataType.VIDEO)
        return raw_state, latent_state, condition, uncondition

    def model_dict(self) -> Dict[str, Any]:
        return {"net": self.net}

    def state_dict(self) -> Dict[str, Any]:
        if self.config.lora_enabled:
            lora_state = get_peft_model_state_dict(self.net, adapter_name="default")
            return {"net." + k: v for k, v in lora_state.items()}
        net_state_dict = self.net.state_dict(prefix="net.")
        if self.config.ema.enabled:
            ema_state_dict = self.net_ema.state_dict(prefix="net_ema.")
            net_state_dict.update(ema_state_dict)
        return net_state_dict

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
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
            reg_results: _IncompatibleKeys = self.net.load_state_dict(_reg_state_dict, strict=strict, assign=assign)
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

    def ema_beta(self, iteration: int) -> float:
        iteration = iteration + self.config.ema.iteration_shift
        if iteration < 1:
            return 0.0
        return (1 - 1 / (iteration + 1)) ** (self.ema_exp_coefficient + 1)

    def model_param_stats(self) -> Dict[str, int]:
        return {"total_learnable_param_num": self._param_count}

    def is_image_batch(self, data_batch: dict[str, Tensor]) -> bool:
        return False

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

    @contextmanager
    def ema_scope(self, context=None, is_cpu=False):
        if self.config.ema.enabled:
            for module in self.net.modules():
                if isinstance(module, FSDPModule):
                    module.reshard()
            self.net_ema_worker.cache(self.net.parameters(), is_cpu=is_cpu)
            self.net_ema_worker.copy_to(src_model=self.net_ema, tgt_model=self.net)
            if context is not None:
                log.info(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.config.ema.enabled:
                for module in self.net.modules():
                    if isinstance(module, FSDPModule):
                        module.reshard()
                self.net_ema_worker.restore(self.net.parameters())
                if context is not None:
                    log.info(f"{context}: Restored training weights")

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
        return clip_grad_norm_(
            self.net.parameters(),
            max_norm=max_norm,
            norm_type=norm_type,
            error_if_nonfinite=error_if_nonfinite,
            foreach=foreach,
        ).cpu()
