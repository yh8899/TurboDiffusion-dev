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

"""Wrapper around HuggingFace diffusers QwenImageTransformer2DModel.

Adapts the diffusers-style interface (packed latents, scalar timestep) to the
TurboDiffusion-standard 5-D interface used by WanModel and all other nets:

    forward(x_B_C_T_H_W, timesteps_B_T, crossattn_emb, crossattn_mask, **kwargs)
        → Tensor (B, C, T, H, W)
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor
from torch.distributed._composable.fsdp import fully_shard as fsdp_fully_shard


class QwenImageNetWrapper(nn.Module):
    """Thin wrapper that bridges QwenImageTransformer2DModel to the TurboDiffusion net interface.

    The underlying transformer expects packed latents of shape (B, N, patch_size^2 * C)
    and a scalar timestep in [0, 1].  This wrapper performs the necessary pack / unpack
    conversions and exposes the same forward signature as WanModel.

    Args:
        model_path: Path to the diffusers model directory that contains a
            ``transformer/`` subdirectory with the QwenImageTransformer2DModel weights.
        dtype: Compute dtype string, e.g. ``"bfloat16"`` or ``"float16"``.
        patch_size: Spatial patch size used by _pack_latents / _unpack_latents (default 2).
    """

    def __init__(
        self,
        model_path: str,
        dtype: str = "bfloat16",
        patch_size: int = 2,
    ) -> None:
        super().__init__()
        from diffusers import QwenImageTransformer2DModel

        self._dtype = getattr(torch, dtype)
        self._patch_size = patch_size
        self._model_path = model_path  # saved for init_weights() to load pretrained weights

        # Build model structure only (no weight loading).
        # build_net() wraps this call with torch.device("meta"), so from_pretrained
        # would only record tensor shapes without allocating memory anyway.
        # Actual pretrained weights are loaded in init_weights().
        # Filter out diffusers internal meta keys that are not constructor arguments.
        _meta_keys = {"_class_name", "_diffusers_version", "_name_or_path",
                      "pooled_projection_dim"}
        raw_config = QwenImageTransformer2DModel.load_config(model_path, subfolder="transformer")
        init_kwargs = {k: v for k, v in raw_config.items() if k not in _meta_keys}
        self.transformer: QwenImageTransformer2DModel = QwenImageTransformer2DModel(**init_kwargs)

    # ------------------------------------------------------------------
    # Pack / unpack helpers (mirror QwenImagePipeline._pack/_unpack_latents)
    # ------------------------------------------------------------------

    def _pack_latents(self, latents: Tensor) -> Tensor:
        """(B, C, H, W) → (B, N, patch_size^2 * C) where N = (H/p)*(W/p)."""
        p = self._patch_size
        B, C, H, W = latents.shape
        latents = latents.view(B, C, H // p, p, W // p, p)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        return latents.reshape(B, (H // p) * (W // p), C * p * p)

    def _unpack_latents(self, latents: Tensor, H_lat: int, W_lat: int) -> Tensor:
        """(B, N, patch_size^2 * C) → (B, C, H, W)."""
        p = self._patch_size
        B, _N, packed_dim = latents.shape
        C = packed_dim // (p * p)
        latents = latents.view(B, H_lat // p, W_lat // p, C, p, p)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        return latents.reshape(B, C, H_lat, W_lat)

    # ------------------------------------------------------------------
    # Compatibility stubs expected by T2VModel_SFT.build_net
    # ------------------------------------------------------------------

    def init_weights(self) -> None:
        """Load pretrained weights and materialise non-parameter frequency tensors.

        Called by build_net() after ``net.to_empty("cuda")``.  At that point all
        parameters are uninitialised CUDA tensors.  This method:

        1. Loads the pretrained checkpoint from ``self._model_path`` on CPU and
           copies the state dict into ``self.transformer`` in-place.
        2. Re-computes the RoPE frequency tensors (pos_freqs / neg_freqs), which
           are plain Python attributes (not nn.Parameter) and are therefore not
           moved by to_empty().
        """
        from diffusers import QwenImageTransformer2DModel
        from imaginaire.utils import log

        log.info(f"[QwenImageNetWrapper] Loading pretrained weights from {self._model_path} …")
        pretrained = QwenImageTransformer2DModel.from_pretrained(
            self._model_path, subfolder="transformer"
        )
        # assign=True lets load_state_dict copy CPU tensors into already-allocated CUDA params
        self.transformer.load_state_dict(pretrained.state_dict(), strict=True, assign=False)
        del pretrained

        # Cast parameters to the configured compute dtype.
        self.transformer.to(self._dtype)

        # Re-compute RoPE frequency tensors (plain attrs, not nn.Parameter).
        pe = self.transformer.pos_embed
        pos_index = torch.arange(4096)
        neg_index = torch.arange(4096).flip(0) * -1 - 1
        pe.pos_freqs = torch.cat(
            [
                pe.rope_params(pos_index, pe.axes_dim[0], pe.theta),
                pe.rope_params(pos_index, pe.axes_dim[1], pe.theta),
                pe.rope_params(pos_index, pe.axes_dim[2], pe.theta),
            ],
            dim=1,
        ).to(device="cuda")
        pe.neg_freqs = torch.cat(
            [
                pe.rope_params(neg_index, pe.axes_dim[0], pe.theta),
                pe.rope_params(neg_index, pe.axes_dim[1], pe.theta),
                pe.rope_params(neg_index, pe.axes_dim[2], pe.theta),
            ],
            dim=1,
        ).to(device="cuda")
        log.info("[QwenImageNetWrapper] Pretrained weights loaded successfully.")

    def fully_shard(self, mesh, mp_policy) -> None:
        """Apply FSDP2 fully_shard to the underlying QwenImageTransformer2DModel.

        Strategy (mirrors WanModel.fully_shard):
        - Each ``QwenImageTransformerBlock`` in ``transformer_blocks`` is sharded
          individually with ``reshard_after_forward=True`` to minimise peak memory.
        - Small but non-trivial modules (``txt_in``, ``img_in``, ``time_text_embed``,
          ``txt_norm``, ``norm_out``, ``proj_out``) are also sharded so that their
          parameters participate in the all-gather / reduce-scatter cycle.
        - ``pos_embed`` is intentionally skipped: it holds only pre-computed RoPE
          frequency tensors (no learnable parameters) and is tiny.

        Args:
            mesh: The ``DeviceMesh`` created by ``hsdp_device_mesh``.
            mp_policy: ``MixedPrecisionPolicy`` passed from the trainer.
        """
        t = self.transformer

        # Shard each transformer block individually (largest parameter holders)
        for block in t.transformer_blocks:
            fsdp_fully_shard(block, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)

        # Shard the remaining parameterised sub-modules
        fsdp_fully_shard(t.time_text_embed, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fsdp_fully_shard(t.txt_norm, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fsdp_fully_shard(t.img_in, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fsdp_fully_shard(t.txt_in, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fsdp_fully_shard(t.norm_out, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=True)
        fsdp_fully_shard(t.proj_out, mesh=mesh, mp_policy=mp_policy, reshard_after_forward=False)

    def disable_context_parallel(self) -> None:
        """No-op: QwenImageTransformer has no context parallel implementation."""

    def enable_context_parallel(self, cp_group) -> None:
        """No-op: QwenImageTransformer has no context parallel implementation."""

    # ------------------------------------------------------------------
    # Main forward — TurboDiffusion standard 5-D interface
    # ------------------------------------------------------------------

    def forward(
        self,
        x_B_C_T_H_W: Tensor,
        timesteps_B_T: Tensor,
        crossattn_emb: Tensor,
        crossattn_mask: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        """Forward pass with TurboDiffusion-style 5-D inputs and outputs.

        Args:
            x_B_C_T_H_W: Noisy latent of shape ``(B, C, 1, H_lat, W_lat)``.
                The time dimension T must be 1 (image, not video).
            timesteps_B_T: Timestep tensor of shape ``(B, 1)`` in ``[0, 1]``.
                The transformer internally multiplies by 1000.
            crossattn_emb: Text embeddings ``(B, L, D)`` from the Qwen text encoder.
            crossattn_mask: Optional attention mask ``(B, L)`` where 1 = valid token.
            **kwargs: Ignored; present for interface compatibility.

        Returns:
            Predicted velocity field ``(B, C, 1, H_lat, W_lat)``.
        """
        B, C, T, H_lat, W_lat = x_B_C_T_H_W.shape
        assert T == 1, f"QwenImageNetWrapper expects T=1, got T={T}"

        # (B, C, 1, H, W) → (B, C, H, W)
        x_4d = x_B_C_T_H_W.squeeze(2).to(self._dtype)

        # Pack spatial tokens: (B, C, H, W) → (B, N, p^2*C)
        hidden_states = self._pack_latents(x_4d)

        # timesteps_B_T: (B,), (B, 1), or (B, 1, 1, ...) → flatten to (B,)
        timestep = timesteps_B_T.reshape(B).to(self._dtype)

        # img_shapes: (F, H_tokens, W_tokens) after patch embedding.
        # The latent (H_lat, W_lat) is divided into (H_lat/p, W_lat/p) tokens by _pack_latents.
        p = self._patch_size
        img_shapes: List[Tuple[int, int, int]] = [(1, H_lat // p, W_lat // p)] * B

        # txt_seq_lens: total text sequence length (including padding) per sample.
        # RoPE must cover every token position; attention masking is handled separately
        # via encoder_hidden_states_mask in the attention processor.
        L_txt = crossattn_emb.shape[1]
        txt_seq_lens: List[int] = [L_txt] * B

        output = self.transformer(
            hidden_states=hidden_states,
            encoder_hidden_states=crossattn_emb.to(self._dtype),
            encoder_hidden_states_mask=(
                crossattn_mask.to(self._dtype) if crossattn_mask is not None else None
            ),
            timestep=timestep,
            img_shapes=img_shapes,
            txt_seq_lens=txt_seq_lens,
            return_dict=False,
        )
        # output[0]: (B, N, p^2*C) → (B, C, H, W) → (B, C, 1, H, W)
        unpacked = self._unpack_latents(output[0].to(x_B_C_T_H_W.dtype), H_lat, W_lat)
        return unpacked.unsqueeze(2)
