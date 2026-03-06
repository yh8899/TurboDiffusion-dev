"""QwenImage VAE interface wrapping diffusers AutoencoderKLQwenImage.

AutoencoderKLQwenImage is a 3D causal VAE fine-tuned from the Wan Video VAE for
image generation.  Compared to Wan2pt1VAEInterface the key differences are:

* The model is loaded via `from_pretrained` (HuggingFace checkpoint directory).
* `encode` returns `AutoencoderKLOutput`; latent normalisation must be applied
  externally (stored mean/std are read from the model config).
* `decode` returns `DecoderOutput`; the sample tensor is already clamped to
  [-1, 1].
* Temporal compression factor is 1 (image-only: T always = 1 after encode).
* Spatial compression factor is 8 (same as Wan2pt1).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .interface import VideoTokenizerInterface


class QwenImageVAEInterface(VideoTokenizerInterface):
    """Wraps ``diffusers.AutoencoderKLQwenImage`` as a ``VideoTokenizerInterface``.

    Args:
        model_path: Path to the HuggingFace model directory that contains a
            ``vae`` subfolder with the AutoencoderKLQwenImage config and weights.
        dtype: PyTorch dtype string (``"bfloat16"`` or ``"float16"``).  The VAE
            runs in this dtype; encode/decode cast inputs/outputs accordingly.
    """

    def __init__(self, model_path: str = "", dtype: str = "bfloat16", **kwargs) -> None:
        from diffusers import AutoencoderKLQwenImage

        self._dtype = getattr(torch, dtype)
        self._model: nn.Module = (
            AutoencoderKLQwenImage.from_pretrained(model_path, subfolder="vae")
            .eval()
            .requires_grad_(False)
            .to(device="cuda", dtype=self._dtype)
        )

        # Latent normalisation parameters stored in the model config.
        # Shape: (z_dim,) — applied per-channel via broadcast over (B, C, T, H, W).
        cfg = self._model.config
        self._latents_mean = torch.tensor(cfg.latents_mean, dtype=torch.float32)
        self._latents_std = torch.tensor(cfg.latents_std, dtype=torch.float32)

    # ------------------------------------------------------------------
    # VideoTokenizerInterface — required abstract methods
    # ------------------------------------------------------------------

    def reset_dtype(self) -> None:
        pass

    @torch.no_grad()
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode pixel frames to latents.

        Args:
            state: ``(B, 3, T, H, W)`` or ``(B, 3, H, W)`` in range ``[-1, 1]``.
                   Single images (4-D) are unsqueezed to add a T=1 dimension.

        Returns:
            Normalised latent tensor of shape ``(B, 16, T, H//8, W//8)``
            where T is preserved (temporal_compression_factor=1).
        """
        if state.ndim == 4:
            state = state.unsqueeze(2)

        x = state.to(self._dtype)
        output = self._model.encode(x)
        z = output.latent_dist.sample()  # (B, 16, T, H/8, W/8), fp dtype

        # Normalise: z_norm = (z - mean) / std
        mean = self._latents_mean.to(device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        std = self._latents_std.to(device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        return (z - mean) / std

    @torch.no_grad()
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latents back to pixel frames.

        Args:
            latent: Normalised latent ``(B, 16, T, H//8, W//8)``.

        Returns:
            Reconstructed frames ``(B, 3, T, H, W)`` in range ``[-1, 1]``.
        """
        z = latent.to(self._dtype)

        # Inverse normalise: z_raw = z * std + mean
        mean = self._latents_mean.to(device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        std = self._latents_std.to(device=z.device, dtype=z.dtype).view(1, -1, 1, 1, 1)
        z_raw = z * std + mean

        output = self._model.decode(z_raw)
        return output.sample  # (B, 3, T, H, W), clamped to [-1, 1] by the model

    def get_latent_num_frames(self, num_pixel_frames: int) -> int:
        # Image VAE: no temporal compression — latent T = pixel T
        return num_pixel_frames

    def get_pixel_num_frames(self, num_latent_frames: int) -> int:
        return num_latent_frames

    # ------------------------------------------------------------------
    # VideoTokenizerInterface — required properties
    # ------------------------------------------------------------------

    @property
    def spatial_compression_factor(self) -> int:
        return 8

    @property
    def temporal_compression_factor(self) -> int:
        # Image VAE: time dimension is not compressed
        return 1

    @property
    def spatial_resolution(self) -> int:
        return 512

    @property
    def pixel_chunk_duration(self) -> int:
        return 1

    @property
    def latent_chunk_duration(self) -> int:
        return 1

    @property
    def latent_ch(self) -> int:
        return 16

    @property
    def dtype(self):
        return self._dtype

    @property
    def name(self) -> str:
        return "qwen_image_tokenizer"
