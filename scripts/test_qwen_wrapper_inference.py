"""Test QwenImageNetWrapper inference vs official pipeline.

Run:
    cd /picassox/cephfs/optimization/yh7/Project/TurboDiffusion-dev
    PYTHONPATH=turbodiffusion CUDA_VISIBLE_DEVICES=0 .venv/bin/python3 scripts/test_qwen_wrapper_inference.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "turbodiffusion"))

import torch
from PIL import Image
from diffusers import QwenImagePipeline
from rcm.networks.qwen_image import QwenImageNetWrapper
from rcm.tokenizers.qwen_image import QwenImageVAEInterface
from rcm.samplers.euler import FlowEulerSampler
from imaginaire.lazy_config import LazyDict

MODEL_PATH = "/simple/Qwen/Qwen-Image"
DEVICE = "cuda"
DTYPE = torch.bfloat16

print("=== Step 1: Load official pipeline & get embeddings ===")
pipe = QwenImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
pipe = pipe.to(DEVICE)

prompt = "a blue cartoon dragon pokemon on white background"
with torch.no_grad():
    prompt_embeds, prompt_mask = pipe.encode_prompt(
        prompt=[prompt],
        device=DEVICE,
        max_sequence_length=512,
    )
print(f"Official embed: {prompt_embeds.shape}, mask: {prompt_mask.shape}")

# ── Check official scheduler ──────────────────────────────────────────────────
print("\n=== Step 2: Official scheduler info ===")
sched = pipe.scheduler
print(f"Scheduler type: {type(sched).__name__}")
print(f"Scheduler config: shift={getattr(sched.config, 'shift', 'N/A')}, "
      f"use_dynamic_shifting={getattr(sched.config, 'use_dynamic_shifting', 'N/A')}, "
      f"base_shift={getattr(sched.config, 'base_shift', 'N/A')}, "
      f"max_shift={getattr(sched.config, 'max_shift', 'N/A')}")

H, W = 512, 512
H_lat, W_lat = H // 8, W // 8

# Official pipeline uses mu for dynamic shifting
image_seq_len = (H_lat // 2) * (W_lat // 2)
mu = pipe.calculate_shift(image_seq_len)
print(f"image_seq_len={image_seq_len}, mu={mu}")
sched.set_timesteps(20, device=DEVICE, mu=mu)
print(f"Official timesteps[:5]: {sched.timesteps[:5]}")
print(f"Official timesteps[-5:]: {sched.timesteps[-5:]}")

# ── Check our sampler ──────────────────────────────────────────────────────────
print("\n=== Step 3: Our FlowEulerSampler timesteps ===")
our_sampler = FlowEulerSampler(num_train_timesteps=1000, sigma_max=1.0, sigma_min=0.0)
our_sampler.set_timesteps(20, device=DEVICE, shift=1.0)
print(f"Our timesteps[:5]: {our_sampler.timesteps[:5]}")
print(f"Our timesteps[-5:]: {our_sampler.timesteps[-5:]}")

# ── Load our wrapper  ──────────────────────────────────────────────────────────
print("\n=== Step 4: Test QwenImageNetWrapper forward ===")
wrapper_cfg = LazyDict(dict(
    model_path=MODEL_PATH,
    patch_size=2,
    dtype="bfloat16",
))
net = QwenImageNetWrapper(wrapper_cfg)
# Reuse already-loaded transformer to save memory
net.transformer = pipe.transformer
net = net.to(DEVICE)

# test at single timestep
torch.manual_seed(42)
x_latent = torch.randn(1, 16, 1, H_lat, W_lat, device=DEVICE, dtype=DTYPE)
sigma = 0.5
t_for_net = torch.tensor([[sigma]], device=DEVICE, dtype=DTYPE)  # (B, T=1)

# official: pack then call
packed = pipe._pack_latents(x_latent.squeeze(2), 1, 16, 2)  # (1, N, 64)
img_shapes = [(1, H_lat // 2, W_lat // 2)]
t_official = torch.tensor([sigma * 1000], device=DEVICE, dtype=DTYPE)

with torch.no_grad():
    out_official = pipe.transformer(
        hidden_states=packed.to(DTYPE),
        timestep=t_official,
        encoder_hidden_states=prompt_embeds.to(DTYPE),
        img_shapes=img_shapes,
    ).sample  # (1, N, 64)
    # unpack
    unpacked_official = pipe._unpack_latents(out_official, H_lat, W_lat, 2)  # (1, 16, H_lat, W_lat)
    unpacked_official = unpacked_official.unsqueeze(2)  # (1, 16, 1, H_lat, W_lat)
    print(f"Official transformer out: {unpacked_official.shape}, mean={unpacked_official.float().mean():.4f}, std={unpacked_official.float().std():.4f}")

    out_wrapper = net(
        x_B_C_T_H_W=x_latent.to(DTYPE),
        timesteps_B_T=t_for_net.to(DTYPE),
        crossattn_emb=prompt_embeds.to(DTYPE),
        crossattn_mask=prompt_mask.to(DEVICE),
    )
    print(f"Our wrapper out: {out_wrapper.shape}, mean={out_wrapper.float().mean():.4f}, std={out_wrapper.float().std():.4f}")

    diff = (out_wrapper.float() - unpacked_official.float()).abs()
    print(f"Max abs diff: {diff.max():.6f}, mean diff: {diff.mean():.6f}")

# ── Full denoising with our sampler using OFFICIAL shift/mu ───────────────────
print("\n=== Step 5: Full denoising with official shift (mu-based) ===")
# Compute proper shift from official pipeline
shift_sampler = FlowEulerSampler(num_train_timesteps=1000, sigma_max=1.0, sigma_min=0.0)
# Use mu from official pipeline
shift_val = float(sched.timesteps[0]) / 1000.0  # approx sigma_max
print(f"Official first sigma = {float(sched.timesteps[0])/1000:.4f}")
print(f"Official last  sigma = {float(sched.timesteps[-1])/1000:.4f}")

torch.manual_seed(42)
x = torch.randn(1, 16, 1, H_lat, W_lat, device=DEVICE, dtype=torch.float64)
print(f"Init noise: mean={x.mean():.4f} std={x.std():.4f}")

sched.set_timesteps(20, device=DEVICE, mu=mu)
ones = torch.ones(1, 1, device=DEVICE, dtype=torch.float64)

with torch.no_grad():
    for step_i, t in enumerate(sched.timesteps):
        sigma_t = t / 1000.0  # [0,1]
        timesteps_bt = sigma_t * ones
        v = net(
            x_B_C_T_H_W=x.to(DTYPE),
            timesteps_B_T=timesteps_bt.to(DTYPE),
            crossattn_emb=prompt_embeds.to(DTYPE),
            crossattn_mask=prompt_mask.to(DEVICE),
        ).float()
        # Use official scheduler step
        x_packed = pipe._pack_latents(x.squeeze(2).float(), 1, 16, 2)
        v_packed = pipe._pack_latents(v.squeeze(2), 1, 16, 2)
        out_step = sched.step(v_packed.to(torch.float64), t, x_packed.to(torch.float64))
        x = pipe._unpack_latents(out_step.prev_sample.float(), H_lat, W_lat, 2).unsqueeze(2).to(torch.float64)

        if step_i % 5 == 0 or step_i == len(sched.timesteps) - 1:
            print(f"  step {step_i:2d}: t={t:.1f}, v: mean={v.mean():.4f} std={v.std():.4f}, x: mean={x.mean():.4f} std={x.std():.4f}")

print(f"Final latent: mean={x.mean():.4f} std={x.std():.4f}")

# ── Decode ────────────────────────────────────────────────────────────────────
print("\n=== Step 6: Decode with our VAE wrapper ===")
vae = QwenImageVAEInterface(model_path=MODEL_PATH, dtype="bfloat16")
with torch.no_grad():
    pixel = vae.decode(x.float())
print(f"Decoded pixel: {pixel.shape}, range=[{pixel.min():.3f}, {pixel.max():.3f}], mean={pixel.mean():.4f}")

img_arr = ((pixel[0, :, 0].float().clamp(-1, 1) + 1) / 2 * 255).byte().permute(1, 2, 0).cpu().numpy()
img = Image.fromarray(img_arr)
img.save("/tmp/qwen_wrapper_test.png")
print("Saved to /tmp/qwen_wrapper_test.png")
