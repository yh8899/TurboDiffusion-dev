"""Quick standalone test to verify QwenImage sampling pipeline.

Run:
    cd /picassox/cephfs/optimization/yh7/Project/TurboDiffusion-dev
    PYTHONPATH=turbodiffusion .venv/bin/python3 scripts/test_qwen_image_inference.py
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "turbodiffusion"))

import torch
from diffusers import QwenImagePipeline

MODEL_PATH = "/simple/Qwen/Qwen-Image"
DEVICE = "cuda"
DTYPE = torch.bfloat16

print("Loading QwenImagePipeline...")
pipe = QwenImagePipeline.from_pretrained(MODEL_PATH, torch_dtype=DTYPE)
pipe = pipe.to(DEVICE)
print("Pipeline loaded!")

prompt = "a blue cartoon dragon pokemon on white background"
print(f"Generating: '{prompt}'")

with torch.no_grad():
    result = pipe(
        prompt=prompt,
        num_inference_steps=20,
        guidance_scale=5.0,
        height=512,
        width=512,
    )

img = result.images[0]
out_path = "/tmp/qwen_image_test.png"
img.save(out_path)
print(f"Saved to {out_path}")
print(f"Image size: {img.size}, mode: {img.mode}")
