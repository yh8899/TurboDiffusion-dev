"""Visualise latents stored in a QwenImage webdataset tar shard.

Each latent is decoded back to a pixel image using QwenImageVAEInterface and
the results are arranged into a grid saved as a PNG (or JPEG).

The tar is expected to contain files built by ``build_qwen_image_dataset.py``:

  {key}.latent.pt   — torch tensor (16, 1, H/8, W/8), bfloat16
  {key}.embed.pt    — (optional) text embedding
  {key}.mask.pt     — (optional) attention mask
  {key}.prompt.txt  — (optional) UTF-8 caption

Usage::

    python qwen_image_visualize_tar.py \\
        --tar_path /path/to/shard_000000.tar \\
        --model_path /checkpoints/QwenImage \\
        --save_path preview.png \\
        --num_samples 16 \\
        --grid_cols 4
"""

from __future__ import annotations

import argparse
import io
import os
import tarfile

import torch
import torchvision.utils as vutils
from tqdm import tqdm

from rcm.tokenizers.qwen_image import QwenImageVAEInterface


def main(args: argparse.Namespace) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading QwenImageVAEInterface from {args.model_path} ...")
    vae = QwenImageVAEInterface(model_path=args.model_path, dtype="bfloat16")
    vae._model = vae._model.to(device)

    if not os.path.exists(args.tar_path):
        print(f"Error: tar file not found: {args.tar_path}")
        return

    # Collect latent member names
    with tarfile.open(args.tar_path, "r") as tar:
        latent_names = sorted(n for n in tar.getnames() if n.endswith(".latent.pt"))

    if not latent_names:
        print("No .latent.pt files found in the tar archive.")
        return

    print(f"Found {len(latent_names)} latent(s) in {args.tar_path}.")

    if args.num_samples > 0:
        latent_names = latent_names[: args.num_samples]
        print(f"Decoding the first {len(latent_names)} sample(s).")

    decoded_images: list[torch.Tensor] = []
    captions: list[str] = []

    with tarfile.open(args.tar_path, "r") as tar:
        all_names = set(tar.getnames())
        for member_name in tqdm(latent_names, desc="Decoding"):
            # Load latent
            fobj = tar.extractfile(member_name)
            if fobj is None:
                print(f"Warning: cannot extract {member_name}, skipping.")
                continue
            latent = torch.load(io.BytesIO(fobj.read()), map_location=device)

            # latent shape from dataset: (16, 1, H_lat, W_lat)
            # QwenImageVAEInterface.decode expects (B, 16, 1, H_lat, W_lat)
            if latent.dim() == 4:
                latent = latent.unsqueeze(0)  # add batch dim → (1, 16, 1, H, W)

            with torch.no_grad():
                pixel = vae.decode(latent.to(device, dtype=torch.bfloat16))
            # pixel: (1, 3, 1, H_px, W_px) — squeeze T and B
            img = pixel.squeeze(0).squeeze(1).float().cpu()  # (3, H, W)
            decoded_images.append(img)

            # Optionally load caption
            key = member_name.split(".latent.pt")[0]
            caption_name = f"{key}.prompt.txt"
            if caption_name in all_names:
                txt_obj = tar.extractfile(caption_name)
                if txt_obj is not None:
                    captions.append(txt_obj.read().decode("utf-8", errors="replace"))
                else:
                    captions.append("")
            else:
                captions.append("")

    if not decoded_images:
        print("No images decoded. Exiting.")
        return

    # Print captions
    for i, cap in enumerate(captions):
        if cap:
            print(f"[{i:03d}] {cap[:120]}")

    # Normalise to [0, 1]
    grid_tensors = [(t.clamp(-1, 1) + 1.0) / 2.0 for t in decoded_images]

    # Pad to a full grid if needed
    n = len(grid_tensors)
    cols = args.grid_cols
    rows = (n + cols - 1) // cols
    pad_count = rows * cols - n
    if pad_count > 0:
        h, w = grid_tensors[0].shape[-2:]
        grid_tensors += [torch.zeros(3, h, w)] * pad_count

    grid = vutils.make_grid(grid_tensors, nrow=cols, padding=2, pad_value=0.5)

    save_dir = os.path.dirname(args.save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)

    from torchvision.transforms.functional import to_pil_image
    pil_img = to_pil_image(grid.clamp(0, 1))
    pil_img.save(args.save_path)
    print(f"Saved {n} image(s) ({rows}×{cols} grid) → {args.save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualise QwenImage latents from a webdataset tar shard.")
    parser.add_argument("--tar_path", type=str, required=True, help="Path to the tar shard.")
    parser.add_argument("--model_path", type=str, required=True, help="QwenImage model root (contains vae/ subdirectory).")
    parser.add_argument("--save_path", type=str, default="preview.png", help="Output image path (PNG or JPEG).")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of latents to decode. 0 = all.")
    parser.add_argument("--grid_cols", type=int, default=4, help="Number of columns in the output grid.")
    args = parser.parse_args()
    main(args)
