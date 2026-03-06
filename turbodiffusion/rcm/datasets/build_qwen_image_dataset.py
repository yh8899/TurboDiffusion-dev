"""Offline dataset builder for QwenImage training.

Supports two data sources (mutually exclusive):

1. **JSONL** (``--jsonl_path``): each line contains an image path and a prompt.
2. **HuggingFace dataset** (``--hf_dataset``): loaded via ``datasets.load_dataset``.

The field names are configurable via ``--image_column`` and ``--caption_column``
(defaults: ``"image"`` and ``"prompt"``).

For the HuggingFace source, ``--image_column`` can point to either a PIL Image
object (datasets ``Image`` feature) or a string path; both are handled automatically.
Use ``--hf_split`` to select the split (default: ``"train"``), and optionally
``--hf_subset`` to specify the config/subset name.

Encodes images with QwenImageVAEInterface and text with QwenImagePipeline.encode_prompt,
then writes webdataset tar shards containing:

  {key}.latent.pt   — torch tensor (16, 1, H/8, W/8), bfloat16
  {key}.embed.pt    — torch tensor (L, D),             bfloat16  → batch key: "embed"
  {key}.mask.pt     — torch tensor (L,),               int8      → batch key: "mask"
  {key}.prompt.txt  — UTF-8 bytes of the prompt

The conditioner ``input_key`` must be set to ``["embed", "mask"]`` to match these keys.

If ``--negative_prompt`` is provided (non-empty), a single shared file is saved at::

  {output_dir}/neg_embed.pt  — dict {"embed": (1, L, D) bfloat16, "mask": (1, L) int8}

This file can be referenced via ``model.config.neg_embed_path`` during training for CFG.

Usage (JSONL)::

    torchrun --nproc_per_node=8 build_qwen_image_dataset.py \\
        --jsonl_path data/captions.jsonl \\
        --output_dir /output/shards \\
        --model_path /checkpoints/qwen-image \\
        --image_size 512 \\
        --max_sequence_length 512 \\
        --shard_size 1000 \\
        --batch_size 8 \\
        --image_column image \\
        --caption_column prompt \\
        --negative_prompt ""

Usage (HuggingFace dataset)::

    torchrun --nproc_per_node=8 build_qwen_image_dataset.py \\
        --hf_dataset reach-vb/pokemon-blip-captions \\
        --hf_split train \\
        --output_dir /output/shards \\
        --model_path /checkpoints/qwen-image \\
        --image_size 512 \\
        --image_column image \\
        --caption_column caption
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import tarfile
import time
from collections import defaultdict

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from imaginaire.utils import distributed
from rcm.tokenizers.qwen_image import QwenImageVAEInterface


def write_to_tar(tar: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    tar.addfile(info, io.BytesIO(data))


def _tensor_to_bytes(t: torch.Tensor) -> bytes:
    buf = io.BytesIO()
    torch.save(t, buf)
    return buf.getvalue()


def _is_shard_done(path: str) -> bool:
    return os.path.exists(path) and os.path.getsize(path) > 0


def _load_pipeline_text_only(model_path: str, device: torch.device, dtype: torch.dtype):
    """Load QwenImagePipeline with only text_encoder + tokenizer (no VAE, no transformer)."""
    from diffusers import QwenImagePipeline

    pipe = QwenImagePipeline.from_pretrained(
        model_path,
        vae=None,
        transformer=None,
        torch_dtype=dtype,
    )
    pipe.text_encoder = pipe.text_encoder.to(device=device, dtype=dtype)
    pipe.text_encoder.eval().requires_grad_(False)
    return pipe


def _load_image(path: str, image_size: int) -> torch.Tensor:
    """Load and preprocess image from file path → (3, H, W) float32 in [-1, 1]."""
    img = Image.open(path).convert("RGB")
    return _preprocess_pil(img, image_size)


def _preprocess_pil(img: Image.Image, image_size: int) -> torch.Tensor:
    """Preprocess a PIL Image → (3, H, W) float32 in [-1, 1]."""
    img = img.convert("RGB")
    img = TF.resize(img, image_size, interpolation=TF.InterpolationMode.BILINEAR)
    img = TF.center_crop(img, image_size)
    t = TF.to_tensor(img)  # [0, 1]
    t = t * 2.0 - 1.0      # [-1, 1]
    return t


def _load_hf_dataset(
    dataset_name: str,
    split: str,
    subset: str | None,
    image_column: str,
    caption_column: str,
    hf_cache_dir: str | None,
) -> list[dict]:
    """Load a HuggingFace dataset and convert to a list of dicts with 'image' and 'prompt' keys.

    The image value is a PIL Image when the column has the datasets Image feature,
    or a string path otherwise.
    """
    from datasets import load_dataset

    load_kwargs: dict = {"split": split}
    if subset:
        load_kwargs["name"] = subset
    if hf_cache_dir:
        load_kwargs["cache_dir"] = hf_cache_dir

    ds = load_dataset(dataset_name, **load_kwargs)

    records = []
    for row in ds:
        img_val = row[image_column]
        records.append({"_image": img_val, "_prompt": row[caption_column]})
    return records


def _check_resume(tmp_path: str) -> set[int]:
    """Return set of global indices already written in an existing .tmp tar."""
    processed: set[int] = set()
    if not os.path.exists(tmp_path):
        return processed
    try:
        with tarfile.open(tmp_path, "r") as existing:
            files_by_idx: dict[int, list[str]] = defaultdict(list)
            for member in existing.getmembers():
                parts = member.name.split(".", 1)
                if len(parts) == 2 and parts[0].isdigit():
                    files_by_idx[int(parts[0])].append(parts[1])
            for idx, types in files_by_idx.items():
                if all(f in types for f in ("latent.pt", "embed.pt", "mask.pt", "prompt.txt")):
                    processed.add(idx)
    except (tarfile.ReadError, EOFError):
        os.remove(tmp_path)
    return processed


def main(args: argparse.Namespace) -> None:
    distributed.init()
    rank = distributed.get_rank()
    world_size = distributed.get_world_size()
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")

    if args.jsonl_path and args.hf_dataset:
        raise ValueError("Specify either --jsonl_path or --hf_dataset, not both.")
    if not args.jsonl_path and not args.hf_dataset:
        raise ValueError("Must specify one of --jsonl_path or --hf_dataset.")

    # Load records
    use_hf = bool(args.hf_dataset)
    if use_hf:
        if rank == 0:
            print(f"[Rank 0] Loading HuggingFace dataset '{args.hf_dataset}' (split={args.hf_split}) ...")
        records = _load_hf_dataset(
            dataset_name=args.hf_dataset,
            split=args.hf_split,
            subset=args.hf_subset,
            image_column=args.image_column,
            caption_column=args.caption_column,
            hf_cache_dir=args.hf_cache_dir,
        )
    else:
        with open(args.jsonl_path, "r", encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

    total = len(records)
    num_shards = math.ceil(total / args.shard_size)
    my_shards = [i for i in range(num_shards) if i % world_size == rank]

    print(f"[Rank {rank}] {total} records → {num_shards} shards; this rank handles {len(my_shards)}.")

    # Load models
    dtype = torch.bfloat16
    vae = QwenImageVAEInterface(model_path=args.model_path, dtype="bfloat16")
    vae._model = vae._model.to(device=device)

    pipeline = _load_pipeline_text_only(args.model_path, device=device, dtype=dtype)

    os.makedirs(args.output_dir, exist_ok=True)

    # Encode negative prompt (once, shared across all shards)
    if args.negative_prompt:
        neg_embed_path = os.path.join(args.output_dir, "neg_embed.pt")
        if rank == 0 and not os.path.exists(neg_embed_path):
            with torch.no_grad():
                neg_embeds, neg_mask = pipeline.encode_prompt(
                    prompt=[args.negative_prompt],
                    device=device,
                    max_sequence_length=args.max_sequence_length,
                )
            torch.save(
                neg_embeds[0].cpu().to(dtype),
                neg_embed_path,
            )
            print(f"[Rank 0] Saved negative embed → {neg_embed_path}")

    for shard_id in my_shards:
        shard_path = os.path.join(args.output_dir, f"shard_{shard_id:06d}.tar")
        if _is_shard_done(shard_path):
            print(f"[Rank {rank}] Shard {shard_id} already done, skipping.")
            continue

        start = shard_id * args.shard_size
        end = min(total, start + args.shard_size)
        print(f"[Rank {rank}] Building shard {shard_id} ({start}..{end - 1})")

        tmp_path = shard_path + ".tmp"
        processed = _check_resume(tmp_path)
        if processed:
            print(f"[Rank {rank}] Resuming: {len(processed)} items already in tmp.")

        batch_size = args.batch_size
        for batch_start in tqdm(range(start, end, batch_size), disable=(rank != 0)):
            batch_end = min(batch_start + batch_size, end)
            global_indices = list(range(batch_start, batch_end))
            global_indices = [i for i in global_indices if i not in processed]
            if not global_indices:
                continue

            batch_records = [records[i] for i in global_indices]
            if use_hf:
                prompts = [r["_prompt"] for r in batch_records]
                raw_images = [r["_image"] for r in batch_records]
            else:
                prompts = [r[args.caption_column] for r in batch_records]
                raw_images = [r[args.image_column] for r in batch_records]

            # ---- Text encoding ------------------------------------------------
            with torch.no_grad():
                prompt_embeds, prompt_embeds_mask = pipeline.encode_prompt(
                    prompt=prompts,
                    device=device,
                    max_sequence_length=args.max_sequence_length,
                )
            # prompt_embeds: (B, L, D) bfloat16
            # prompt_embeds_mask: (B, L) long

            # ---- Image encoding -----------------------------------------------
            images = []
            for raw in raw_images:
                if isinstance(raw, Image.Image):
                    images.append(_preprocess_pil(raw, args.image_size))
                else:
                    images.append(_load_image(raw, args.image_size))
            pixel_batch = torch.stack(images).to(device=device, dtype=dtype)  # (B, 3, H, W)

            with torch.no_grad():
                latent_batch = vae.encode(pixel_batch)  # (B, 16, 1, H/8, W/8)

            # ---- Write to tar -------------------------------------------------
            try:
                with tarfile.open(tmp_path, "a") as tar:
                    for j, global_idx in enumerate(global_indices):
                        key = f"{global_idx:09d}"

                        # latent: (16, 1, H/8, W/8)
                        latent = latent_batch[j].cpu().to(dtype)
                        write_to_tar(tar, f"{key}.latent.pt", _tensor_to_bytes(latent))

                        # embed: (L, D)
                        embed = prompt_embeds[j].cpu().to(dtype)
                        write_to_tar(tar, f"{key}.embed.pt", _tensor_to_bytes(embed))

                        # mask: (L,) stored as int8 to save space
                        mask = prompt_embeds_mask[j].cpu().to(torch.int8)
                        write_to_tar(tar, f"{key}.mask.pt", _tensor_to_bytes(mask))

                        write_to_tar(tar, f"{key}.prompt.txt", prompts[j].encode("utf-8"))

                        processed.add(global_idx)
            except Exception as exc:
                print(f"[Rank {rank}] Write error for shard {shard_id}: {exc}")
                raise

            del latent_batch, prompt_embeds, prompt_embeds_mask, pixel_batch

        os.rename(tmp_path, shard_path)
        print(f"[Rank {rank}] Finished shard {shard_id} → {shard_path}")

    print(f"[Rank {rank}] All done.")
    # Wait forever (barrier) so torchrun doesn't kill other ranks prematurely
    if rank == 0:
        time.sleep(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build QwenImage webdataset from JSONL or HuggingFace dataset")
    # --- Data source (specify exactly one) ---
    parser.add_argument("--jsonl_path", type=str, default=None, help="Input JSONL file (image + prompt per line)")
    parser.add_argument("--hf_dataset", type=str, default=None, help="HuggingFace dataset name (e.g. 'reach-vb/pokemon-blip-captions')")
    # --- HuggingFace-specific options ---
    parser.add_argument("--hf_split", type=str, default="train", help="HuggingFace dataset split (default: train)")
    parser.add_argument("--hf_subset", type=str, default=None, help="HuggingFace dataset config/subset name")
    parser.add_argument("--hf_cache_dir", type=str, default=None, help="Local cache directory for HuggingFace datasets")
    # --- Common options ---
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory for tar shards")
    parser.add_argument("--model_path", type=str, required=True, help="QwenImage model root (contains vae/, text_encoder/ etc.)")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Max token length for encode_prompt")
    parser.add_argument("--image_size", type=int, default=512, help="Center-crop size for images")
    parser.add_argument("--shard_size", type=int, default=1000, help="Samples per tar shard")
    parser.add_argument("--batch_size", type=int, default=8, help="Encoding batch size per iteration")
    parser.add_argument("--image_column", type=str, default="image", help="JSONL field name for the image path")
    parser.add_argument("--caption_column", type=str, default="prompt", help="JSONL field name for the caption/prompt")
    parser.add_argument("--negative_prompt", type=str, default="", help="Negative prompt for CFG; if non-empty, encodes and saves to {output_dir}/neg_embed.pt")
    args = parser.parse_args()

    main(args)
