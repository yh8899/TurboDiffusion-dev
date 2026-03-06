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

"""QwenImage Text-to-Image SFT experiments.

Usage:
    torchrun ... -m scripts.train --config=turbodiffusion/rcm/configs/registry_sft.py \\
        -- experiment=qwen_image_t2i_SFT

Debug (single GPU, 2-layer net placeholder not available yet — use full net with fewer iters):
    torchrun ... -m scripts.train --config=turbodiffusion/rcm/configs/registry_sft.py \\
        -- experiment=qwen_image_t2i_SFT_debug
"""

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyDict


QWEN_IMAGE_T2I_SFT: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "standard"},
            {"override /data_train": "qwen_image_webdataset"},
            {"override /model": "fsdp_t2i_sft_qwen_image"},
            {"override /net": "qwen_image"},
            {"override /conditioner": "qwen_text_nodrop"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_sft",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "qwen_image_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="SFT_QwenImage",
            name="qwen_image_t2i_SFT",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                lora_enabled=True,
                lora_r=128,
                lora_alpha=128,
                lora_dropout=0.0,
                lora_target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                fsdp_shard_size=8,
                state_ch=16,
                state_t=1,
                adjust_video_noise=False,
                timestep_shift=1.0,
                grad_clip=False,
                guidance_scale=5.0,
                pretrained_ckpt="",
                tokenizer=dict(model_path="assets/checkpoints/QwenImage"),
                text_encoder_path="",
                neg_embed_path="",
                p_t=dict(
                    p_mean=0.0,
                    p_std=1.6,
                ),
                precision="bfloat16",
                net=dict(
                    model_path="assets/checkpoints/QwenImage",
                    dtype="bfloat16",
                ),
            )
        ),
        checkpoint=dict(
            save_iter=500,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100_000,
            logging_iter=50,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=6,
                    num_samples=1,
                    run_at_start=True,
                ),
                every_n_sample_ema=dict(
                    every_n=6,
                    num_samples=1,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/qwen_image_t2i/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)


QWEN_IMAGE_T2I_SFT_DEBUG: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "standard"},
            {"override /data_train": "qwen_image_webdataset"},
            {"override /model": "fsdp_t2i_sft_qwen_image"},
            {"override /net": "qwen_image"},
            {"override /conditioner": "qwen_text_nodrop"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                    "viz_online_sampling_sft",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "qwen_image_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="SFT_QwenImage_Debug",
            name="qwen_image_t2i_SFT_debug",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                lora_enabled=True,
                lora_r=8,
                lora_alpha=8,
                lora_dropout=0.0,
                lora_target_modules=["to_q", "to_k", "to_v", "to_out.0"],
                fsdp_shard_size=1,
                state_ch=16,
                state_t=1,
                adjust_video_noise=False,
                timestep_shift=1.0,
                grad_clip=False,
                guidance_scale=5.0,
                pretrained_ckpt="",
                tokenizer=dict(model_path="assets/checkpoints/QwenImage"),
                text_encoder_path="",
                neg_embed_path="",
                p_t=dict(
                    p_mean=0.0,
                    p_std=1.6,
                ),
                precision="bfloat16",
                net=dict(
                    model_path="assets/checkpoints/QwenImage",
                    dtype="bfloat16",
                ),
            )
        ),
        checkpoint=dict(
            save_iter=50,
            load_path="",
            load_training_state=False,
            strict_resume=False,
        ),
        trainer=dict(
            max_iter=100,
            logging_iter=5,
            callbacks=dict(
                every_n_sample_reg=dict(
                    every_n=20,
                    num_samples=1,
                    run_at_start=True,
                ),
            ),
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/qwen_image_t2i/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)


cs = ConfigStore.instance()

cs.store(group="experiment", package="_global_", name="qwen_image_t2i_SFT", node=QWEN_IMAGE_T2I_SFT)
cs.store(
    group="experiment",
    package="_global_",
    name="qwen_image_t2i_SFT_debug",
    node=QWEN_IMAGE_T2I_SFT_DEBUG,
)
