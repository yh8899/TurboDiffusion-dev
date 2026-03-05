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

"""Wan2.2 I2V SFT experiment with dual high/low noise models.

Usage:
    torchrun ... -m scripts.train --config=turbodiffusion/rcm/configs/registry_sft.py \\
        -- experiment=wan2pt2_14B_res720p_i2v_SFT

See wan2pt2_t2v.py for the T2V variant.
"""

from hydra.core.config_store import ConfigStore

from imaginaire.lazy_config import LazyDict


WAN2PT2_14B_RES720P_I2V_SFT: LazyDict = LazyDict(
    dict(
        defaults=[
            {"override /trainer": "standard"},
            {"override /data_train": "webdataset"},
            {"override /model": "fsdp_t2v_sft_wan22"},
            # net here sets model.config.net (low noise model architecture)
            {"override /net": "wan2pt2_14B_i2v"},
            {"override /conditioner": "text_nodrop"},
            {"override /ckpt_type": "dcp"},
            {"override /optimizer": "fusedadamw"},
            {
                "override /callbacks": [
                    "basic",
                    "dataloading_speed",
                    "wandb",
                ]
            },
            {"override /checkpoint": "local"},
            {"override /tokenizer": "wan2pt1_tokenizer"},
            "_self_",
        ],
        job=dict(
            group="SFT_Wan22",
            name="wan2pt2_14B_res720p_i2v_SFT",
        ),
        optimizer=dict(
            lr=1e-5,
            weight_decay=0.01,
            betas=(0.9, 0.999),
        ),
        model=dict(
            config=dict(
                # dual model settings
                boundary_ratio=0.9,
                # high noise model uses same architecture as low noise model (wan2pt2_14B_i2v)
                # net_high will be set to same config as net unless overridden at launch
                lora_enabled=True,
                lora_r=128,
                lora_alpha=128,
                lora_dropout=0.0,
                lora_target_modules=["q", "k", "v", "o"],
                fsdp_shard_size=8,
                resolution="720p",
                timestep_shift=5,
                state_t=21,
                grad_clip=False,
                guidance_scale=5.0,
                tokenizer=dict(vae_pth="assets/checkpoints/Wan2.1_VAE.pth"),
                text_encoder_path="assets/checkpoints/models_t5_umt5-xxl-enc-bf16.pth",
                neg_embed_path="assets/checkpoints/umT5_wan_negative_emb.pt",
                p_t=dict(
                    p_mean=0.0,
                    p_std=1.6,
                ),
                precision="bfloat16",
                net=dict(
                    sac_config=dict(
                        mode="block_wise",
                    ),
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
        ),
        model_parallel=dict(
            context_parallel_size=1,
        ),
        dataloader_train=dict(
            tar_path_pattern="assets/datasets/wan2pt2_i2v/shard*.tar",
            batch_size=1,
        ),
    ),
    flags={"allow_objects": True},
)

cs = ConfigStore.instance()
cs.store(
    group="experiment",
    package="_global_",
    name="wan2pt2_14B_res720p_i2v_SFT",
    node=WAN2PT2_14B_RES720P_I2V_SFT,
)
