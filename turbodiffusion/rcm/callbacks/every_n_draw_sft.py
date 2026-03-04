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

import os
from contextlib import nullcontext
from functools import partial
from typing import Optional

import torch
import torch.distributed as dist
import torchvision
import torchvision.transforms.functional as torchvision_F
from einops import rearrange
from megatron.core import parallel_state

import wandb
from imaginaire.callbacks.every_n import EveryN
from imaginaire.model import ImaginaireModel
from imaginaire.utils import distributed, log, misc
from imaginaire.utils.easy_io import easy_io
from imaginaire.utils.parallel_state_helper import is_tp_cp_pp_rank0
from imaginaire.utils.io import save_image_or_video


def resize_image(image: torch.Tensor, size: int = 1024) -> torch.Tensor:
    _, h, w = image.shape
    ratio = size / max(h, w)
    new_h, new_w = int(ratio * h), int(ratio * w)
    return torchvision_F.resize(image, (new_h, new_w))


def is_primitive(value):
    return isinstance(value, (int, float, str, bool, type(None)))


def convert_to_primitive(value):
    if isinstance(value, (list, tuple)):
        return [convert_to_primitive(v) for v in value if is_primitive(v) or isinstance(v, (list, dict))]
    elif isinstance(value, dict):
        return {k: convert_to_primitive(v) for k, v in value.items() if is_primitive(v) or isinstance(v, (list, dict))}
    elif is_primitive(value):
        return value
    else:
        return "non-primitive"


class EveryNDrawSample_SFT(EveryN):
    def __init__(
        self,
        every_n: int,
        step_size: int = 1,
        n_sample_to_save: int = 64,
        num_sampling_step: int = 50,
        is_sample: bool = True,
        save_s3: bool = False,
        is_ema: bool = False,
        show_all_frames: bool = False,
        is_image: bool = False,
        num_samples: int = 10,
        run_at_start: bool = False,
    ):
        super().__init__(every_n, step_size, run_at_start=run_at_start)
        self.n_sample_to_save = n_sample_to_save
        self.save_s3 = save_s3
        self.is_sample = is_sample
        self.name = self.__class__.__name__
        self.is_ema = is_ema
        self.show_all_frames = show_all_frames
        self.num_sampling_step = num_sampling_step
        self.rank = distributed.get_rank()
        self.is_image = is_image
        self.num_samples = num_samples

    def on_train_start(self, model: ImaginaireModel, iteration: int = 0) -> None:
        config_job = self.config.job
        self.local_dir = f"{config_job.path_local}/{self.name}"
        if distributed.get_rank() == 0:
            os.makedirs(self.local_dir, exist_ok=True)
            log.info(f"Callback: local_dir: {self.local_dir}")

        if parallel_state.is_initialized():
            self.data_parallel_id = parallel_state.get_data_parallel_rank()
        else:
            self.data_parallel_id = self.rank

    @torch.no_grad()
    def every_n_impl(self, trainer, model, data_batch, output_batch, loss, iteration):
        if self.is_ema:
            if not model.config.ema.enabled:
                return
            context = partial(model.ema_scope, "every_n_sampling")
        else:
            context = nullcontext

        tag = "ema" if self.is_ema else "reg"
        sample_counter = getattr(trainer, "sample_counter", iteration)
        batch_info = {
            "data": {k: convert_to_primitive(v) for k, v in data_batch.items() if is_primitive(v) or isinstance(v, (list, dict))},
            "sample_counter": sample_counter,
            "iteration": iteration,
        }
        if is_tp_cp_pp_rank0():
            if self.data_parallel_id < self.n_sample_to_save:
                easy_io.dump(
                    batch_info,
                    f"{self.local_dir}/BatchInfo_ReplicateID{self.data_parallel_id:04d}_Iter{iteration:09d}.json",
                )

        with context():
            if self.is_sample:
                sample_img_fp, MSE = self.sample(trainer, model, data_batch, output_batch, loss, iteration)

            dist.barrier()

        if wandb.run:
            data_type = "image" if model.is_image_batch(data_batch) else "video"
            tag += f"_{data_type}"
            info = {"trainer/global_step": iteration, "sample_counter": sample_counter}

            if self.is_sample:
                info[f"{self.name}/{tag}_sample"] = wandb.Image(sample_img_fp, caption=f"{sample_counter}")
                info[f"{self.name}/{tag}_MSE"] = MSE
            wandb.log(info, step=iteration)
        torch.cuda.empty_cache()

    @misc.timer("EveryNDrawSample: sample")
    def sample(self, trainer, model, data_batch, output_batch, loss, iteration):
        tag = "ema" if self.is_ema else "reg"
        raw_data, x0, _, _ = model.get_data_and_condition(data_batch)

        sample = model.generate_samples_from_batch(
            data_batch,
            state_shape=x0.shape[1:],
            n_sample=x0.shape[0],
            num_steps=self.num_sampling_step,
        )
        if hasattr(model, "decode"):
            sample = model.decode(sample)

        # MSE between generated sample and ground-truth video in pixel space
        MSE = torch.mean((sample.float() - raw_data.float()) ** 2)
        dist.all_reduce(MSE, op=dist.ReduceOp.AVG)

        # [sample, raw_data] side by side
        to_show = [sample.cpu(), raw_data.cpu()]

        base_fp_wo_ext = f"{tag}_ReplicateID{self.data_parallel_id:04d}_Sample_Iter{iteration:09d}"

        batch_size = x0.shape[0]
        if is_tp_cp_pp_rank0():
            local_path = self.run_save(to_show, batch_size, base_fp_wo_ext)
            return local_path, MSE.cpu().item()
        return None, None

    def run_save(self, to_show, batch_size, base_fp_wo_ext) -> Optional[str]:
        to_show = (1.0 + torch.stack(to_show, dim=0).clamp(-1, 1)) / 2.0  # [n, b, c, t, h, w]
        is_single_frame = to_show.shape[3] == 1
        n_viz_sample = batch_size

        if self.data_parallel_id < self.n_sample_to_save:
            save_image_or_video(rearrange(to_show, "n b c t h w -> c t (n h) (b w)"), f"{self.local_dir}/{base_fp_wo_ext}")

        file_base_fp = f"{base_fp_wo_ext}_resize.jpg"
        local_path = f"{self.local_dir}/{file_base_fp}"

        if self.rank == 0 and wandb.run:
            if is_single_frame:
                to_show = rearrange(to_show[:, :n_viz_sample], "n b c t h w -> t c (n h) (b w)")
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                torchvision.utils.save_image(resize_image(image_grid, 1024), local_path, nrow=1, scale_each=True)
            else:
                to_show = to_show[:, :n_viz_sample]  # [n, b, c, t, h, w]
                if not self.show_all_frames:
                    _T = to_show.shape[3]
                    three_frames_list = [0, _T // 2, _T - 1]
                    to_show = to_show[:, :, :, three_frames_list]
                    log_image_size = 1024
                else:
                    log_image_size = 512 * to_show.shape[3]
                to_show = rearrange(to_show, "n b c t h w -> 1 c (n h) (b t w)")
                image_grid = torchvision.utils.make_grid(to_show, nrow=1, padding=0, normalize=False)
                torchvision.utils.save_image(resize_image(image_grid, log_image_size), local_path, nrow=1, scale_each=True)

            return local_path
        return None
