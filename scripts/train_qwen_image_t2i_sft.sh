WORKDIR="/picassox/cephfs/optimization/yh7/Project/TurboDiffusion-dev"
cd $WORKDIR
export PYTHONPATH=turbodiffusion

# the "IMAGINAIRE_OUTPUT_ROOT" environment variable is the path to save experiment output files
export IMAGINAIRE_OUTPUT_ROOT=${WORKDIR}/outputs
MODEL_ROOT="/simple/Qwen/Qwen-Image"
DATASET_ROOT="/picassox/cephfs/optimization/yh7/Project/TurboDiffusion-dev/assets/qwen_image_cache"

# your Wandb information
export WANDB_API_KEY=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
export WANDB_ENTITY=yangh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

registry=registry_sft
# experiment=qwen_image_t2i_SFT
experiment=qwen_image_t2i_SFT_debug

torchrun --nproc_per_node=1 \
    -m scripts.train --config=turbodiffusion/rcm/configs/${registry}.py -- experiment=${experiment} \
        model.config.net.model_path=${MODEL_ROOT} \
        model.config.tokenizer.model_path=${MODEL_ROOT} \
        model.config.neg_embed_path=${DATASET_ROOT}/neg_embed.pt \
        model.config.ema.enabled=False \
        model.config.lora_enabled=True \
        model.config.lora_r=128 \
        model.config.lora_alpha=128 \
        model.config.fsdp_shard_size=8 \
        dataloader_train.tar_path_pattern="${DATASET_ROOT}/shard*.tar"
