WORKDIR="/picassox/cephfs/optimization/yh7/Project/TurboDiffusion-dev"
cd $WORKDIR
export PYTHONPATH=turbodiffusion

# the "IMAGINAIRE_OUTPUT_ROOT" environment variable is the path to save experiment output files
export IMAGINAIRE_OUTPUT_ROOT=${WORKDIR}/outputs

CHECKPOINT_ROOT="/simple/Wan-AI/Wan2.2-T2V-A14B/"
DATASET_ROOT="/picassox/cephfs/optimization/yh7/datasets/worstcoder/Wan_datasets/Wan2.1_14B_720p_16:9_Euler-step100_shift-5.0_cfg-5.0_seed-0_250K"

# your Wandb information
export WANDB_API_KEY=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
export WANDB_ENTITY=yangh
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

registry=registry_sft
experiment=wan2pt2_14B_res720p_t2v_SFT

torchrun --nproc_per_node=1 \
    -m scripts.train --config=turbodiffusion/rcm/configs/${registry}.py -- experiment=${experiment} \
        model.config.tokenizer.vae_pth=${CHECKPOINT_ROOT}/Wan2.1_VAE.pth \
        model.config.pretrained_ckpt_high=${CHECKPOINT_ROOT}/Wan2.2-T2V-14B-high.dcp \
        model.config.pretrained_ckpt_low=${CHECKPOINT_ROOT}/Wan2.2-T2V-14B-low.dcp \
        model.config.text_encoder_path=${CHECKPOINT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth \
        model.config.neg_embed_path=${CHECKPOINT_ROOT}/umT5_wan_negative_emb.pt \
        model.config.ema.enabled=False \
        model.config.lora_enabled=True \
        model.config.lora_r=128 \
        model.config.lora_alpha=128 \
        model.config.boundary_ratio=0.9 \
        model.config.fsdp_shard_size=8 \
        dataloader_train.tar_path_pattern=${DATASET_ROOT}/shard*.tar
