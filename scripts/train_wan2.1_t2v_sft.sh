WORKDIR="/picassox/cephfs/optimization/yh7/Project/TurboDiffusion-dev"
cd $WORKDIR
export PYTHONPATH=turbodiffusion

# the "IMAGINAIRE_OUTPUT_ROOT" environment variable is the path to save experiment output files
export IMAGINAIRE_OUTPUT_ROOT=${WORKDIR}/outputs
CHECKPOINT_ROOT="/picassox/cephfs/optimization/yh7/Models/worstcoder/Wan"
DATASET_ROOT="/picassox/cephfs/optimization/yh7/datasets/worstcoder/Wan_datasets/Wan2.1_14B_480p_16:9_Euler-step100_shift-3.0_cfg-5.0_seed-0_250K"

# your Wandb information
export WANDB_API_KEY=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
export WANDB_ENTITY=yangh
export CUDA_VISIBLE_DEVICES=2,3

registry=registry_sft
# experiment=wan2pt1_1pt3B_res480p_t2v_SFT
experiment=wan2pt1_1pt3B_res480p_t2v_SFT_debug

torchrun --nproc_per_node=2 \
    -m scripts.train --config=turbodiffusion/rcm/configs/${registry}.py -- experiment=${experiment} \
        model.config.tokenizer.vae_pth=${CHECKPOINT_ROOT}/Wan2.1_VAE.pth \
        model.config.pretrained_ckpt=${CHECKPOINT_ROOT}/Wan2.1-T2V-1.3B.dcp \
        model.config.text_encoder_path=${CHECKPOINT_ROOT}/models_t5_umt5-xxl-enc-bf16.pth \
        model.config.neg_embed_path=${CHECKPOINT_ROOT}/umT5_wan_negative_emb.pt \
        model.config.lora_enabled=True \
        model.config.lora_r=128 \
        model.config.lora_alpha=128 \
        model.config.fsdp_shard_size=2 \
        dataloader_train.tar_path_pattern=${DATASET_ROOT}/shard*.tar