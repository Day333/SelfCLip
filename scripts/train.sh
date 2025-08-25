#!/bin/bash

# Simple training script for CLIP model
set -e

echo "Starting SelfCLIP training..."

# Activate conda environment
source $(conda info --base)/etc/profile.d/conda.sh
conda activate llm

# Run training with basic parameters (using only 50 samples for quick testing)
python train.py \
    --train_file ./data/train.tsv \
    --valid_file ./data/val.tsv \
    --test_file ./data/test.tsv \
    --sample_size 1000 \
    --epochs 20 \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --dim 256 \
    --eval_epoch 1 \
    --log_step 5 \
    --text_ptm roberta \
    --img_ptm vit \
    --accumulation_steps 2 \
    --freeze_text_layers all \
    --freeze_img_layers all

# 其他可用的冻结策略（取消注释来使用）:
# 1. 冻结前8层: --freeze_text_layers 0-7 --freeze_img_layers 0-7
# 2. 全部冻结: --freeze_text_layers all --freeze_img_layers all  
# 3. 全部解冻: --freeze_text_layers none --freeze_img_layers none
# 4. 传统冻结: --freeze

echo "Training completed!"
