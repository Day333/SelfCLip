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
    --sample_size 100 \
    --epochs 1 \
    --output_dir ./checkpoints/tmp \
    --batch_size 32 \
    --learning_rate 1e-4 \
    --dim 256 \
    --eval_epoch 1 \
    --log_step 5 \
    --text_ptm roberta \
    --img_ptm vit \
    --accumulation_steps 2

echo "Training completed!"
