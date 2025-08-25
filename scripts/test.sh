#!/bin/bash

# SelfCLIP Testing Script
# 测试脚本 - 计算文本检索图片准确率

echo "开始 SelfCLIP 测试..."

# 设置参数
MODEL_PATH="checkpoints/pretrain-roberta-vit-saved/best_checkpoint.pt"  # 训练好的模型路径

# 检查模型文件是否存在
if [ ! -f "$MODEL_PATH" ]; then
    echo "警告: 模型文件不存在 $MODEL_PATH"
    echo "将使用随机初始化权重进行测试"
    MODEL_PATH=""
fi

# 运行测试 (参数与train.sh对齐)
TORCH_LOAD_SAFE_MODE=False python test.py \
    --model_path "$MODEL_PATH" \
    --dim 256 \
    --text_ptm roberta \
    --img_ptm vit \
    --device cuda:0

echo "测试完成！"
