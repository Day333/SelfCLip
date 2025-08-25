# SelfCLIP 冻结策略指南

## 概述
新增了灵活的冻结策略，可以精确控制ViT和文本模型的哪些层参与训练，适合不同的微调场景。

## 可用参数

### 1. 分层冻结控制
- `--freeze_text_layers`: 控制文本编码器的冻结策略
- `--freeze_img_layers`: 控制图像编码器的冻结策略

**取值选项:**
- `"all"`: 冻结所有层
- `"none"`: 不冻结任何层
- `"0-8"`: 冻结第0到8层（例如）

### 2. 末层解冻控制
- `--unfreeze_last_n_text`: 解冻文本编码器最后N层
- `--unfreeze_last_n_img`: 解冻图像编码器最后N层

**注意**: 末层解冻会覆盖分层冻结设置

### 3. 传统冻结
- `--freeze`: 传统的全冻结模式（向后兼容）

## 使用示例

### 策略1: 只训练最后3层 (推荐用于快速微调)
```bash
python train.py \
    --unfreeze_last_n_text 3 \
    --unfreeze_last_n_img 3 \
    ... 其他参数
```

### 策略2: 冻结前8层，训练后面的层
```bash
python train.py \
    --freeze_text_layers 0-7 \
    --freeze_img_layers 0-7 \
    ... 其他参数
```

### 策略3: 完全冻结编码器，只训练投影层
```bash
python train.py \
    --freeze_text_layers all \
    --freeze_img_layers all \
    ... 其他参数
```

### 策略4: 全参数微调
```bash
python train.py \
    --freeze_text_layers none \
    --freeze_img_layers none \
    ... 其他参数
```

### 策略5: 混合策略
```bash
python train.py \
    --freeze_text_layers 0-9 \
    --unfreeze_last_n_img 2 \
    ... 其他参数
```

## 训练过程中的输出
训练开始时会打印冻结状态，例如：
```
文本编码器：解冻最后 3 层 (层 9-11)
图像编码器：解冻最后 3 层 (层 9-11)
```

## 建议的使用场景

1. **快速原型验证**: `--unfreeze_last_n_text 2 --unfreeze_last_n_img 2`
2. **数据量少**: `--freeze_text_layers all --freeze_img_layers all` (只训练投影层)
3. **数据量适中**: `--unfreeze_last_n_text 3 --unfreeze_last_n_img 3`
4. **数据量大**: `--freeze_text_layers none --freeze_img_layers none`
5. **领域适应**: `--freeze_text_layers 0-7 --freeze_img_layers 0-7`

## 注意事项

1. 冻结层数应根据具体的预训练模型调整（RoBERTa有12层，ViT-Base有12层）
2. 投影层（text_projection, img_projection）始终参与训练
3. 使用过多的解冻层可能导致过拟合
4. 建议先用较少的解冻层开始实验

## 模型层数参考

- **RoBERTa-Base**: 12个Transformer层 (0-11)
- **ViT-Base**: 12个Transformer层 (0-11)
- **RoBERTa-Large**: 24个Transformer层 (0-23)
- **ViT-Large**: 24个Transformer层 (0-23)
