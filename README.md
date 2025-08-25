# SelfCLIP

一个简洁的中文多模态预训练模型，基于CLIP架构实现文本-图像匹配，仅仅只是一个toy example。

## 快速开始

### 1. 环境要求

```bash
pip install -r requirements.txt
```

### 2. 数据集

数据集已按8:1:1自动划分：
- `data/train.tsv` - 训练集 (799条)
- `data/val.tsv` - 验证集 (98条)  
- `data/test.tsv` - 测试集 (98条)

### 3. 训练

```bash
bash scripts/train.sh
```

### 4. 测试

```bash
bash scripts/test.sh
```

## 模型结构

```
SelfCLIP
├── 文本编码器: 中文RoBERTa (hfl/chinese-roberta-wwm-ext)
├── 图像编码器: ViT (google/vit-base-patch16-224)
├── 投影层: 2048维特征空间
└── 对比学习损失
```

## 项目结构

```
SelfCLIP/
├── models/model.py          # 模型定义
├── config.py               # 配置管理
├── train.py                # 训练脚本
├── test.py                 # 测试脚本
├── scripts/
│   ├── simple_train.sh     # 训练脚本
│   └── test.sh             # 测试脚本
├── data/                   # 数据集
├── checkpoints/            # 模型检查点
│   └── models/             # 预训练模型
└── requirements.txt        # 依赖列表
```

## 评价指标

**文本检索图片准确率**: 给定一个文本描述，模型能正确检索到对应图片的概率。

计算方式: `准确预测数量 / 总测试数量`

## 使用示例

```python
from models.model import SelfCLIP
from config import get_ptm_path

# 初始化模型
model = SelfCLIP(
    dim=2048,
    text_ptm_name=get_ptm_path('roberta'),
    img_ptm_name=get_ptm_path('vit'),
    device='cuda:0',
    pretrained=True
)

# 训练或推理...
```

## 配置参数

主要参数可在 `config.py` 中修改：

- `--epochs`: 训练轮数 (默认: 3)
- `--batch_size`: 批次大小 (默认: 128)
- `--learning_rate`: 学习率 (默认: 1e-4)
- `--dim`: 特征维度 (默认: 2048)
- `--text_ptm`: 文本模型 (默认: roberta)
- `--img_ptm`: 图像模型 (默认: vit)

## License

MIT License
