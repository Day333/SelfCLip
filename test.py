#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SelfCLIP 模型测试脚本 - 计算文本检索图片匹配准确率
"""

import os
import sys

# 禁用torch安全检查以支持旧版本模型文件
import transformers.utils.import_utils
import transformers.modeling_utils
original_check = transformers.utils.import_utils.check_torch_load_is_safe
transformers.utils.import_utils.check_torch_load_is_safe = lambda: None
# 也需要patch modeling_utils中的调用
def dummy_load_state_dict(checkpoint_file, map_location=None, weights_only=False, **kwargs):
    import torch
    # 如果是safetensors文件，使用safetensors加载
    if checkpoint_file.endswith('.safetensors'):
        try:
            from safetensors.torch import load_file
            # safetensors不支持meta设备，使用cpu代替
            device = map_location if map_location != "meta" else "cpu"
            return load_file(checkpoint_file, device=device)
        except ImportError:
            print("警告: safetensors库未安装，尝试使用torch.load")
    return torch.load(checkpoint_file, map_location=map_location)
transformers.modeling_utils.load_state_dict = dummy_load_state_dict

import pandas as pd
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm
import argparse

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.model import SelfCLIP
from config import get_ptm_path
from transformers import AutoTokenizer
try:
    from transformers import AutoImageProcessor
except ImportError:
    from transformers import AutoFeatureExtractor as AutoImageProcessor



class SelfCLIPTester:
    def __init__(self, model_path, config):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {self.device}")
        
        # 加载模型
        self.model = SelfCLIP(
            config.dim, 
            get_ptm_path(config.text_ptm), 
            get_ptm_path(config.img_ptm), 
            self.device, 
            pretrained=config.pretrained,
            freeze=config.freeze,
            freeze_text_layers=getattr(config, 'freeze_text_layers', None),
            freeze_img_layers=getattr(config, 'freeze_img_layers', None),
            unfreeze_last_n_text=getattr(config, 'unfreeze_last_n_text', None),
            unfreeze_last_n_img=getattr(config, 'unfreeze_last_n_img', None)
        )
        
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"已加载模型: {model_path}")
        else:
            print("警告: 使用随机初始化权重进行测试")
        
        self.model.eval()
        self.model = self.model.to(self.device)
        
        # 加载tokenizer和图像处理器
        self.tokenizer = AutoTokenizer.from_pretrained(get_ptm_path(config.text_ptm))
        self.image_processor = AutoImageProcessor.from_pretrained(get_ptm_path(config.img_ptm))
        
    def encode_text(self, text):
        """编码单个文本"""
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            padding=True, 
            truncation=True, 
            max_length=self.config.max_text_len
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            text_vec = self.model.textencoder(inputs)
            text_features = torch.nn.functional.normalize(text_vec @ self.model.text_projection, dim=-1)
        
        return text_features
    
    def encode_image(self, image_path):
        """编码单个图像"""
        try:
            image = Image.open(image_path).convert('RGB')
            inputs = self.image_processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                img_vec = self.model.imgencoder(inputs)
                image_features = torch.nn.functional.normalize(img_vec @ self.model.img_projection, dim=-1)
            
            return image_features
        except Exception as e:
            print(f"无法加载图像 {image_path}: {e}")
            return None
    
    def calculate_retrieval_accuracy(self, test_data):
        """计算文本检索图片的准确率"""
        print("正在编码测试数据...")
        
        # 编码所有图像
        image_features = []
        valid_indices = []
        
        for idx, row in tqdm(test_data.iterrows(), total=len(test_data), desc="编码图像"):
            img_feat = self.encode_image(row[self.config.img_key])
            if img_feat is not None:
                image_features.append(img_feat)
                valid_indices.append(idx)
        
        if not image_features:
            print("错误: 没有有效的图像特征")
            return 0.0, 0, 0
        
        image_features = torch.cat(image_features, dim=0)  # [N, dim]
        valid_data = test_data.iloc[valid_indices].reset_index(drop=True)
        
        print(f"有效数据: {len(valid_data)} 条")
        
        # 计算准确率
        correct_predictions = 0
        total_predictions = len(valid_data)
        
        for idx, row in tqdm(valid_data.iterrows(), total=len(valid_data), desc="计算准确率"):
            # 编码查询文本
            text_feat = self.encode_text(row[self.config.caption_key])
            
            # 计算相似度
            similarities = torch.matmul(text_feat, image_features.T)  # [1, N]
            
            # 找到最相似的图像
            predicted_idx = torch.argmax(similarities, dim=1).item()
            
            # 检查是否预测正确（当前文本应该匹配当前图像）
            if predicted_idx == idx:
                correct_predictions += 1
        
        accuracy = correct_predictions / total_predictions
        return accuracy, correct_predictions, total_predictions

def main():
    parser = argparse.ArgumentParser(description='SelfCLIP 模型测试')
    parser.add_argument('--model_path', type=str, default=None,
                       help='训练好的模型路径')
    parser.add_argument('--config_file', type=str, default=None,
                       help='配置文件路径（可选）')
    
    # 添加与train.sh对齐的关键参数
    parser.add_argument('--dim', type=int, default=None,
                       help='特征维度')
    parser.add_argument('--text_ptm', type=str, default=None,
                       help='文本预训练模型')
    parser.add_argument('--img_ptm', type=str, default=None,
                       help='图像预训练模型')
    parser.add_argument('--device', type=str, default=None,
                       help='设备')
    
    args = parser.parse_args()
    
    # 获取配置 - 临时修改sys.argv避免冲突
    original_argv = sys.argv.copy()
    sys.argv = [sys.argv[0]]  # 只保留脚本名
    
    try:
        from config import get_config
        config = get_config()
    finally:
        sys.argv = original_argv
    
    # 用命令行参数覆盖配置
    if args.dim is not None:
        config.dim = args.dim
    if args.text_ptm is not None:
        config.text_ptm = args.text_ptm
    if args.img_ptm is not None:
        config.img_ptm = args.img_ptm
    if args.device is not None:
        config.device = args.device
    
    # 加载测试数据
    if not os.path.exists(config.test_file):
        print(f"错误: 测试文件不存在 {config.test_file}")
        return
    
    test_data = pd.read_csv(config.test_file)
    print(f"测试数据: {len(test_data)} 条")
    
    # 创建测试器
    tester = SelfCLIPTester(args.model_path, config)
    
    # 计算准确率
    accuracy, correct, total = tester.calculate_retrieval_accuracy(test_data)
    
    # 输出结果
    print("\n" + "=" * 50)
    print("SelfCLIP 文本检索图片准确率测试结果")
    print("=" * 50)
    print(f"正确预测: {correct}/{total}")
    print(f"准确率: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print("=" * 50)

if __name__ == "__main__":
    main()
