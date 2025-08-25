#!/usr/bin/env python
# coding: utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoConfig, AutoModel, AutoModelForImageClassification, ViTModel, ViTConfig, AutoTokenizer

# #### TextEncoder

class TextEncoder(nn.Module):
    def __init__(self, ptm_name, device, pretrained, freeze=False, freeze_layers=None, unfreeze_last_n=None):
        super().__init__()
        if pretrained:
            self.model = AutoModel.from_pretrained(ptm_name)
        else:
            self.model = AutoModel.from_config(AutoConfig.from_pretrained(ptm_name))
        self.feat_dim = AutoConfig.from_pretrained(ptm_name).hidden_size

        self.freeze = freeze
        self.freeze_layers = freeze_layers
        self.unfreeze_last_n = unfreeze_last_n
        
        # 应用冻结策略
        self.apply_freeze_strategy()
    
    def apply_freeze_strategy(self):
        """应用冻结策略"""
        if self.freeze:
            # 传统的全冻结模式
            for param in self.model.parameters():
                param.requires_grad = False
            print("文本编码器：全部冻结")
            return
        
        if self.freeze_layers is not None:
            if self.freeze_layers == "all":
                for param in self.model.parameters():
                    param.requires_grad = False
                print("文本编码器：全部冻结")
            elif self.freeze_layers == "none":
                for param in self.model.parameters():
                    param.requires_grad = True
                print("文本编码器：全部解冻")
            elif isinstance(self.freeze_layers, str) and "-" in self.freeze_layers:
                # 解析层范围，如 "0-8" 表示冻结第0到8层
                start, end = map(int, self.freeze_layers.split("-"))
                self._freeze_layers_by_range(start, end)
        
        if self.unfreeze_last_n is not None:
            # 解冻最后N层
            self._unfreeze_last_n_layers(self.unfreeze_last_n)
    
    def _freeze_layers_by_range(self, start_layer, end_layer):
        """冻结指定范围的层"""
        # 冻结embeddings
        if start_layer == 0:
            for param in self.model.embeddings.parameters():
                param.requires_grad = False
        
        # 冻结encoder层
        for i in range(start_layer, min(end_layer + 1, len(self.model.encoder.layer))):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = False
        
        print(f"文本编码器：冻结层 {start_layer}-{end_layer}")
    
    def _unfreeze_last_n_layers(self, n):
        """解冻最后N层"""
        total_layers = len(self.model.encoder.layer)
        start_unfreeze = max(0, total_layers - n)
        
        # 先冻结所有层
        for param in self.model.parameters():
            param.requires_grad = False
        
        # 解冻最后N层
        for i in range(start_unfreeze, total_layers):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True
        
        # 解冻pooler和其他输出层
        if hasattr(self.model, 'pooler') and self.model.pooler is not None:
            for param in self.model.pooler.parameters():
                param.requires_grad = True
        
        print(f"文本编码器：解冻最后 {n} 层 (层 {start_unfreeze}-{total_layers-1})")
        
    def forward(self, inputs):
        if self.freeze:
            self.model.eval()
        last_hidden_state = self.model(**inputs).last_hidden_state # [batch_size, seq_len, hidden_size]
        # feature = torch.mean(last_hidden_state, axis=1) # [batch_size, hidden_size]
        feature = last_hidden_state[:, 0, :]  # 取序列的第一个 token
        # tok = AutoTokenizer.from_pretrained(self.model.name_or_path)
        # print("First token text:", tok.convert_ids_to_tokens(inputs["input_ids"][0][0].item()))
        # raise ValueError

        return feature

# #### ImageEncoder

class ImageEncoder(nn.Module):
    def __init__(self, ptm_name, pretrained, freeze=False, freeze_layers=None, unfreeze_last_n=None):
        super().__init__()
        if pretrained:
            # Use base model instead of classification model to get better features
            if 'vit' in ptm_name.lower():
                # ViT模型优先使用safetensors
                self.model = ViTModel.from_pretrained(ptm_name, use_safetensors=True)
                self.feat_dim = self.model.config.hidden_size
            else:
                # For ResNet or other models, use the feature extractor
                self.model = AutoModelForImageClassification.from_pretrained(ptm_name)
                # Get feature dimension from the classifier input
                self.feat_dim = self.model.classifier.in_features if hasattr(self.model, 'classifier') else 1000
        else:
            if 'vit' in ptm_name.lower():
                config = ViTConfig.from_pretrained(ptm_name)
                self.model = ViTModel(config)
                self.feat_dim = config.hidden_size
            else:
                self.model = AutoModelForImageClassification.from_config(AutoConfig.from_pretrained(ptm_name))
                self.feat_dim = self.model.classifier.in_features if hasattr(self.model, 'classifier') else 1000
        
        self.use_vit = 'vit' in ptm_name.lower()
        self.freeze = freeze
        self.freeze_layers = freeze_layers
        self.unfreeze_last_n = unfreeze_last_n
        
        # 应用冻结策略
        self.apply_freeze_strategy()
    
    def apply_freeze_strategy(self):
        """应用冻结策略"""
        if self.freeze:
            # 传统的全冻结模式
            for param in self.model.parameters():
                param.requires_grad = False
            print("图像编码器：全部冻结")
            return
        
        if self.freeze_layers is not None:
            if self.freeze_layers == "all":
                for param in self.model.parameters():
                    param.requires_grad = False
                print("图像编码器：全部冻结")
            elif self.freeze_layers == "none":
                for param in self.model.parameters():
                    param.requires_grad = True
                print("图像编码器：全部解冻")
            elif isinstance(self.freeze_layers, str) and "-" in self.freeze_layers:
                # 解析层范围，如 "0-8" 表示冻结第0到8层
                start, end = map(int, self.freeze_layers.split("-"))
                self._freeze_layers_by_range(start, end)
        
        if self.unfreeze_last_n is not None:
            # 解冻最后N层
            self._unfreeze_last_n_layers(self.unfreeze_last_n)
    
    def _freeze_layers_by_range(self, start_layer, end_layer):
        """冻结指定范围的层"""
        if self.use_vit:
            # 冻结patch embeddings
            if start_layer == 0:
                for param in self.model.embeddings.parameters():
                    param.requires_grad = False
            
            # 冻结encoder层
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                for i in range(start_layer, min(end_layer + 1, len(self.model.encoder.layer))):
                    for param in self.model.encoder.layer[i].parameters():
                        param.requires_grad = False
        else:
            # 对于ResNet等其他模型的处理
            print("警告：非ViT模型的分层冻结功能需要进一步实现")
        
        print(f"图像编码器：冻结层 {start_layer}-{end_layer}")
    
    def _unfreeze_last_n_layers(self, n):
        """解冻最后N层"""
        if self.use_vit:
            if hasattr(self.model, 'encoder') and hasattr(self.model.encoder, 'layer'):
                total_layers = len(self.model.encoder.layer)
                start_unfreeze = max(0, total_layers - n)
                
                # 先冻结所有层
                for param in self.model.parameters():
                    param.requires_grad = False
                
                # 解冻最后N层
                for i in range(start_unfreeze, total_layers):
                    for param in self.model.encoder.layer[i].parameters():
                        param.requires_grad = True
                
                # 解冻layernorm和其他输出层
                if hasattr(self.model, 'layernorm'):
                    for param in self.model.layernorm.parameters():
                        param.requires_grad = True
                
                print(f"图像编码器：解冻最后 {n} 层 (层 {start_unfreeze}-{total_layers-1})")
            else:
                print("警告：无法找到encoder.layer结构")
        else:
            print("警告：非ViT模型的分层解冻功能需要进一步实现")

    def forward(self, inputs):
        if self.use_vit:
            # For ViT models, use pooler_output or last_hidden_state[:, 0]
            outputs = self.model(**inputs, return_dict=True)
            # lhs = outputs.last_hidden_state
            
            # cfg = self.model.config
            # # 计算期望的 patch 数
            # img_h, img_w = (cfg.image_size, cfg.image_size) if isinstance(cfg.image_size, int) else cfg.image_size
            # patch_h, patch_w = (cfg.patch_size, cfg.patch_size) if isinstance(cfg.patch_size, int) else cfg.patch_size
            # num_patches = (img_h // patch_h) * (img_w // patch_w)

            # has_cls_param = hasattr(self.model, "embeddings") and hasattr(self.model.embeddings, "cls_token")
            # print("[ImageEncoder][DEBUG] has_cls_param:", has_cls_param)
            # print("[ImageEncoder][DEBUG] seq_len:", lhs.shape[1], " expected:", 1 + num_patches)
            # print("[ImageEncoder][DEBUG] CLS should be at index 0 -> cls_vec shape:", lhs[:, 0, :].shape)
            # print("[ImageEncoder][DEBUG] last token (patch) shape:", lhs[:, -1, :].shape)
            # raise ValueError
        
            feature = outputs.last_hidden_state[:, 0]
        else:
            # For ResNet models, extract features before the final classification layer
            outputs = self.model(**inputs, output_hidden_states=True)
            if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
                feature = outputs.hidden_states[-1]  # Last hidden layer
                # Global average pooling for CNN features
                if len(feature.shape) > 2:
                    feature = torch.mean(feature.view(feature.size(0), feature.size(1), -1), dim=-1)
            else:
                # Fallback to logits if no hidden states available
                feature = outputs.logits
        
        return feature



# #### SimpleCLIP

class SelfCLIP(nn.Module):
    def __init__(self, dim, text_ptm_name, img_ptm_name, device, pretrained, freeze=False,
                 freeze_text_layers=None, freeze_img_layers=None,
                 unfreeze_last_n_text=None, unfreeze_last_n_img=None):
        super().__init__()
        self.device = device
        self.textencoder = TextEncoder(
            text_ptm_name, device, pretrained=pretrained, freeze=freeze,
            freeze_layers=freeze_text_layers, unfreeze_last_n=unfreeze_last_n_text
        )
        self.imgencoder = ImageEncoder(
            img_ptm_name, pretrained=pretrained,
            freeze_layers=freeze_img_layers, unfreeze_last_n=unfreeze_last_n_img
        )

        self.text_projection = nn.Parameter(torch.empty(self.textencoder.feat_dim, dim))
        self.img_projection = nn.Parameter(torch.empty(self.imgencoder.feat_dim, dim))
        self.logit_scale = nn.Parameter(torch.ones([]))
        self.init_parameters()
    
    def init_parameters(self):
        nn.init.constant_(self.logit_scale, np.log(1 / 0.07))
        nn.init.normal_(self.text_projection, std=0.02)
        nn.init.normal_(self.img_projection, std=0.02)

    def loss(self, text_feat, img_feat, logit_scale):
        labels = torch.arange(text_feat.shape[0], device=self.device, dtype=torch.long)

        logits_per_image = logit_scale * img_feat @ text_feat.T   # [batch_size, batch_size]
        logits_per_text = logit_scale * text_feat @ img_feat.T   # [batch_size, batch_size]
        
        total_loss = (
            F.cross_entropy(logits_per_image, labels) +
            F.cross_entropy(logits_per_text, labels)
            ) / 2
        return total_loss

    def forward(self, text_inputs, img_inputs, outputLoss=False):
        # 编码 + 投影 + 归一化（投影后再 norm）
        text_vec = self.textencoder(text_inputs)
        img_vec  = self.imgencoder(img_inputs)
        text_feat = F.normalize(text_vec @ self.text_projection, dim=-1)
        img_feat  = F.normalize(img_vec  @ self.img_projection,  dim=-1)

        with torch.no_grad():
            self.logit_scale.clamp_(max=np.log(100.0))
        logit_scale = self.logit_scale.exp()

        if outputLoss:
            loss = self.loss(text_feat, img_feat, logit_scale)
            return loss, text_feat, img_feat, logit_scale
        else:
            return text_feat, img_feat, logit_scale
