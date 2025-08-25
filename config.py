#!/usr/bin/env python
# coding: utf-8
import argparse
import os

def get_config():
    """
    Parse command line arguments and return configuration
    """
    parser = argparse.ArgumentParser(description='Simple CLIP Training Configuration')
    
    # Data parameters
    parser.add_argument('--train_file', type=str, default='./data/train.tsv', 
                       help='Path to training data file')
    parser.add_argument('--valid_file', type=str, default='./data/val.tsv', 
                       help='Path to validation data file')
    parser.add_argument('--test_file', type=str, default='./data/test.tsv',
                       help='Path to test data file')
    parser.add_argument('--img_key', type=str, default='filepath', 
                       help='Column name for image path in data file')
    parser.add_argument('--caption_key', type=str, default='title', 
                       help='Column name for text caption in data file')
    parser.add_argument('--max_text_len', type=int, default=34, 
                       help='Maximum text sequence length')
    parser.add_argument('--sample_size', type=int, default=0, 
                       help='Sample size for training (0 for all data)')
    
    # Model parameters
    parser.add_argument('--text_ptm', type=str, default='roberta-large', 
                       choices=['roberta', 'roberta-large'],
                       help='Text pretrained model type')
    parser.add_argument('--img_ptm', type=str, default='vit', 
                       choices=['resnet50', 'resnet152', 'vit', 'vit-large'],
                       help='Image pretrained model type')
    parser.add_argument('--dim', type=int, default=2048, 
                       help='Feature dimension for projection layers')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Load pretrained model weights')
    parser.add_argument('--freeze', action='store_true', default=False,
                       help='Freeze text encoder parameters')
    
    # 高级冻结控制参数
    parser.add_argument('--freeze_text_layers', type=str, default=None,
                       help='Text encoder freeze strategy: "all", "none", "0-8" (freeze layers 0-8), etc.')
    parser.add_argument('--freeze_img_layers', type=str, default=None,
                       help='Image encoder freeze strategy: "all", "none", "0-8" (freeze layers 0-8), etc.')
    parser.add_argument('--unfreeze_last_n_text', type=int, default=None,
                       help='Unfreeze last N layers of text encoder (overrides freeze_text_layers)')
    parser.add_argument('--unfreeze_last_n_img', type=int, default=None,
                       help='Unfreeze last N layers of image encoder (overrides freeze_img_layers)')
    
    parser.add_argument('--load_model', type=str, default=None, 
                       help='Path to load existing model checkpoint')
    
    # Training parameters
    parser.add_argument('--output_dir', type=str, default=None, 
                       help='Output directory for checkpoints')
    parser.add_argument('--device', type=str, default='cuda:0', 
                       help='Training device')
    parser.add_argument('--epochs', type=int, default=3, 
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4, 
                       help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=128, 
                       help='Training batch size')
    parser.add_argument('--accumulation_steps', type=int, default=8, 
                       help='Gradient accumulation steps')
    parser.add_argument('--eval_epoch', type=int, default=1, 
                       help='Evaluation frequency in epochs')
    parser.add_argument('--apex', action='store_true', default=True,
                       help='Use mixed precision training')
    parser.add_argument('--seed', type=int, default=42, 
                       help='Random seed for reproducibility')
    
    # Scheduler parameters
    parser.add_argument('--scheduler', type=str, default='cosine', 
                       choices=['linear', 'cosine'],
                       help='Learning rate scheduler type')
    parser.add_argument('--last_epoch', type=int, default=-1, 
                       help='Last epoch for scheduler')
    parser.add_argument('--batch_scheduler', action='store_true', default=True,
                       help='Update scheduler every batch')
    parser.add_argument('--weight_decay', type=float, default=0.01, 
                       help='Weight decay for optimizer')
    parser.add_argument('--num_warmup_steps', type=int, default=0, 
                       help='Number of warmup steps for scheduler')
    parser.add_argument('--num_cycles', type=float, default=0.5, 
                       help='Number of cycles for cosine scheduler')
    
    # Logging parameters
    parser.add_argument('--log_step', type=int, default=100, 
                       help='Logging frequency in steps')
    parser.add_argument('--wandb', action='store_true', default=False,
                       help='Use wandb for logging (disabled by default)')
    parser.add_argument('--key_metrics', type=str, default='image_to_text_R@10', 
                       help='Key metric for model selection')
    
    args = parser.parse_args()
    
    # Set default output directory based on model parameters
    if args.output_dir is None:
        args.output_dir = f'checkpoints/pretrain-{args.text_ptm}-{args.img_ptm}-saved'
    
    os.makedirs(args.output_dir, exist_ok=True)
    return args

# Pretrained model paths mapping
# Use local paths for downloaded models, HuggingFace names for online access
PTM_PATHS = {
    "roberta": './checkpoints/models/chinese-roberta-wwm-ext',
    "roberta-large": 'hfl/chinese-roberta-wwm-ext-large',
    "resnet50": 'microsoft/resnet-50',
    "resnet152": 'microsoft/resnet-152', 
    "vit": './checkpoints/models/vit-base-patch16-224',
    "vit-large": 'google/vit-large-patch16-224',
}

def get_ptm_path(ptm_name):
    """Get pretrained model path by name"""
    return PTM_PATHS.get(ptm_name, ptm_name)
