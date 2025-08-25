#!/usr/bin/env python
# coding: utf-8
import os
import pandas as pd
import warnings
from pillow_avif import AvifImagePlugin  # noqa: F401


# Suppress common warnings
warnings.filterwarnings('ignore', category=FutureWarning, module='transformers')
warnings.filterwarnings('ignore', category=UserWarning, module='PIL')
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import numpy as np
import random
# import wandb  # Removed dependency
from pprint import pprint

from tqdm import tqdm
import torch
from models.model import SelfCLIP
from torch.optim import Adam, SGD, AdamW
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import transformers
from transformers import AutoTokenizer
try:
    from transformers import AutoImageProcessor
except ImportError:
    from transformers import AutoFeatureExtractor as AutoImageProcessor
print(f"transformers.__version__: {transformers.__version__}")
from transformers import get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from config import get_ptm_path

transformers.logging.set_verbosity_error()

#=======设置全局seed保证结果可复现====
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# ### 数据预处理

class TrainDataset(Dataset):
    def __init__(self, input_file, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(get_ptm_path(config.text_ptm))

        # 尝试自动分隔符
        try:
            data_df = pd.read_csv(input_file)
        except Exception:
            try:
                data_df = pd.read_csv(input_file, sep='\t')
            except Exception as e:
                raise ValueError(f"Could not parse data file {input_file}: {e}")

        self.img_paths = data_df[config.img_key].values
        self.texts = data_df[config.caption_key].values

        self.feature_extractor = AutoImageProcessor.from_pretrained(get_ptm_path(config.img_ptm))

        # 采样（如有需要）
        if hasattr(config, 'sample_size') and config.sample_size > 0 and len(self.texts) > config.sample_size:
            import random
            indices = random.sample(range(len(self.texts)), config.sample_size)
            self.texts = self.texts[indices]
            self.img_paths = self.img_paths[indices]
            print(f'sampled {config.sample_size} from {input_file}, total len={len(self.texts)}')
        else:
            print(f'load data from {input_file} len={len(self.texts)}')

        # 过滤不存在的图片
        valid_indices = []
        for i, img_path in enumerate(self.img_paths):
            if os.path.exists(img_path):
                valid_indices.append(i)
            else:
                print(f"Warning: Image not found: {img_path}")

        if len(valid_indices) < len(self.img_paths):
            self.texts = self.texts[valid_indices]
            self.img_paths = self.img_paths[valid_indices]
            print(f'filtered to {len(self.texts)} samples with valid images')

        # 可选：是否在遇到坏图时跳过（默认 True）
        self.skip_broken = getattr(config, "skip_broken", True)

    def __len__(self):
        return len(self.texts)

    # --- 关键：统一把图片变成 HxWx3 的 numpy.uint8 ---
    def _load_rgb_array(self, img_path: str) -> np.ndarray:
        with Image.open(img_path) as img:
            # 先交给 PIL 做一次模式规整
            if img.mode != "RGB":
                img = img.convert("RGB")

            arr = np.asarray(img)

            # 万一还是灰度（HxW），堆叠成三通道
            if arr.ndim == 2:
                arr = np.stack([arr, arr, arr], axis=-1)
            # 万一是 RGBA（HxWx4），丢弃 alpha
            elif arr.ndim == 3 and arr.shape[-1] == 4:
                arr = arr[..., :3]

            # 强制类型为 uint8，避免个别库返回 float
            if arr.dtype != np.uint8:
                arr = arr.astype(np.uint8)

            # 最终必须是 HxWx3
            if not (arr.ndim == 3 and arr.shape[-1] == 3):
                raise ValueError(f"Unexpected image shape for {img_path}: {arr.shape}")

            return arr

    def __getitem__(self, item):
        img_path = self.img_paths[item]
        text = str(self.texts[item])

        # 文本
        text_tensor = self.tokenizer(
            text,
            max_length=self.config.max_text_len,
            truncation=True,
            return_tensors='pt',
            padding="max_length",
        )

        # 图片（稳健处理）
        try:
            img_arr = self._load_rgb_array(img_path)
            img_tensor = self.feature_extractor(
                images=img_arr,
                return_tensors="pt"
            )
        except Exception as e:
            msg = f"[WARN] failed to load/process image {img_path}: {e}"
            if self.skip_broken:
                print(msg)
                # 跳过坏图：递归取下一个样本，避免整个 DataLoader 崩
                return self.__getitem__((item + 1) % len(self))
            else:
                # 不跳过：抛错让你定位
                raise RuntimeError(msg)

        # squeeze 成你原本使用的形状
        for k, v in text_tensor.items():
            text_tensor[k] = v.squeeze(0)
        for k, v in img_tensor.items():
            img_tensor[k] = v.squeeze(0)

        return {'text': text_tensor, 'img': img_tensor}

# ### 主程序
# #### evaluate

def get_metrics(image_features, text_features, logit_scale):
    metrics = {}
    logits_per_image = (logit_scale * image_features @ text_features.t()).detach().cpu()
    logits_per_text = logits_per_image.t().detach().cpu()

    logits = {"image_to_text": logits_per_image, "text_to_image": logits_per_text}
    ground_truth = torch.arange(len(text_features)).view(-1, 1)

    for name, logit in logits.items():
        ranking = torch.argsort(logit, descending=True)
        preds = torch.where(ranking == ground_truth)[1]
        preds = preds.detach().cpu().numpy()
        metrics[f"{name}_mean_rank"] = preds.mean() + 1
        metrics[f"{name}_median_rank"] = np.floor(np.median(preds)) + 1
        for k in [1, 5, 10]:
            metrics[f"{name}_R@{k}"] = np.mean(preds < k)

    return metrics

def evaluate(model, valid_dataloader, device, config):
    model.eval()
    all_text_feat = []
    all_img_feat = []
    tk0 = tqdm(enumerate(valid_dataloader),total=len(valid_dataloader), desc="[Dev]")
    total_loss = 0
    for step, batch in tk0:
        for k,v in batch['img'].items():
            batch['img'][k] = v.to(device)
        for k,v in batch['text'].items():
            batch['text'][k] = v.to(device)
        with torch.no_grad():
            device_type = 'cuda' if config.device.startswith('cuda') else 'cpu'
            with torch.amp.autocast(device_type, enabled=config.apex and config.device.startswith('cuda')):
                loss, text_feat, img_feat, logit_scale = model(batch['text'], 
                                                                batch['img'], 
                                                                outputLoss=True)
        total_loss += loss.item()
        all_text_feat.append(text_feat)
        all_img_feat.append(img_feat)
        
    metrics = get_metrics(image_features=torch.cat(all_img_feat),
                          text_features=torch.cat(all_text_feat),
                          logit_scale=logit_scale)
    metrics['eval_loss'] = total_loss / len(valid_dataloader)
    return metrics

# #### train loop

def train_eval(model, train_dataloader, valid_dataloader, config):
    assert config.device.startswith('cuda') or config.device == 'cpu', ValueError("Invalid device.")
    device = torch.device(config.device)
    best_score = 0
    total_step = 0
    model = model.to(device)
    if config.apex and config.device.startswith('cuda'):
        scaler = torch.amp.GradScaler('cuda', enabled=True)
    else:
        scaler = torch.amp.GradScaler('cpu', enabled=False)
    if not len(train_dataloader):
        raise EOFError("Empty train_dataloader.")

    # 过滤掉冻结的权重
    param_optimizer = [(n, p) for n, p in model.named_parameters() if p.requires_grad]

    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    # 设置权重decay
    optimizer_grouped_parameters = [
        {"params": [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], "weight_decay": config.weight_decay},
        {"params": [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], "weight_decay": 0.0}]
    optimizer = AdamW(optimizer_grouped_parameters, lr=config.learning_rate, weight_decay=config.weight_decay)
    
    num_train_steps = int(len(train_dataloader) * config.epochs / config.accumulation_steps)
    if config.scheduler=='cosine':
        scheduler = get_cosine_schedule_with_warmup(
                    optimizer, 
                    num_warmup_steps=config.num_warmup_steps, 
                    num_training_steps=num_train_steps, 
                    num_cycles=config.num_cycles, 
#                     last_epoch = ((config.last_epoch+1)/config.epochs)*num_train_steps
                )
    else:
        scheduler = get_linear_schedule_with_warmup(
                optimizer, num_warmup_steps=config.num_warmup_steps, num_training_steps=num_train_steps
            )
    
    for cur_epc in range(int(config.epochs)):
        
        training_loss = 0
        model.train()
        tk0 = tqdm(enumerate(train_dataloader),total=len(train_dataloader), desc="Epoch: {}".format(cur_epc))
        for step, batch in tk0:
            total_step += 1
            for k,v in batch['img'].items():
                batch['img'][k] = v.to(device)
            for k,v in batch['text'].items():
                batch['text'][k] = v.to(device)
            device_type = 'cuda' if config.device.startswith('cuda') else 'cpu'
            with torch.amp.autocast(device_type, enabled=config.apex and config.device.startswith('cuda')):
                loss, _, _, _ = model(batch['text'], batch['img'], outputLoss=True)
            scaler.scale(loss).backward()
            if (step+1) % config.accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                if config.batch_scheduler:
                    scheduler.step()
            training_loss += loss.item()
            tk0.set_postfix(Epoch=cur_epc, Loss=training_loss/(step+1))
            if config.wandb and (step + 1) % config.log_step == 0:
                print(f"Step {total_step}: train_loss={loss:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}, epoch={cur_epc}")
        if cur_epc % config.eval_epoch == 0:
            metrics = evaluate(model, valid_dataloader, device, config)
            print(f"eval metrics = ")
            pprint(metrics)
            if config.wandb:
                print(f"Metrics at step {total_step}: {metrics}")
            if metrics[config.key_metrics] >= best_score:
                best_score = metrics[config.key_metrics]
                # model_save_path = os.path.join(config.output_dir,f'epoch{cur_epc}.pt') # 保留所有checkpoint
                model_save_path = os.path.join(config.output_dir,f'best_checkpoint.pt') # 保留最优checkpoint
                torch.save(model.state_dict(), model_save_path)
                print(f'save at {model_save_path}')
    torch.cuda.empty_cache()          

# #### 训练过程

def main_train(config):
    """Main training function that accepts config object"""
    seed_everything(seed=config.seed)
    
    # 加载数据
    train_dataset = TrainDataset(config.train_file, config)
    valid_dataset = TrainDataset(config.valid_file, config)
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=5)
    valid_dataloader = DataLoader(valid_dataset, batch_size=config.batch_size, num_workers=5)
    
    # 加载模型
    device = torch.device(config.device)
    clipModel = SelfCLIP(config.dim, 
                          get_ptm_path(config.text_ptm), 
                          get_ptm_path(config.img_ptm), 
                          device, 
                          pretrained=config.pretrained,
                          freeze=config.freeze,
                          freeze_text_layers=config.freeze_text_layers,
                          freeze_img_layers=config.freeze_img_layers,
                          unfreeze_last_n_text=config.unfreeze_last_n_text,
                          unfreeze_last_n_img=config.unfreeze_last_n_img)
    
    if config.load_model is not None:
        clipModel.load_state_dict(torch.load(config.load_model))
        print(f"load state from {config.load_model}")
    
    if config.wandb:
        print(f"Training: {config.text_ptm}-{config.img_ptm}-batch{config.batch_size}-dim{config.dim}")
    
    # 训练
    train_eval(clipModel, train_dataloader, valid_dataloader, config)


if __name__ == '__main__':
    # This is for backward compatibility, but main entry should be through run.py
    print("Warning: Please use run.py as the main entry point for training.")
    print("Example: python run.py --help")
    
    from config import get_config
    config = get_config()
    main_train(config)