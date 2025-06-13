import os
import time
import copy
import numpy as np
from PIL import Image
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torch.optim as optim
# MARS 需要 Optimizer 基类
from torch.optim.optimizer import Optimizer
from tqdm import tqdm # 导入tqdm用于进度条
import argparse # 导入argparse用于命令行参数
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    auc,
    confusion_matrix,
    roc_auc_score, # 用于直接计算AUC
    precision_recall_curve, # 用于PR曲线
    average_precision_score # 用于计算PR曲线的AP
)
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # 用于3D绘图
import seaborn as sns
import umap # 导入umap

# --- MARS 优化器代码 (仅包含 MARS-AdamW) ---
# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
# 基于用户请求简化

def exists(val):
    return val is not None

# MARS-AdamW 更新函数 (核心逻辑) - 注意：此函数未在 MARS_AdamW 类中直接调用，类本身实现了更新逻辑
def update_fn_mars_adamw(p, grad, exp_avg, exp_avg_sq, lr, wd, beta1, beta2, last_grad, eps, amsgrad, max_exp_avg_sq, step, gamma):
    c_t = (grad - last_grad).mul(gamma * (beta1 / (1. - beta1))).add(grad)
    exp_avg.mul_(beta1).add_(c_t, alpha=1. - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1. - beta2)
    bias_correction1 = 1.0 - beta1 ** step
    bias_correction2 = 1.0 - beta2 ** step

    if amsgrad:
        torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
        denom = max_exp_avg_sq.sqrt().mul(1/math.sqrt(bias_correction2) if bias_correction2 > 0 else 1.0).add(eps).mul(bias_correction1) # 修正分母
    else:
        denom = exp_avg_sq.sqrt().mul(1/math_sqrt_bias_correction2(bias_correction2)).add(eps).mul(bias_correction1) # 修正分母

    update = exp_avg.div(denom)
    if wd != 0:
        p.data.mul_(1 - lr * wd) 

    p.data.add_(update, alpha=-lr) 
    return exp_avg, exp_avg_sq

# Helper function to avoid recalculating sqrt(bias_correction2) - 注意：此函数未在 MARS_AdamW 类中直接调用
def math_sqrt_bias_correction2(bias_correction2):
    if bias_correction2 == 0: return 0 
    return math.sqrt(bias_correction2)


# MARS 优化器类 (简化版 MARS-AdamW)
class MARS_AdamW(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.95, 0.99), eps=1e-8, weight_decay=1e-2, amsgrad=False, gamma=0.01):
        if not 0.0 <= lr: raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps: raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay: raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        if not 0.0 <= gamma: raise ValueError(f"Invalid gamma value: {gamma}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad, gamma=gamma)
        super(MARS_AdamW, self).__init__(params, defaults)
        # self.update_fn = update_fn_mars_adamw # 此行存在，但 step 方法中未使用该函数指针

    def __setstate__(self, state):
        super(MARS_AdamW, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if exists(closure):
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr, wd, beta1, beta2, amsgrad, eps_group, current_gamma = group['lr'], group['weight_decay'], group['betas'][0], group['betas'][1], group['amsgrad'], group['eps'], group['gamma']

            for p in filter(lambda p: exists(p.grad), group['params']):
                if p.grad is None: continue
                grad = p.grad.data
                if grad.is_sparse: raise RuntimeError('MARS-AdamW 不支持稀疏梯度')

                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    state['last_grad'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)
                    if amsgrad: state['max_exp_avg_sq'] = torch.zeros_like(p.data, memory_format=torch.preserve_format)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                last_grad = state['last_grad']
                max_exp_avg_sq = state['max_exp_avg_sq'] if amsgrad else None

                state['step'] += 1
                step = state['step']
                
                if wd != 0:
                    p.data.mul_(1 - lr * wd)

                c_t = (grad - last_grad).mul(current_gamma * (beta1 / (1. - beta1))).add(grad)

                exp_avg.mul_(beta1).add_(c_t, alpha=1. - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(c_t, c_t, value=1. - beta2)
                
                bias_correction1 = 1.0 - beta1 ** step
                bias_correction2 = 1.0 - beta2 ** step

                if amsgrad:
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2 if bias_correction2 > 0 else 1.0)).add_(eps_group)
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2 if bias_correction2 > 0 else 1.0)).add_(eps_group)

                step_size = lr / bias_correction1
                p.data.addcdiv_(exp_avg, denom, value=-step_size)
                
                state['last_grad'].copy_(grad)
        return loss

# --- MARS 优化器代码结束 ---

# 确保结果目录存在
os.makedirs('results', exist_ok=True)

# -----------------------------
# Focal Loss 实现
# -----------------------------
class FocalLoss(nn.Module):
    """Focal Loss 用于处理类别不平衡"""
    def __init__(self, alpha=0.35, gamma=1.5, reduction='mean', pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none', pos_weight=pos_weight)

    def forward(self, inputs, targets):
        bce_loss = self.bce_loss(inputs, targets)
        probas = torch.sigmoid(inputs)
        pt = torch.where(targets == 1, probas, 1 - probas)
        alpha_t = torch.where(targets == 1, self.alpha, 1 - self.alpha)
        focal_loss = alpha_t * (1 - pt)**self.gamma * bce_loss
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

# -----------------------------
# 数据集定义
# -----------------------------
class ChestXrayDataset(Dataset):
    """胸部X光数据集类"""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.true_dir = os.path.join(root_dir, 'true')
        self.false_dir = os.path.join(root_dir, 'false')
        if not os.path.isdir(self.true_dir): raise FileNotFoundError(f"目录未找到: {self.true_dir}")
        if not os.path.isdir(self.false_dir): raise FileNotFoundError(f"目录未找到: {self.false_dir}")
        self.true_images = [os.path.join(self.true_dir, f) for f in os.listdir(self.true_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.false_images = [os.path.join(self.false_dir, f) for f in os.listdir(self.false_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        self.all_images = self.true_images + self.false_images
        self.labels = [1] * len(self.true_images) + [0] * len(self.false_images)
        if not self.all_images: print(f"警告: 在 {root_dir} 中未找到图像")

    def __len__(self):
        return len(self.all_images)

    def __getitem__(self, idx):
        img_path = self.all_images[idx]
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
             print(f"打开图像错误 {img_path}: {e}")
             # 返回一个红色占位符图像和默认标签0，以避免训练中断
             image = Image.new('RGB', (224, 224), color = 'red') 
             label = 0 
        else: 
            label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

# -----------------------------
# 经典 CBAM 模块 (供对比)
# -----------------------------
class ClassicCBAM(nn.Module):
    """经典的 CBAM 注意力模块"""
    def __init__(self, channel, reduction=16):
        super(ClassicCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(), 
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        self.sigmoid_channel = nn.Sigmoid()
        self.conv_spatial = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid_channel(avg_out + max_out)
        x_out = x * channel_att 
        avg_pool_s = torch.mean(x_out, dim=1, keepdim=True)
        max_pool_s, _ = torch.max(x_out, dim=1, keepdim=True)
        spatial_cat = torch.cat([avg_pool_s, max_pool_s], dim=1)
        spatial_att = self.sigmoid_spatial(self.conv_spatial(spatial_cat))
        return x_out * spatial_att

# -----------------------------
# 改进的 CBAM 模块
# -----------------------------
class EnhancedCBAM(nn.Module):
    """增强型 CBAM：通道注意力使用 SiLU 和 Tanh(ax), 空间注意力融合 Scharr 和膨胀卷积"""
    def __init__(self, channel, reduction=16):
        super(EnhancedCBAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.SiLU(), 
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
        )
        self.tanh_a = nn.Parameter(torch.tensor(0.95)) 
        self.spatial_channel_combiner = nn.Conv2d(2, 1, kernel_size=1, padding=0, bias=False)
        scharr_x_kernel = torch.tensor([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]], dtype=torch.float32).view(1,1,3,3)
        scharr_y_kernel = torch.tensor([[-3, -10, -3], [0, 0, 0], [3, 10, 3]], dtype=torch.float32).view(1,1,3,3)
        self.scharr_conv_x = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.scharr_conv_y = nn.Conv2d(1, 1, 3, padding=1, bias=False)
        self.scharr_conv_x.weight = nn.Parameter(scharr_x_kernel, requires_grad=False)
        self.scharr_conv_y.weight = nn.Parameter(scharr_y_kernel, requires_grad=False)
        self.dilated_conv = nn.Conv2d(1, 1, kernel_size=3, padding=2, dilation=2, bias=False)
        self.weight_dil = nn.Parameter(torch.tensor(0.7))
        self.weight_scharr = nn.Parameter(torch.tensor(0.3)) 
        self.sigmoid_spatial = nn.Sigmoid()

    def forward(self, x):
        b, c, h, w = x.size()
        y_gap = self.avg_pool(x)
        y_gmp = self.max_pool(x)
        y = self.fc(y_gap + y_gmp)
        channel_att = 1.0 + torch.tanh(self.tanh_a * y) 
        x_out = x * channel_att
        avg_pool_s = torch.mean(x_out, dim=1, keepdim=True)
        max_pool_s, _ = torch.max(x_out, dim=1, keepdim=True)
        spatial_pooled_concat = torch.cat([avg_pool_s, max_pool_s], dim=1)
        single_channel_feature_map = self.spatial_channel_combiner(spatial_pooled_concat)
        grad_x = self.scharr_conv_x(single_channel_feature_map)
        grad_y = self.scharr_conv_y(single_channel_feature_map)
        grad = torch.sqrt(grad_x**2 + grad_y**2 + 1e-12) 
        dil = self.dilated_conv(single_channel_feature_map)
        w_dil = torch.sigmoid(self.weight_dil)
        w_scharr = torch.sigmoid(self.weight_scharr) 
        spatial_features_fused = w_dil * dil + w_scharr * grad
        spatial_attn = self.sigmoid_spatial(spatial_features_fused)
        return x_out * spatial_attn

# -----------------------------
# 辅助模块
# -----------------------------
class MultiScaleDSConv(nn.Module):
    """多尺度深度可分离卷积"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.depthwise3 = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.depthwise5 = nn.Conv2d(in_channels, in_channels, kernel_size=5, stride=stride, padding=2, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels*2, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x1 = self.depthwise3(x)
        x2 = self.depthwise5(x)
        x_cat = torch.cat([x1, x2], dim=1) 
        x_out = self.pointwise(x_cat)
        x_out = self.bn(x_out)
        return x_out

class RMSNorm(nn.Module):
    """均方根层归一化"""
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.scale = dim ** -0.5 
        self.eps = eps 
        self.g = nn.Parameter(torch.ones(dim)) 

    def forward(self, x):
        norm = torch.norm(x, dim=-1, keepdim=True) * self.scale
        return x / norm.clamp(min=self.eps) * self.g

class GeGLU(nn.Module):
    """门控 GELU 线性单元"""
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2) 

    def forward(self, x):
        x = self.proj(x)
        x, gate = x.chunk(2, dim=-1) 
        return x * F.gelu(gate)

# -----------------------------
# 残差块
# -----------------------------
class ResidualBlock(nn.Module):
    """包含 MultiScaleDSConv 和 CBAM 的残差块"""
    def __init__(self, in_ch, out_ch, stride, use_enhanced_cbam=True):
        super().__init__()
        self.use_enhanced_cbam = use_enhanced_cbam
        cbam_layer = EnhancedCBAM(out_ch) if use_enhanced_cbam else ClassicCBAM(out_ch)
        self.conv_block = nn.Sequential(
            MultiScaleDSConv(in_ch, out_ch, stride),
            cbam_layer,
        )
        self.use_shortcut = (stride != 1 or in_ch != out_ch)
        if self.use_shortcut:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        else:
            self.shortcut = nn.Identity()
        self.final_act = nn.SiLU() 

    def forward(self, x):
        identity = self.shortcut(x) 
        out = self.conv_block(x)   
        out = identity + out      
        out = self.final_act(out) 
        return out

# -----------------------------
# 主模型定义
# -----------------------------
class EnhancedCNN(nn.Module):
    """增强型 CNN 模型"""
    def __init__(self, num_classes=1, use_enhanced_cbam=True):
        super(EnhancedCNN, self).__init__()
        self.use_enhanced_cbam = use_enhanced_cbam
        print(f"模型初始化: 使用 {'增强型 (Enhanced)' if use_enhanced_cbam else '经典 (Classic)'} CBAM")
        self.stem = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU()
        )
        self.stage1 = self._make_stage(32, 64, stride=1)
        self.stage2 = self._make_stage(64, 128, stride=2)
        self.stage3 = self._make_stage(128, 256, stride=2)
        self.stage4 = self._make_stage(256, 512, stride=2)
        self.stage5 = self._make_stage(512, 1024, stride=2) 
        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.head_flatten = nn.Flatten()
        self.classifier_geglu1 = GeGLU(1024, 512) 
        self.classifier_norm = RMSNorm(512)
        self.classifier_fc1 = nn.Linear(512, 128) # 调整维度以匹配 GeGLU 输出
        self.classifier_dropout = nn.Dropout(0.5)
        self.classifier_geglu2 = GeGLU(128, 64)
        self.classifier_norm2 = RMSNorm(64)
        self.classifier_fc2 = nn.Linear(64, num_classes)

    def _make_stage(self, in_ch, out_ch, stride):
        return ResidualBlock(in_ch, out_ch, stride, self.use_enhanced_cbam)

    def forward(self, x, return_features=False):
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x) 
        x_pool = self.global_max_pool(x) 
        features = self.head_flatten(x_pool) 
        if return_features:
             return features 
        x = self.classifier_geglu1(features) 
        x = self.classifier_norm(x)          
        x = self.classifier_fc1(x)           
        x = self.classifier_dropout(x)       
        x = self.classifier_geglu2(x)        
        x = self.classifier_norm2(x)
        logits = self.classifier_fc2(x)      
        return logits

# -----------------------------
# Grad-CAM 实现
# -----------------------------
class GradCAM:
    """Grad-CAM 可视化类"""
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.forward_handle = None # 初始化 handle 属性
        self.backward_handle = None # 初始化 handle 属性
        # _register_hooks 应该在实际使用时调用，或者由 __call__ 管理

    def _register_hooks(self):
        # 确保旧的钩子被移除 (如果存在)
        if self.forward_handle: self.forward_handle.remove()
        if self.backward_handle: self.backward_handle.remove()
        
        self.forward_handle = self.target_layer.register_forward_hook(self._forward_hook)
        self.backward_handle = self.target_layer.register_full_backward_hook(self._backward_hook) 

    def _forward_hook(self, module, inp, out):
        self.activations = out.detach() 

    def _backward_hook(self, module, grad_in, grad_out):
        # grad_out 是一个元组，通常我们关心第一个元素
        if grad_out[0] is not None:
            self.gradients = grad_out[0].detach()
        else: # 有时 grad_out[0] 可能为 None，例如当层没有参数或者不需要梯度时
            print("Warning: backward_hook received None gradient.")
            self.gradients = None


    def __call__(self, x, class_idx=None, retain_graph=False):
        self.model.eval() # 确保模型在评估模式
        self._register_hooks() # 在每次调用时注册钩子
        
        logits = self.model(x) 
        
        if class_idx is None: # 默认目标是第一个输出（对于二分类的单一logit，即正类的logit）
            class_idx = 0 
        
        score = logits[:, class_idx] 
        
        self.model.zero_grad() 
        score.backward(torch.ones_like(score), retain_graph=retain_graph)
        
        if self.gradients is None or self.activations is None:
             raise RuntimeError("未能捕获梯度或激活。请检查钩子注册和目标层。")
        
        # 全局平均池化梯度
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3]) # [B, C]
        # 调整维度以匹配激活图
        pooled_gradients = pooled_gradients.unsqueeze(-1).unsqueeze(-1) # [B, C, 1, 1]
        
        # 激活图加权
        weighted_activations = self.activations * pooled_gradients # [B, C, H, W]
        heatmap = torch.mean(weighted_activations, dim=1) # [B, H, W]
        
        heatmap = F.relu(heatmap) # 应用 ReLU 截断负值
        
        # 逐个图像归一化热力图
        batch_size = heatmap.size(0)
        normalized_heatmap_batch = torch.zeros_like(heatmap)
        for i in range(batch_size):
            img_heatmap = heatmap[i]
            max_val = torch.max(img_heatmap)
            if max_val > 0:
                normalized_heatmap_batch[i] = img_heatmap / max_val
        
        # 通常 GradCAM 可视化单张图，如果输入 x 是 batch=1，则直接返回
        return normalized_heatmap_batch[0].cpu().numpy() # 返回第一张图的热力图

    def remove_hooks(self):
        if self.forward_handle:
            self.forward_handle.remove()
            self.forward_handle = None
        if self.backward_handle:
            self.backward_handle.remove()
            self.backward_handle = None
        self.gradients = None
        self.activations = None

# -----------------------------
# 训练、评估和可视化函数
# -----------------------------
def train_model(model, dataloaders, optimizer, criterion, scheduler, num_epochs=50, device='cuda', run_name="default_run"):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_test_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [], 'test_loss': [], 'test_acc': [], 'lr': []
    }
    current_device = torch.device(device) if isinstance(device, str) else device
    print(f"[{run_name}] 开始训练, 设备: {current_device}")
    use_amp = current_device.type == 'cuda'
    scaler = GradScaler() if use_amp else None
    if use_amp: print(f"[{run_name}] 已启用 CUDA Autocast 混合精度训练。")

    for epoch in range(num_epochs):
        print(f"\n[{run_name}] Epoch {epoch}/{num_epochs-1}")
        print('-' * 15)
        for phase in ['train', 'test']:
            model.train() if phase == 'train' else model.eval()
            running_loss, running_corrects, num_samples = 0.0, 0, 0
            data_iterator = tqdm(dataloaders[phase], desc=f"[{run_name}] {phase.capitalize()}")
            for inputs, labels in data_iterator:
                inputs, labels = inputs.to(current_device), labels.to(current_device).float().unsqueeze(1)
                optimizer.zero_grad(set_to_none=True)
                autocast_enabled = use_amp and phase == 'train'
                with autocast(enabled=autocast_enabled):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    # 在 autocast 上下文内或之外计算 preds 均可，但为保持一致性，可在 autocast 外进行
                    # 如果 outputs 是 float16, sigmoid 后也是 float16, >0.5 比较后再转 float
                
                # 在 autocast 区域外进行 preds 计算，确保使用 float32 进行比较
                if autocast_enabled:
                    preds = (torch.sigmoid(outputs.float()) > 0.5).float()
                else:
                    preds = (torch.sigmoid(outputs) > 0.5).float()

                if phase == 'train':
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                batch_size = inputs.size(0)
                running_loss += loss.item() * batch_size
                running_corrects += torch.sum(preds == labels.data)
                num_samples += batch_size
                current_loss, current_acc = running_loss / num_samples, running_corrects.double() / num_samples
                data_iterator.set_postfix(loss=f"{current_loss:.4f}", acc=f"{current_acc:.4f}")
            
            epoch_loss, epoch_acc = running_loss / num_samples, running_corrects.double() / num_samples
            if phase == 'train':
                history['train_loss'].append(epoch_loss)
                history['train_acc'].append(epoch_acc.cpu().item())
                if scheduler:
                    lr_to_log = scheduler.get_last_lr()[0] if hasattr(scheduler, 'get_last_lr') else optimizer.param_groups[0]['lr']
                    history['lr'].append(lr_to_log)
                    scheduler.step() # Scheduler step after epoch
                else: 
                    history['lr'].append(optimizer.param_groups[0]['lr'])

            else: # phase == 'test'
                history['test_loss'].append(epoch_loss)
                history['test_acc'].append(epoch_acc.cpu().item())
                if epoch_acc > best_test_acc:
                    best_test_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    print(f" -> [{run_name}] 新的最佳测试准确率: {best_test_acc:.4f} (Epoch {epoch})")
        
        print(f"  训练损失: {history['train_loss'][-1]:.4f}, 训练准确率: {history['train_acc'][-1]:.4f}")
        print(f"  测试损失: {history['test_loss'][-1]:.4f}, 测试准确率: {history['test_acc'][-1]:.4f}")
        if history['lr'] and len(history['lr']) > epoch: # 确保 history['lr'] 已被填充
             print(f"  当前学习率: {history['lr'][epoch]:.6e}")
        elif history['lr']: # Fallback if scheduler logic has issues with length
             fallback_lr = history['lr'][-1]
             print(f"  当前学习率 (fallback): {fallback_lr:.6e}")
             if len(history['lr']) < epoch + 1: history['lr'].extend([fallback_lr] * (epoch + 1 - len(history['lr'])))


    time_elapsed = time.time() - since
    print(f'\n[{run_name}] 训练完成, 耗时 {time_elapsed // 60:.0f} 分 {time_elapsed % 60:.0f} 秒')
    print(f'[{run_name}] 最佳测试准确率: {best_test_acc:4f}')
    model.load_state_dict(best_model_wts)
    return model, history, best_test_acc

def evaluate_model(model, dataloader, device='cuda', run_name="default_run"):
    model.eval() 
    all_labels, all_probs, all_preds = [], [], []
    print(f"[{run_name}] 正在评估模型...")
    with torch.no_grad(): 
        for inputs, labels in tqdm(dataloader, desc=f"[{run_name}] 评估中"):
            inputs, labels_cpu = inputs.to(device), labels # labels保持在CPU上，因为extend需要CPU数据
            outputs = model(inputs)
            # squeeze() 以处理 batch_size=1 时 probs 维度问题
            probs_batch = torch.sigmoid(outputs).squeeze().cpu().numpy() 
            # 如果batch_size=1且squeeze后是0维, 转为1维数组
            if probs_batch.ndim == 0: probs_batch = np.array([probs_batch])
            
            preds_batch = (probs_batch > 0.5).astype(int)
            
            all_labels.extend(labels_cpu.numpy())
            all_probs.extend(probs_batch)
            all_preds.extend(preds_batch)
            
    return np.array(all_labels), np.array(all_probs), np.array(all_preds)

def save_plot(base_filename, run_name="default_run", is_3d=False):
    if not is_3d and plt.get_current_fig_manager() is None :
        # 对于UMAP 3D图，figure是显式创建的，这里主要针对2D图
        print(f"[{run_name}] 警告: save_plot 调用时没有活动的 Matplotlib 图形 ({base_filename})。")
    
    png_filename = os.path.join('results', f"{run_name}_{base_filename}.png")
    svg_filename = os.path.join('results', f"{run_name}_{base_filename}.svg")
    
    try:
        plt.savefig(svg_filename, format='svg', bbox_inches='tight')
        plt.savefig(png_filename, dpi=800, bbox_inches='tight') 
        print(f"[{run_name}] 绘图已保存: {png_filename} 和 {svg_filename}")
    except Exception as e:
        print(f"[{run_name}] 保存绘图失败 ({base_filename}): {e}")
    finally:
        plt.close() 

def print_and_save_evaluation_report(labels, preds, probs, run_name="default_run"):
    report_filename = os.path.join('results', f"{run_name}_evaluation_report.txt")
    prec = precision_score(labels, preds, zero_division=0)
    rec = recall_score(labels, preds, zero_division=0)
    f1 = f1_score(labels, preds, zero_division=0)
    try:
        roc_auc_val = roc_auc_score(labels, probs)
    except ValueError as e:
        print(f"[{run_name}] 警告: 无法计算 ROC AUC 分数. 错误: {e}. 可能所有样本都属于一个类别或概率值恒定。")
        roc_auc_val = float('nan')
    report_lines = [
        f"--- [{run_name}] 评估报告 ---",
        f"Precision: {prec:.4f}",
        f"Recall: {rec:.4f}",
        f"F1-score: {f1:.4f}",
        f"ROC AUC: {roc_auc_val:.4f}",
        "-------------------------"
    ]
    for line in report_lines: print(line)
    with open(report_filename, 'w') as f:
        for line in report_lines: f.write(line + '\n')
    print(f"[{run_name}] 评估报告已保存到: {report_filename}")
    return roc_auc_val

def plot_metrics(history, run_name="default_run"):
    if not history or not history.get('train_acc'): # 检查 history 是否有效
        print(f"[{run_name}] 训练历史记录为空或不完整，跳过绘图。")
        return
        
    epochs = range(len(history['train_acc'])) 

    plt.figure(figsize=(10, 6)) 
    plt.plot(epochs, history['train_loss'], label='Training Loss', linestyle='-', alpha=0.9, color='blue')
    plt.plot(epochs, history['test_loss'], label='Test Loss', linestyle='-', linewidth=1.5, color='red')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('loss_curve', run_name)

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, history['train_acc'], label='Training Accuracy', linestyle='-', color='green')
    plt.plot(epochs, history['test_acc'], label='Test Accuracy', linestyle='-', linewidth=1.5, color='orange')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('accuracy_curve', run_name)

    plt.figure(figsize=(10, 6))
    if history.get('lr') and len(history['lr']) > 0:
        epochs_lr = range(len(history['lr'])) # lr 历史长度可能与 acc/loss 不同
        plt.plot(epochs_lr, history['lr'], label='Learning Rate', color='purple', alpha=0.7)
        plt.title('Learning Rate Schedule')
        plt.xlabel('Epoch')
        plt.ylabel('Learning Rate')
        plt.yscale('log') 
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
    else:
        plt.title('Learning Rate (No History)')
        plt.text(0.5, 0.5, 'Learning rate history not recorded or empty', horizontalalignment='center', verticalalignment='center')
    save_plot('learning_rate_curve', run_name)

def plot_confusion_matrix(labels, preds, run_name="default_run"):
    cm = confusion_matrix(labels, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted Normal','Predicted Enlarged'],
                yticklabels=['True Normal','True Enlarged'],
                annot_kws={"size": 14}) 
    plt.title('Confusion Matrix', fontsize=16)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.xticks(rotation=0) 
    plt.yticks(rotation=0) 
    save_plot('confusion_matrix', run_name)

def plot_roc_curve(labels, probs, run_name="default_run"):
    plt.figure(figsize=(8, 7))
    try:
        if len(np.unique(labels)) < 2: # 检查标签是否有多个类别
            raise ValueError("ROC AUC score is not defined in cases where only one class is present in y_true.")
        fpr, tpr, thresholds = roc_curve(labels, probs) 
        roc_auc_val = auc(fpr, tpr) 
        plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC curve (AUC = {roc_auc_val:.4f})')
    except ValueError as e:
        print(f"[{run_name}] 警告: 无法计算 ROC 曲线数据. 错误: {e}")
        roc_auc_val = float('nan') 
        plt.text(0.5, 0.5, 'ROC curve calculation failed\n(e.g. single class in labels)', 
                 horizontalalignment='center', verticalalignment='center', color='red')

    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Guess (AUC = 0.50)')
    plt.xlim([-0.02, 1.0]) 
    plt.ylim([-0.02, 1.05]) 
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=14)
    plt.legend(loc="lower right", fontsize=11) 
    plt.grid(True, linestyle='--', alpha=0.6) 
    save_plot('roc_curve', run_name)
    return roc_auc_val 

def plot_pr_curve(labels, probs, run_name="default_run"): # 新增PR曲线函数
    plt.figure(figsize=(8, 7))
    if not isinstance(labels, np.ndarray): labels = np.array(labels)
    if not isinstance(probs, np.ndarray): probs = np.array(probs)

    ap = float('nan')
    try:
        if len(np.unique(labels)) < 2:
            raise ValueError("Average precision score is not defined in cases where only one class is present in y_true.")
        precision, recall, _ = precision_recall_curve(labels, probs)
        ap = average_precision_score(labels, probs)
        plt.plot(recall, precision, color='blue', lw=2.5, label=f'PR curve (AP = {ap:.4f})')
    except ValueError as e:
        print(f"[{run_name}] 警告: 无法计算 PR 曲线数据. 错误: {e}")
        plt.text(0.5, 0.5, 'PR curve calculation failed\n(e.g. single class in labels)', 
                 horizontalalignment='center', verticalalignment='center', color='red')

    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve', fontsize=14)
    plt.legend(loc="lower left", fontsize=11)
    plt.xlim([-0.02, 1.0])
    plt.ylim([-0.02, 1.05])
    plt.grid(True, linestyle='--', alpha=0.6)
    save_plot('pr_curve', run_name)
    if not math.isnan(ap):
        print(f"[{run_name}] Average Precision (AP) for PR curve: {ap:.4f}")
    return ap

def visualize_gradcam(model, dataloader, device='cuda', target_layer_name='stage5', run_name="default_run", num_tp=2, num_tn=2):
    model.eval()
    try:
        target_layer = dict(model.named_modules())[target_layer_name]
        print(f"[{run_name}] Grad-CAM 目标层: {target_layer_name}")
    except KeyError:
        print(f"错误: 目标层 '{target_layer_name}' 在模型中未找到。可用层:")
        # 打印部分可用层名辅助调试
        count = 0
        for name, module in model.named_modules():
             if not isinstance(module, (nn.Sequential, nn.ModuleList, nn.ModuleDict)) and list(module.parameters(recurse=False)):
                 print(f"- {name}")
                 count +=1
                 if count > 10: print("... (更多层未显示)"); break # 避免过多输出
        return

    gradcam = GradCAM(model, target_layer)
    
    tp_samples_data = [] 
    tn_samples_data = []

    denormalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225], 
                                     std=[1/0.229, 1/0.224, 1/0.225])
    
    print(f"[{run_name}] 正在搜索 {num_tp} 个真阳性 (TP) 和 {num_tn} 个真阴性 (TN) 样本用于 Grad-CAM...")
    collected_tp = 0
    collected_tn = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader: 
            if collected_tp >= num_tp and collected_tn >= num_tn: break 

            inputs_device = inputs.to(device)
            # outputs, probs, preds 在循环内部处理，因为我们需要单个图像的 tensor
            
            for i in range(inputs.size(0)):
                if collected_tp >= num_tp and collected_tn >= num_tn: break

                current_input_tensor = inputs[i].unsqueeze(0).to(device) # 单个图像的tensor，到设备
                current_label = labels[i].item()
                
                output_single = model(current_input_tensor)
                prob_single = torch.sigmoid(output_single).item()
                pred_single = 1 if prob_single > 0.5 else 0

                img_for_display = denormalize(inputs[i].cpu()).permute(1, 2, 0).numpy()
                img_for_display = np.clip(img_for_display, 0, 1)
                
                if current_label == 1 and pred_single == 1 and collected_tp < num_tp: # 真阳性
                    tp_samples_data.append({
                        "tensor_device": current_input_tensor, 
                        "display_img": img_for_display,
                        "true_label_str": "Enlarged", 
                        "pred_label_str": "Enlarged", 
                        "prob_str": f"{prob_single:.2f}"
                    })
                    collected_tp += 1
                elif current_label == 0 and pred_single == 0 and collected_tn < num_tn: # 真阴性
                    tn_samples_data.append({
                        "tensor_device": current_input_tensor,
                        "display_img": img_for_display,
                        "true_label_str": "Normal",
                        "pred_label_str": "Normal",
                        "prob_str": f"{prob_single:.2f}"
                    })
                    collected_tn += 1
    
    selected_samples = tp_samples_data + tn_samples_data
    actual_n_images = len(selected_samples)

    if actual_n_images == 0:
        print(f"[{run_name}] 未找到合适的 TP/TN 图像进行 Grad-CAM 可视化。")
        gradcam.remove_hooks()
        return

    print(f"[{run_name}] 已找到 {len(tp_samples_data)} 个 TP 和 {len(tn_samples_data)} 个 TN 样本。正在生成 Grad-CAM...")

    fig_gradcam, axes_gradcam = plt.subplots(actual_n_images, 2, figsize=(10, 4 * actual_n_images if actual_n_images > 0 else 4))
    if actual_n_images == 1: axes_gradcam = np.array([axes_gradcam]) 
    elif actual_n_images == 0: # 以防万一，虽然前面有检查
        gradcam.remove_hooks()
        plt.close(fig_gradcam)
        return
        
    fig_gradcam.suptitle('Grad-CAM (Correct Predictions: TP & TN)', fontsize=16, y=1.02 if actual_n_images > 1 else 1.05)

    for i, sample_data in enumerate(selected_samples):
        img_tensor_device = sample_data["tensor_device"] # 已在设备上
        img_vis = sample_data["display_img"]
        
        current_axes_row = axes_gradcam[i] if actual_n_images > 1 else axes_gradcam # 处理单行情况

        try:
            # class_idx=0 对应于模型单输出logit（即类别1的logit）
            heatmap = gradcam(img_tensor_device, class_idx=0, retain_graph=True) 
        except RuntimeError as e:
            print(f"[{run_name}] Grad-CAM 调用错误 (样本 {i}): {e}")
            try: gradcam.remove_hooks() # 尝试清理当前gradcam实例的钩子
            except Exception as he: print(f"[{run_name}] Grad-CAM 错误后移除钩子失败: {he}")
            gradcam = GradCAM(model, target_layer) # 为下一个图像重新初始化gradcam
            
            current_axes_row[0].text(0.5, 0.5, 'Grad-CAM Error', ha='center', va='center')
            current_axes_row[0].axis('off')
            current_axes_row[1].text(0.5, 0.5, 'Grad-CAM Error', ha='center', va='center')
            current_axes_row[1].axis('off')
            continue 

        ax_orig = current_axes_row[0]
        ax_orig.imshow(img_vis)
        ax_orig.set_title(f'Original (True: {sample_data["true_label_str"]})\nPred: {sample_data["pred_label_str"]} ({sample_data["prob_str"]})', fontsize=10)
        ax_orig.axis('off')

        ax_overlay = current_axes_row[1]
        ax_overlay.imshow(img_vis)
        ax_overlay.imshow(heatmap, cmap='jet', alpha=0.5)
        ax_overlay.set_title('Grad-CAM Overlay', fontsize=10)
        ax_overlay.axis('off')

    fig_gradcam.tight_layout(rect=[0, 0.03, 1, 0.98 if actual_n_images > 1 else 0.95])
    save_plot('grad_cam_selected_tp_tn', run_name)
    
    gradcam.remove_hooks()


def extract_features(model, dataloader, device='cuda', run_name="default_run"):
    model.eval() 
    all_features, all_labels = [], []
    print(f"[{run_name}] 正在提取 UMAP 特征...")
    with torch.no_grad():
        for inputs, labels in tqdm(dataloader, desc=f"[{run_name}] 特征提取"):
            inputs = inputs.to(device)
            features = model(inputs, return_features=True)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy()) # labels 通常已在CPU
    if not all_features:
        print(f"[{run_name}] 警告: 未提取到任何特征。")
        return np.array([]), np.array([])
    all_features = np.concatenate(all_features, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    return all_features, all_labels

def plot_umap(features, labels, n_neighbors=15, min_dist=0.1, metric='euclidean', run_name="default_run"):
    if features.ndim != 2 or features.shape[0] < 2: 
        print(f"[{run_name}] 跳过 UMAP: 特征样本不足或维度不正确 (shape: {features.shape})。")
        return
    if len(np.unique(labels)) < 1: 
         print(f"[{run_name}] 跳过 UMAP: 标签类别不足 ({len(np.unique(labels))})。")
         return
    
    # 确保 n_neighbors 不超过样本数减一
    n_neighbors_actual = min(n_neighbors, features.shape[0] - 1)
    if n_neighbors_actual < 1 : # UMAP 要求 n_neighbors >= 1
         print(f"[{run_name}] 跳过 UMAP: 样本数太少 ({features.shape[0]}) 导致无法设置有效的 n_neighbors ({n_neighbors_actual})。")
         return

    # --- UMAP 2D ---
    print(f"[{run_name}] 运行 UMAP 2D (邻居数={n_neighbors_actual}, 最小距离={min_dist})...")
    reducer_2d = umap.UMAP(n_neighbors=n_neighbors_actual, min_dist=min_dist, metric=metric, random_state=42, n_components=2)
    try:
        embedding_2d = reducer_2d.fit_transform(features)
    except Exception as e:
        print(f"[{run_name}] 运行 UMAP 2D 时出错: {e}")
        # 可能不需要立即返回，如果3D部分可以独立尝试
    else:
        plt.figure(figsize=(10, 8))
        # colors = ['blue' if l == 0 else 'red' for l in labels] # 假设0和1
        # 更通用的颜色映射
        unique_label_values = sorted(np.unique(labels))
        color_map = {val: plt.cm.jet(i / max(1, len(unique_label_values)-1 )) for i, val in enumerate(unique_label_values)}
        colors = [color_map.get(l, 'gray') for l in labels]


        plt.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=15, alpha=0.7)
        handles = []
        # unique_labels = np.unique(labels) # 已在上面计算
        if 0 in unique_label_values: handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Normal (0)', markersize=10, markerfacecolor=color_map.get(0,'blue')))
        if 1 in unique_label_values: handles.append(plt.Line2D([0], [0], marker='o', color='w', label='Enlarged (1)', markersize=10, markerfacecolor=color_map.get(1,'red')))
        # 如果有更多类别，可以动态添加图例
        
        plt.title('UMAP 2D Projection of Feature Space', fontsize=14)
        plt.xlabel('UMAP Component 1', fontsize=12)
        plt.ylabel('UMAP Component 2', fontsize=12)
        if handles: plt.legend(handles=handles, title="Class")
        plt.grid(True, linestyle='--', alpha=0.5) 
        plt.gca().set_aspect('equal', 'datalim') 
        save_plot('umap_feature_visualization_2d', run_name)

    # --- UMAP 3D ---
    if features.shape[0] < 3 and n_components ==3 : # UMAP for 3D might need more samples
        print(f"[{run_name}] 跳过 UMAP 3D: 样本数 ({features.shape[0]}) 可能不足以进行3D降维。")
        return

    print(f"[{run_name}] 运行 UMAP 3D (邻居数={n_neighbors_actual}, 最小距离={min_dist})...")
    reducer_3d = umap.UMAP(n_neighbors=n_neighbors_actual, min_dist=min_dist, metric=metric, random_state=42, n_components=3)
    try:
        embedding_3d = reducer_3d.fit_transform(features)
    except Exception as e:
        print(f"[{run_name}] 运行 UMAP 3D 时出错: {e}")
        return # 如果3D出错，则不继续绘制3D图

    fig_3d = plt.figure(figsize=(12, 10))
    ax_3d = fig_3d.add_subplot(111, projection='3d')
    # 使用与2D图相同的颜色逻辑
    unique_label_values_3d = sorted(np.unique(labels)) # Re-calculate or pass from 2D part
    color_map_3d = {val: plt.cm.jet(i / max(1, len(unique_label_values_3d)-1 )) for i, val in enumerate(unique_label_values_3d)}
    colors_3d = [color_map_3d.get(l, 'gray') for l in labels]

    ax_3d.scatter(embedding_3d[:, 0], embedding_3d[:, 1], embedding_3d[:, 2], c=colors_3d, s=15, alpha=0.7)
    ax_3d.set_title('UMAP 3D Projection of Feature Space', fontsize=14)
    ax_3d.set_xlabel('UMAP Component 1', fontsize=12)
    ax_3d.set_ylabel('UMAP Component 2', fontsize=12)
    ax_3d.set_zlabel('UMAP Component 3', fontsize=12)
    
    handles_3d = []
    if 0 in unique_label_values_3d: handles_3d.append(plt.Line2D([0], [0], marker='o', color='w', label='Normal (0)', markersize=10, markerfacecolor=color_map_3d.get(0,'blue')))
    if 1 in unique_label_values_3d: handles_3d.append(plt.Line2D([0], [0], marker='o', color='w', label='Enlarged (1)', markersize=10, markerfacecolor=color_map_3d.get(1,'red')))
    if handles_3d: ax_3d.legend(handles=handles_3d, title="Class")
    
    save_plot('umap_feature_visualization_3d', run_name, is_3d=True)

# -----------------------------
# 主执行函数
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description='Train Enhanced CNN with MARS-AdamW or PyTorch AdamW')
    parser.add_argument('--data_dir_train', type=str, default='./train', help='训练数据目录路径')
    parser.add_argument('--data_dir_test', type=str, default='./test', help='测试数据目录路径')
    parser.add_argument('--batch_size', type=int, default=64, help='批量大小 (原为64, 调整为32尝试减少显存)')
    parser.add_argument('--epochs', type=int, default=52, help='训练轮数 (原为70, 调整为50进行测试)')
    parser.add_argument('--lr', type=float, default=5e-4, help='基础学习率 (原为5e-4)')
    parser.add_argument('--wd', type=float, default=1e-3, help='权重衰减')
    parser.add_argument('--warmup_epochs', type=int, default=3, help='学习率预热轮数')
    parser.add_argument('--focal_alpha', type=float, default=0.35, help='Focal Loss alpha 参数')
    parser.add_argument('--focal_gamma', type=float, default=2.0, help='Focal Loss gamma 参数 (原为2.0)') 
    parser.add_argument('--num_workers', type=int, default=4, help='数据加载器工作进程数 (原为4)') 
    parser.add_argument('--device', type=str, default='auto', help='设备选择 (cuda, cpu, auto)')
    parser.add_argument('--use_classic_cbam', action='store_true', help='使用经典 CBAM 而非增强型 CBAM')
    parser.add_argument('--use_pytorch_adamw', action='store_true', help='启用时使用 PyTorch 官方 AdamW, 否则使用自定义 MARS-AdamW')
    parser.add_argument('--mars_betas', type=float, nargs=2, default=[0.9, 0.99], help='MARS-AdamW 或 AdamW 的 beta 参数 (beta1, beta2)')
    parser.add_argument('--mars_gamma', type=float, default=0.01, help='MARS-AdamW gamma 参数 (0 关闭 MARS 效应)')
    args = parser.parse_args()

    cbam_type = 'classic' if args.use_classic_cbam else 'enhanced'
    optimizer_name_suffix = "pytorch_adamw" if args.use_pytorch_adamw else f"mars_g{args.mars_gamma:.2f}".replace(".","_")
    run_name = f"cbam_{cbam_type}_opt_{optimizer_name_suffix}_lr{args.lr:.0e}_wd{args.wd:.0e}_ep{args.epochs}"
    print(f"运行名称: {run_name}")
    print(f"参数设置: {args}")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else torch.device(args.device)
    print(f"使用设备: {device}")

    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256), 
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), 
            transforms.RandomHorizontalFlip(), 
            transforms.RandomRotation(15), 
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05), 
            transforms.ToTensor(), 
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256), 
            transforms.CenterCrop(224), 
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    data_dir = {'train': args.data_dir_train, 'test': args.data_dir_test}
    print("加载数据集中...")
    try:
        image_datasets = {x: ChestXrayDataset(data_dir[x], data_transforms[x]) for x in ['train','test']}
    except FileNotFoundError as e:
        print(f"错误: 加载数据集失败: {e}\n请确保 '{args.data_dir_train}' 和 '{args.data_dir_test}' 目录存在，并包含 'true'/'false' 子目录。")
        return
    if len(image_datasets['train']) == 0: print("错误: 训练数据集为空。请检查路径和图像文件。"); return
    if len(image_datasets['test']) == 0: print("警告: 测试数据集为空。评估和部分可视化将无法进行。") # 改为警告，允许只训练

    dataloaders = {
        x: DataLoader(image_datasets[x], batch_size=args.batch_size,
                      shuffle=(x=='train'), num_workers=args.num_workers, pin_memory=(device.type=='cuda')) # pin_memory仅在CUDA上有效
        for x in ['train','test']
    }
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'test']}
    print(f"数据集大小: 训练集={dataset_sizes['train']}, 测试集={dataset_sizes['test']}")

    model = EnhancedCNN(num_classes=1, use_enhanced_cbam=(not args.use_classic_cbam)).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"模型参数: 总计={total_params:,}, 可训练={trainable_params:,}")

    if args.use_pytorch_adamw:
        print("设置优化器: PyTorch 官方 AdamW")
        optimizer = optim.AdamW(model.parameters(), lr=args.lr, betas=tuple(args.mars_betas), weight_decay=args.wd, eps=1e-8)
    else:
        print(f"设置优化器: 自定义 MARS-AdamW (gamma={args.mars_gamma})")
        optimizer = MARS_AdamW(model.parameters(), lr=args.lr, betas=tuple(args.mars_betas), weight_decay=args.wd, gamma=args.mars_gamma, eps=1e-8, amsgrad=False)
    print(f"  优化器参数: lr={args.lr}, betas={tuple(args.mars_betas)}, wd={args.wd}")
    if not args.use_pytorch_adamw: print(f"  MARS Gamma: {args.mars_gamma}")

    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma, pos_weight=None) # pos_weight可以根据类别比例设置
    print(f"损失函数: Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})")

    actual_warmup_epochs = min(args.warmup_epochs, args.epochs) if args.epochs > 0 else 0
    scheduler = None
    if args.epochs > 0: # 仅当训练时设置调度器
        if actual_warmup_epochs > 0 and args.epochs > actual_warmup_epochs:
            warmup_scheduler = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, total_iters=actual_warmup_epochs)
            cosine_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs - actual_warmup_epochs, eta_min=args.lr * 0.01)
            scheduler = optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[actual_warmup_epochs])
            print(f"学习率调度器: {actual_warmup_epochs} 轮预热 + 余弦退火 (T_max={args.epochs - actual_warmup_epochs}, eta_min={args.lr * 0.01:.2e})")
        elif args.epochs > 0 : 
             scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)
             print(f"学习率调度器: 余弦退火 (T_max={args.epochs}, eta_min={args.lr * 0.01:.2e})")
    
    if scheduler is None and args.epochs > 0: 
        print("学习率调度器: 无 (或恒定学习率)")
    elif args.epochs == 0:
        print("训练轮数为0，跳过训练。")


    start_train_time = time.time()
    best_test_acc_from_train = 0.0 # 重命名以区分
    if args.epochs > 0 :
        model, history, best_test_acc_from_train = train_model(
            model, dataloaders, optimizer, criterion, scheduler,
            num_epochs=args.epochs, device=device, run_name=run_name
        )
        training_duration = time.time() - start_train_time
        print(f"[{run_name}] 总训练时间: {training_duration // 60:.0f} 分 {training_duration % 60:.2f} 秒")
        
        best_model_path = os.path.join('results', f'{run_name}_best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"[{run_name}] 最佳模型权重已保存到: {best_model_path}")
        
        if history and history.get('train_acc'): # 确保 history 有内容
            plot_metrics(history, run_name=run_name)
        else:
            print(f"[{run_name}] 训练历史记录为空，无法绘制训练曲线。")
    else: # epochs == 0
        history = {} # 空历史记录
        training_duration = 0
        print(f"[{run_name}] 未进行训练。如需评估预加载模型，请确保模型已加载。")
        # 此处可以添加加载预训练权重的逻辑 if args.load_model_path ...

    # --- 评估和可视化 ---
    if dataset_sizes['test'] > 0:
        print(f"\n--- [{run_name}] 在测试集上进行最终评估 ---")
        test_labels, test_probs, test_preds = evaluate_model(model, dataloaders['test'], device, run_name=run_name)

        if len(test_labels) == 0: # 再次检查，以防 evaluate_model 返回空
            print(f"[{run_name}] 评估失败或测试集为空，没有结果可处理。")
        else:
            final_roc_auc = print_and_save_evaluation_report(test_labels, test_preds, test_probs, run_name=run_name)
            plot_confusion_matrix(test_labels, test_preds, run_name=run_name)
            plot_roc_curve(test_labels, test_probs, run_name=run_name)
            plot_pr_curve(test_labels, test_probs, run_name=run_name) # <<< 新增PR曲线调用

            try:
                target_gradcam_layer = 'stage5' 
                if target_gradcam_layer not in dict(model.named_modules()):
                     print(f"[{run_name}] 警告: 模型没有 '{target_gradcam_layer}' 层, 尝试 'stage4'.")
                     target_gradcam_layer = 'stage4' 
                
                if target_gradcam_layer in dict(model.named_modules()):
                     visualize_gradcam(model, dataloaders['test'], device, 
                                       target_layer_name=target_gradcam_layer, run_name=run_name,
                                       num_tp=2, num_tn=2) # GradCAM 挑选2 TP + 2 TN
                else:
                     print(f"[{run_name}] 警告: 模型也没有 '{target_gradcam_layer}' (或 'stage4') 层, 跳过 Grad-CAM.")
            except Exception as e:
                print(f"[{run_name}] 无法生成 Grad-CAM 可视化: {e}")
                import traceback
                traceback.print_exc()


            try:
                features, umap_labels = extract_features(model, dataloaders['test'], device, run_name=run_name)
                if features.ndim == 2 and features.shape[0] >= 2: 
                    plot_umap(features, umap_labels, run_name=run_name)
                else:
                    print(f"[{run_name}] 跳过 UMAP 绘图: 特征样本不足或维度不正确 (shape: {features.shape if hasattr(features, 'shape') else 'N/A'})。")
            except Exception as e:
                print(f"[{run_name}] 无法生成 UMAP 可视化: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"[{run_name}] 测试集为空，跳过评估和相关可视化。")


    summary_filename = os.path.join('results', f"{run_name}_final_summary.txt")
    with open(summary_filename, 'w') as f:
        f.write(f"--- [{run_name}] 实验总结 ---\n")
        f.write(f"时间戳: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"运行名称: {run_name}\n")
        f.write(f"命令行参数: {args}\n")
        f.write(f"模型结构: EnhancedCNN with Stage5 (1024 ch), New Classifier Head\n")
        f.write(f"CBAM Type: {'Enhanced (Scharr, Tanh(ax))' if not args.use_classic_cbam else 'Classic'}\n")
        f.write(f"设备: {device}\n")
        f.write(f"优化器: {'PyTorch AdamW' if args.use_pytorch_adamw else f'MARS-AdamW (gamma={args.mars_gamma})'}\n")
        f.write(f"损失函数: Focal Loss (alpha={args.focal_alpha}, gamma={args.focal_gamma})\n")
        f.write(f"总训练轮数: {args.epochs}\n")
        f.write(f"总训练时间: {training_duration // 60:.0f} min {training_duration % 60:.2f} sec\n")
        f.write(f"最佳测试准确率 (训练中): {best_test_acc_from_train:.4f}\n")
        
        if dataset_sizes['test'] > 0 and os.path.exists(os.path.join('results', f"{run_name}_evaluation_report.txt")):
            try:
                with open(os.path.join('results', f"{run_name}_evaluation_report.txt"), 'r') as report_f:
                     f.write("\n--- 最终测试集评估 ---\n")
                     # 跳过报告中的标题行
                     report_content = report_f.readlines()
                     if len(report_content) > 1:
                         f.writelines(report_content[1:])
                     else:
                         f.writelines(report_content)

            except FileNotFoundError:
                f.write("评估报告文件未找到。\n")
        else:
            f.write("未进行最终测试集评估或报告未生成。\n")
        f.write("-------------------------\n")
    print(f"[{run_name}] 最终总结已保存到: {summary_filename}")
    print(f"--- [{run_name}] 脚本执行完毕 ---")

if __name__ == '__main__':
    main()