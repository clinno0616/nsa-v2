import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
import json
import numpy as np
from typing import Optional
import math
import logging
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from torch.nn.parallel import DataParallel
from torch.cuda.amp import autocast, GradScaler
from torch.optim import lr_scheduler
import gc
import os
import time, datetime
import torch.nn.functional as F
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import time
import json
import os
import numpy as np
from training_logger import TrainingLogger
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import glob
import pandas as pd
import sentencepiece as spm
# 设置日志记录
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 設置 CUDA 記憶體分配器配置
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'


class WarmupCosineScheduler:
    """實現帶預熱的余弦學習率調度器"""
    
    def __init__(
        self,
        optimizer,
        warmup_steps,
        total_steps,
        min_lr_ratio=0.1,
        last_epoch=-1
    ):
        """
        初始化學習率調度器
        
        參數:
            optimizer: PyTorch 優化器
            warmup_steps: 預熱步數
            total_steps: 總訓練步數
            min_lr_ratio: 最小學習率與初始學習率的比例
            last_epoch: 上一個 epoch (-1 表示從頭開始)
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr_ratio = min_lr_ratio
        self.last_epoch = last_epoch
        
        # 獲取初始學習率
        self.base_lrs = []
        for group in optimizer.param_groups:
            self.base_lrs.append(group['lr'])
        
        # 存儲當前學習率
        self._last_lr = list(self.base_lrs)
        
        # 初始化
        self.step(last_epoch + 1)
    
    def get_lr(self, step):
        """計算特定步數的學習率"""
        if step < self.warmup_steps:
            # 線性預熱
            return [base_lr * (step / self.warmup_steps) for base_lr in self.base_lrs]
        else:
            # 余弦衰減
            progress = (step - self.warmup_steps) / max(1, self.total_steps - self.warmup_steps)
            cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
            decay = self.min_lr_ratio + (1 - self.min_lr_ratio) * cosine_factor
            return [base_lr * decay for base_lr in self.base_lrs]
    
    def get_last_lr(self):
        """返回最後設置的學習率"""
        return self._last_lr
    
    def step(self, epoch=None):
        """執行一步學習率調整"""
        if epoch is None:
            self.last_epoch += 1
            epoch = self.last_epoch
        else:
            self.last_epoch = epoch
        
        lrs = self.get_lr(epoch)
        self._last_lr = lrs  # 保存當前學習率
        
        for param_group, lr in zip(self.optimizer.param_groups, lrs):
            param_group['lr'] = lr
        
        return lrs

class TrainingMetrics:
    def __init__(self):
        self.train_losses = []  # 每個batch的loss
        self.epoch_losses = []  # 每個epoch的平均loss
        self.validation_losses = []  # 每次驗證的loss
        self.learning_rates = []  # 學習率追蹤
        self.gpu_memory_usage = []  # GPU記憶體使用追蹤
        self.best_loss = float('inf')  # 最佳loss
        self.no_improvement_count = 0  # 用於early stopping
        
    def update_batch_loss(self, loss):
        self.train_losses.append(loss)
        
    def update_epoch_loss(self, loss):
        self.epoch_losses.append(loss)
        
    def update_validation_loss(self, loss):
        self.validation_losses.append(loss)
        if loss < self.best_loss:
            self.best_loss = loss
            self.no_improvement_count = 0
            return True
        else:
            self.no_improvement_count += 1
            return False
            
    def should_early_stop(self, patience=3):
        return self.no_improvement_count >= patience

# 添加到app5.py中的动态稀疏度调整类

class DynamicSparsityController(nn.Module):
    """根据输入序列特性动态调整稀疏参数"""
    def __init__(self, 
                 base_select_k=16, 
                 min_select_k=8, 
                 max_select_k=32, 
                 base_compression_ratio=4,
                 min_compression_ratio=2,
                 max_compression_ratio=8,
                 entropy_factor=0.5):
        super().__init__()
        self.base_select_k = base_select_k
        self.min_select_k = min_select_k
        self.max_select_k = max_select_k
        
        self.base_compression_ratio = base_compression_ratio
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        
        self.entropy_factor = entropy_factor
        
    def _compute_attention_entropy(self, attention_scores):
        """计算注意力分布的熵，用于评估注意力的分散程度"""
        # 将attention_scores归一化为概率分布
        probs = F.softmax(attention_scores, dim=-1)
        # 计算熵: -sum(p * log(p))
        log_probs = torch.log(probs + 1e-10)  # 添加小常数避免log(0)
        entropy = -torch.sum(probs * log_probs, dim=-1)
        # 返回平均熵
        return entropy.mean()
    
    def compute_dynamic_sparsity(self, hidden_states, attention_scores=None):
        """根据输入特征计算合适的稀疏参数"""
        # 如果提供了attention_scores，使用熵进行计算
        if attention_scores is not None:
            entropy = self._compute_attention_entropy(attention_scores)
            entropy_normalized = torch.clamp(entropy / math.log(attention_scores.size(-1)), 0, 1)
            
            # 根据熵调整select_k (熵高意味着注意力分散，需要更多选择点)
            effective_k = int(self.base_select_k * (1 + (entropy_normalized - 0.5) * self.entropy_factor))
            effective_k = min(self.max_select_k, max(self.min_select_k, effective_k))
            
            # 根据熵调整compression_ratio (熵高时使用更小的压缩比)
            compression_ratio = int(self.base_compression_ratio / (1 + (entropy_normalized - 0.5) * self.entropy_factor))
            compression_ratio = min(self.max_compression_ratio, max(self.min_compression_ratio, compression_ratio))
            
            return effective_k, compression_ratio
        
        # 如果没有attention_scores，使用序列长度和内容复杂度进行简单估计
        seq_length = hidden_states.size(1)
        # 使用hidden_states的方差作为内容复杂度的简单度量
        complexity = torch.var(hidden_states).item()
        
        # 根据序列长度和复杂度调整select_k
        length_factor = min(1.0, seq_length / 1024)  # 长序列用更多块
        complexity_factor = min(1.0, complexity)  # 更复杂的内容用更多块
        
        effective_k = int(self.base_select_k * (1 + length_factor * 0.5 + complexity_factor * 0.5))
        effective_k = min(self.max_select_k, max(self.min_select_k, effective_k))
        
        return effective_k, self.base_compression_ratio


class OptimizedTrainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        gradient_accumulation_steps: int = 8,
        mixed_precision: bool = True,
        early_stopping_patience: int = 3,
        logging_steps: int = 10
    ):
        # 檢測可用的GPU並計算可用VRAM
        self.n_gpu = torch.cuda.device_count()
        self.devices = []
        total_memory = 0
        
        for i in range(self.n_gpu):
            memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
            if i == 0 or memory >= (torch.cuda.get_device_properties(0).total_memory / 1024**3) * 0.75:
                self.devices.append(i)
                total_memory += memory
                
        self.n_gpu = len(self.devices)
        self.main_device = "cuda:0"
        
        # 調整batch size和梯度累積步數
        self.adjusted_batch_size = self._calculate_safe_batch_size(batch_size, total_memory)
        self.gradient_accumulation_steps = self._calculate_accumulation_steps(
            batch_size, 
            self.adjusted_batch_size
        )
        
        print(f"Using {self.n_gpu} GPUs: {self.devices}")
        print(f"Original batch size: {batch_size}")
        print(f"Adjusted batch size: {self.adjusted_batch_size}")
        print(f"Gradient accumulation steps: {self.gradient_accumulation_steps}")
        
        # 清理GPU記憶體
        self._clean_gpu_memory()
        
        # 移動模型到主GPU並啟用記憶體優化
        self.model = model.to(self.main_device)
        if self.n_gpu > 1:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.devices)
        
        # 配置數據加載器，使用較小的prefetch factor
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.adjusted_batch_size,
            shuffle=True,
            num_workers=2,  # 減少worker數量
            pin_memory=True,
            prefetch_factor=2,  # 限制預取量
            persistent_workers=True  # 保持worker進程
        )
        
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=self.adjusted_batch_size,
                shuffle=False,
                num_workers=2,
                pin_memory=True,
                prefetch_factor=2,
                persistent_workers=True
            )
        else:
            self.test_loader = None
            
        # 調整學習率
        adjusted_lr = learning_rate * math.sqrt(
            self.adjusted_batch_size * self.gradient_accumulation_steps / batch_size
        )
        
        # 配置優化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=adjusted_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
        # 配置學習率調度器
        self.scheduler = lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=5,  # 每5個epoch重啟
            T_mult=2,
            eta_min=adjusted_lr * 0.01
        )
        
        # 混合精度訓練設置
        self.mixed_precision = mixed_precision
        self.scaler = GradScaler() if mixed_precision else None
        
        # 訓練參數
        self.num_epochs = num_epochs
        
        # 顯示GPU信息
        self._print_gpu_info()

        # 添加新的訓練指標追蹤
        self.metrics = TrainingMetrics()
        self.early_stopping_patience = early_stopping_patience
        self.logging_steps = logging_steps
            
    def _calculate_safe_batch_size(self, original_batch_size, total_memory):
        """計算安全的batch size"""
        # 假設每個樣本約需要0.5GB VRAM，再加上模型和優化器的開銷
        safe_batch_size = int((total_memory * 0.3) / 0.5)  # 只使用30%的VRAM給batch
        safe_batch_size = min(original_batch_size, safe_batch_size)
        safe_batch_size = max(1, (safe_batch_size // 8) * 8)  # 確保是8的倍數
        return safe_batch_size
        
    def _calculate_accumulation_steps(self, target_batch_size, actual_batch_size):
        """計算需要的梯度累積步數"""
        steps = math.ceil(target_batch_size / actual_batch_size)
        return max(steps, 8)  # 至少8步
        
    def _clean_gpu_memory(self):
        """清理GPU記憶體"""
        torch.cuda.empty_cache()
        gc.collect()
        
    def _print_gpu_info(self):
        """打印GPU信息"""
        for i in self.devices:
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: {total:.1f}GB total, "
                  f"{allocated:.1f}GB allocated, "
                  f"{cached:.1f}GB cached")
            
    def _check_memory(self):
        """監控GPU記憶體使用"""
        for i in self.devices:
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            cached = torch.cuda.memory_reserved(i) / 1024**3
            free = (torch.cuda.get_device_properties(i).total_memory / 1024**3) - allocated
            print(f"GPU {i} - Allocated: {allocated:.2f}GB, "
                  f"Cached: {cached:.2f}GB, "
                  f"Free: {free:.2f}GB")
            
    @torch.cuda.amp.autocast()
    def _forward_pass(self, batch):
        """執行前向傳播"""
        try:
            input_ids = batch["input_ids"].to(self.main_device, non_blocking=True)
            attention_mask = batch["attention_mask"].to(self.main_device, non_blocking=True)
            labels = batch["labels"].to(self.main_device, non_blocking=True)
            
            output = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            # If output is a tensor and not a scalar, reduce it (e.g., take mean)
            if isinstance(output, torch.Tensor) and output.numel() > 1:
                return output.mean()
            return output
        except RuntimeError as e:
            if "out of memory" in str(e):
                self._clean_gpu_memory()
                raise e
            raise e
    def _log_training_info(self, epoch, batch_idx, loss, lr):
        """記錄詳細的訓練資訊"""
        # 計算進度百分比
        progress = (batch_idx + 1) / len(self.train_loader) * 100
        
        # 計算預估剩餘時間
        batch_time = time.time() - self.last_log_time
        remaining_batches = len(self.train_loader) - (batch_idx + 1)
        eta = datetime.timedelta(seconds=int(batch_time * remaining_batches))
        
        # 更新時間戳
        self.last_log_time = time.time()
        
        # 獲取GPU記憶體使用情況
        gpu_memory = {i: torch.cuda.memory_allocated(i) / 1024**3 for i in self.devices}
        
        # 格式化輸出
        log_str = (
            f"Epoch: {epoch+1}/{self.num_epochs} | "
            f"Progress: {progress:.2f}% | "
            f"Batch: {batch_idx+1}/{len(self.train_loader)} | "
            f"Loss: {loss:.12f} | "
            f"LR: {lr:.12f} | "
            f"ETA: {eta} | "
            f"GPU Memory: {gpu_memory}"
        )
        
        # 如果有驗證集，添加最佳驗證loss
        if self.test_loader:
            log_str += f" | Best Val Loss: {self.metrics.best_loss:.6f}"
        
        logger.info(log_str)
        
        # 更新指標
        self.metrics.update_batch_loss(loss)
        self.metrics.learning_rates.append(lr)
        self.metrics.gpu_memory_usage.append(gpu_memory)    

    def train(self):
        """優化後的訓練循環，包含更詳細的訓練資訊"""
        self.model.train()
        self._clean_gpu_memory()
        self.last_log_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # 使用混合精度訓練
                    if self.mixed_precision:
                        with autocast():
                            loss = self._forward_pass(batch)
                            loss = loss / self.gradient_accumulation_steps
                            
                        self.scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            self.scheduler.step()
                    else:
                        loss = self._forward_pass(batch)
                        loss = loss / self.gradient_accumulation_steps
                        loss.backward()
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.scheduler.step()
                    
                    total_loss += loss.item() * self.gradient_accumulation_steps
                    num_batches += 1
                    
                    # 定期記錄訓練資訊
                    if (batch_idx + 1) % self.logging_steps == 0:
                        avg_loss = total_loss / num_batches
                        lr = self.scheduler.get_last_lr()[0]
                        self._log_training_info(epoch, batch_idx, avg_loss, lr)
                        self._check_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("WARNING: out of memory, attempting recovery...")
                        self._clean_gpu_memory()
                        
                        # 減少batch size並重新配置
                        self.adjusted_batch_size = max(1, self.adjusted_batch_size // 2)
                        self.gradient_accumulation_steps *= 2
                        logger.info(f"Reducing batch size to {self.adjusted_batch_size}")
                        logger.info(f"Increasing accumulation steps to {self.gradient_accumulation_steps}")
                        
                        # 重新創建數據加載器
                        self.train_loader = DataLoader(
                            self.train_loader.dataset,
                            batch_size=self.adjusted_batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True
                        )
                        
                        continue
                    else:
                        raise e
            # 計算並記錄epoch統計資訊
            epoch_avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f"\nEpoch {epoch+1} Summary:\n"
                f"Average Loss: {epoch_avg_loss:.12f}\n"
                f"Time Elapsed: {datetime.timedelta(seconds=int(epoch_time))}\n"
                f"Learning Rate: {self.scheduler.get_last_lr()[0]:.12f}"
            )
            
            # 更新epoch指標
            self.metrics.update_epoch_loss(epoch_avg_loss)
            
            # 評估並檢查early stopping
            if self.test_loader:
                val_loss = self.evaluate()
                improved = self.metrics.update_validation_loss(val_loss)
                
                if improved:
                    logger.info("New best validation loss! Saving model checkpoint...")
                    self.save_model("best_model_checkpoint.pth")
                
                if self.metrics.should_early_stop(self.early_stopping_patience):
                    logger.info(
                        f"Early stopping triggered after {self.early_stopping_patience} "
                        "epochs without improvement"
                    )
                    break

            # 評估
            if self.test_loader:
                self.evaluate()
                
    def evaluate(self):
        """增強的評估函數"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        eval_start_time = time.time()
        
        with torch.no_grad():
            for batch in self.test_loader:
                if self.mixed_precision:
                    with autocast():
                        loss = self._forward_pass(batch)
                else:
                    loss = self._forward_pass(batch)
                    
                total_loss += loss.item()
                num_batches += 1
                
        avg_loss = total_loss / num_batches
        eval_time = time.time() - eval_start_time
        
        logger.info(
            f"\nValidation Results:\n"
            f"Average Loss: {avg_loss:.12f}\n"
            f"Time Elapsed: {datetime.timedelta(seconds=int(eval_time))}"
        )
        
        return avg_loss
        
    def save_model(self, path: str):
        """保存模型"""
        if isinstance(self.model, nn.DataParallel):
            torch.save(self.model.module.state_dict(), path)
        else:
            torch.save(self.model.state_dict(), path)
            
    def load_model(self, path: str):
        """加載模型"""
        if isinstance(self.model, nn.DataParallel):
            self.model.module.load_state_dict(torch.load(path))
        else:
            self.model.load_state_dict(torch.load(path))

# 修改 OptimizedTrainer 類中的相關方法來使用 TrainingLogger
class OptimizedTrainerWithLogging(OptimizedTrainer):
    def __init__(
        self,
        model,
        train_dataset,
        test_dataset=None,
        batch_size=32,
        learning_rate=5e-5,
        num_epochs=3,
        gradient_accumulation_steps=4,
        mixed_precision=True,
        early_stopping_patience=3,
        logging_steps=10,
        model_name="nsa_model",  # 添加模型名稱參數
        log_dir="training_logs"  # 添加日誌目錄參數
    ):
        # 呼叫原始初始化方法
        super().__init__(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            gradient_accumulation_steps=gradient_accumulation_steps,
            mixed_precision=mixed_precision,
            early_stopping_patience=early_stopping_patience,
            logging_steps=logging_steps
        )
        
        # 初始化訓練記錄器
        self.logger = TrainingLogger(log_dir=log_dir, model_name=model_name)
        
        # 記錄模型配置
        if hasattr(model, 'config'):
            self.logger.record_config(model.config)
        
        # 自動開始記錄
        self.logger.start_recording(interval=10.0)  # 每10秒記錄一次
        
    def _log_training_info(self, epoch, batch_idx, loss, lr):
        """重寫記錄訓練信息的方法，添加訓練記錄器支持"""
        # 調用原始方法
        super()._log_training_info(epoch, batch_idx, loss, lr)
        
        # 同時記錄到訓練記錄器
        self.logger.record_batch(batch_idx, loss, lr)
        
        # 記錄動態稀疏度參數（如果模型支持）
        if hasattr(self.model, 'get_sparsity_metrics'):
            if isinstance(self.model, torch.nn.DataParallel):
                metrics = self.model.module.get_sparsity_metrics()
            else:
                metrics = self.model.get_sparsity_metrics()
                
            if metrics.get("dynamic_sparsity_enabled", False):
                # 獲取最新的壓縮比和選擇數
                history = metrics.get("history", {})
                steps = history.get("steps", [])
                compression_ratios = history.get("compression_ratios", [])
                select_ks = history.get("select_ks", [])
                
                if steps and compression_ratios and select_ks:
                    self.logger.record_sparsity_metrics(
                        is_enabled=True,
                        compression_ratio=compression_ratios[-1],
                        select_k=select_ks[-1],
                        step=steps[-1]
                    )
        
    def train(self):
        """重寫訓練方法，添加訓練記錄器支持"""
        self.model.train()
        self._clean_gpu_memory()
        self.last_log_time = time.time()
        
        for epoch in range(self.num_epochs):
            epoch_start_time = time.time()
            total_loss = 0
            num_batches = 0
            self.optimizer.zero_grad()
            
            for batch_idx, batch in enumerate(self.train_loader):
                try:
                    # 使用混合精度訓練
                    if self.mixed_precision:
                        with autocast():
                            loss = self._forward_pass(batch)
                            loss = loss / self.gradient_accumulation_steps
                            
                        self.scaler.scale(loss).backward()
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            self.scaler.unscale_(self.optimizer)
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            self.scheduler.step()
                    else:
                        loss = self._forward_pass(batch)
                        loss = loss / self.gradient_accumulation_steps
                        loss.backward()
                        
                        if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                            self.optimizer.step()
                            self.optimizer.zero_grad()
                            self.scheduler.step()
                    
                    total_loss += loss.item() * self.gradient_accumulation_steps
                    num_batches += 1
                    
                    # 定期記錄訓練資訊
                    if (batch_idx + 1) % self.logging_steps == 0:
                        avg_loss = total_loss / num_batches
                        lr = self.scheduler.get_last_lr()[0]
                        self._log_training_info(epoch, batch_idx, avg_loss, lr)
                        self._check_memory()
                        
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.warning("WARNING: out of memory, attempting recovery...")
                        self._clean_gpu_memory()
                        
                        # 減少batch size並重新配置
                        self.adjusted_batch_size = max(1, self.adjusted_batch_size // 2)
                        self.gradient_accumulation_steps *= 2
                        logger.info(f"Reducing batch size to {self.adjusted_batch_size}")
                        logger.info(f"Increasing accumulation steps to {self.gradient_accumulation_steps}")
                        
                        # 重新創建數據加載器
                        self.train_loader = DataLoader(
                            self.train_loader.dataset,
                            batch_size=self.adjusted_batch_size,
                            shuffle=True,
                            num_workers=2,
                            pin_memory=True,
                            prefetch_factor=2,
                            persistent_workers=True
                        )
                        
                        continue
                    else:
                        raise e
            
            # 計算並記錄epoch統計資訊
            epoch_avg_loss = total_loss / num_batches
            epoch_time = time.time() - epoch_start_time
            
            logger.info(
                f"\nEpoch {epoch+1} Summary:\n"
                f"Average Loss: {epoch_avg_loss:.12f}\n"
                f"Time Elapsed: {datetime.timedelta(seconds=int(epoch_time))}\n"
                f"Learning Rate: {self.scheduler.get_last_lr()[0]:.12f}"
            )
            
            # 更新epoch指標
            self.metrics.update_epoch_loss(epoch_avg_loss)
            
            # 使用訓練記錄器記錄epoch資訊
            gpu_memory = {i: torch.cuda.memory_allocated(i) / 1024**3 for i in self.devices}
            time_elapsed = datetime.timedelta(seconds=int(epoch_time))
            self.logger.record_epoch(
                epoch=epoch+1,
                train_loss=epoch_avg_loss,
                lr=self.scheduler.get_last_lr()[0],
                gpu_memory=gpu_memory,
                time_elapsed=time_elapsed
            )
            
            # 評估並檢查early stopping
            if self.test_loader:
                val_loss = self.evaluate()
                improved = self.metrics.update_validation_loss(val_loss)
                
                # 更新記錄器中的驗證損失
                self.logger.record_epoch(
                    epoch=epoch+1,
                    train_loss=epoch_avg_loss,
                    val_loss=val_loss,
                    lr=self.scheduler.get_last_lr()[0],
                    gpu_memory=gpu_memory,
                    time_elapsed=time_elapsed
                )
                
                if improved:
                    logger.info("New best validation loss! Saving model checkpoint...")
                    checkpoint_path = f"best_model_checkpoint_epoch_{epoch+1}.pth"
                    if (epoch + 1) % 50 == 0:
                        self.save_model(checkpoint_path)
                    
                    # 記錄檢查點
                    self.logger.record_checkpoint(
                        checkpoint_path=checkpoint_path,
                        epoch=epoch+1,
                        val_loss=val_loss,
                        is_best=True
                    )
                
                if self.metrics.should_early_stop(self.early_stopping_patience):
                    logger.info(
                        f"Early stopping triggered after {self.early_stopping_patience} "
                        "epochs without improvement"
                    )
                    break
                
        # 訓練結束後，生成圖表
        self.logger.generate_plots()
        self.logger.print_summary()
        self.logger.stop_recording()
                
    def evaluate(self):
        """重寫評估方法，添加訓練記錄器支持"""
        val_loss = super().evaluate()
        return val_loss
        
    def save_model(self, path: str):
        """重寫保存模型方法，記錄檢查點信息"""
        super().save_model(path)
        
        # 記錄常規檢查點（非最佳）
        if not path.startswith("best_"):
            current_epoch = self.metrics.epoch_losses
            current_epoch = len(current_epoch) if current_epoch else 0
            self.logger.record_checkpoint(
                checkpoint_path=path,
                epoch=current_epoch,
                is_best=False
            )


# 更新創建訓練器的方法，使用新的帶記錄功能的訓練器類
def setup_optimized_trainer_with_warmup_and_logging(
    model,
    train_dataset,
    test_dataset=None,
    batch_size=32,
    learning_rate=5e-5,
    num_epochs=3,
    gradient_accumulation_steps=4,
    mixed_precision=True,
    early_stopping_patience=3,
    warmup_ratio=0.1,
    model_name="nsa_model",
    log_dir="training_logs"
):
    """創建帶有學習率預熱和訓練記錄功能的優化訓練器"""
    trainer = OptimizedTrainerWithLogging(
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        gradient_accumulation_steps=gradient_accumulation_steps,
        mixed_precision=mixed_precision,
        early_stopping_patience=early_stopping_patience,
        model_name=model_name,
        log_dir=log_dir
    )
    
    # 計算總步數和預熱步數
    total_steps = len(trainer.train_loader) * num_epochs // gradient_accumulation_steps
    warmup_steps = int(total_steps * warmup_ratio)
    
    print(f"總訓練步數: {total_steps}, 預熱步數: {warmup_steps}")
    
    # 使用自定義學習率調度器替換原來的調度器
    trainer.scheduler = WarmupCosineScheduler(
        optimizer=trainer.optimizer,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
        min_lr_ratio=0.1
    )
    
    return trainer

# 更新創建不同階段訓練器的函數
def create_trainer_with_warmup_and_logging(phase, model, train_dataset, test_dataset):
    """根據訓練階段創建適當的訓練器，並添加記錄功能"""
    phase_name = f"phase{phase}"
    
    if phase == 1:
        # 第一階段：較高學習率，較長預熱
        return setup_optimized_trainer_with_warmup_and_logging(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=32, 
            learning_rate=5e-5,
            num_epochs=3,
            gradient_accumulation_steps=4,
            mixed_precision=True,
            early_stopping_patience=5,
            warmup_ratio=0.15,  # 較長預熱
            model_name=f"nsa_model_{phase_name}",
            log_dir=f"training_logs_{phase_name}"
        )
    else:
        # 第二階段：較低學習率，較短預熱
        return setup_optimized_trainer_with_warmup_and_logging(
            model=model,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            batch_size=32,
            learning_rate=1e-5,
            #914
            num_epochs=600,
            gradient_accumulation_steps=4,
            mixed_precision=True,
            early_stopping_patience=3,
            warmup_ratio=0.05,  # 較短預熱
            model_name=f"nsa_model_{phase_name}",
            log_dir=f"training_logs_{phase_name}"
        )

class NSAAttentionExtendedWithRouting(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_size
        self.scale = 1.0 / math.sqrt(self.head_size)
        
        # 設定專家數量
        self.num_routed_experts = 4  # 路由專家數量
        self.num_shared_experts = 2  # 共享專家數量
        self.num_total_experts = self.num_routed_experts + self.num_shared_experts
        self.top_k = 2  # 每次選擇的專家數量
        
        # 路由器
        self.router = nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.num_routed_experts)
        )
        
        # 專家初始化
        self.routed_experts = nn.ModuleList([
            self._create_expert() for _ in range(self.num_routed_experts)
        ])
        
        self.shared_experts = nn.ModuleList([
            self._create_expert() for _ in range(self.num_shared_experts)
        ])
        
        # 輸出層
        self.output = nn.Linear(self.hidden_size, self.hidden_size)
        self.dropout = nn.Dropout(0.1)
        
        # 負載平衡係數
        self.router_z_loss_coef = 0.001
        self.expert_capacity_factor = 1.25
        
    def _create_expert(self):
        """創建單個專家模塊"""
        return nn.Sequential(
            nn.Linear(self.hidden_size, self.hidden_size * 4),
            nn.GELU(),
            nn.Linear(self.hidden_size * 4, self.hidden_size),
            nn.Dropout(0.1)
        )
        
    def _compute_routing_probabilities(self, hidden_states):
        """計算路由概率"""
        # 計算路由分數
        routing_logits = self.router(hidden_states)  # [batch, seq_len, num_routed_experts]
        
        # 應用 top-k gating
        top_k_logits, top_k_indices = torch.topk(
            routing_logits, 
            self.top_k, 
            dim=-1
        )
        
        # 計算 softmax 概率
        routing_weights = F.softmax(top_k_logits, dim=-1)
        
        # 計算路由器 z-loss (用於穩定訓練)
        z_loss = torch.mean(torch.square(torch.logsumexp(
            routing_logits, 
            dim=-1
        )))
        
        return routing_weights, top_k_indices, z_loss
        
    def _compute_expert_capacity(self, batch_size, seq_length):
        """計算每個專家的容量"""
        tokens_per_expert = batch_size * seq_length / self.num_routed_experts
        capacity = int(tokens_per_expert * self.expert_capacity_factor)
        return capacity
        
    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # 1. 路由計算
        routing_weights, top_k_indices, z_loss = self._compute_routing_probabilities(
            hidden_states
        )
        
        # 2. 計算專家容量
        expert_capacity = self._compute_expert_capacity(batch_size, seq_length)
        
        # 3. 初始化輸出張量
        final_output = torch.zeros_like(hidden_states)
        
        # 4. 處理路由專家
        for i in range(self.top_k):
            expert_idx = top_k_indices[..., i]  # [batch_size, seq_length]
            token_weight = routing_weights[..., i]  # [batch_size, seq_length]
            
            # 為每個專家收集輸入
            for j in range(self.num_routed_experts):
                # 創建專家遮罩 [batch_size, seq_length]
                expert_mask = (expert_idx == j)
                if not expert_mask.any():
                    continue
                
                # 獲取需要處理的位置
                batch_indices, seq_indices = torch.where(expert_mask)
                
                # 收集需要處理的輸入
                expert_input = hidden_states[batch_indices, seq_indices]
                
                if len(expert_input) > expert_capacity:
                    # 如果超過容量，隨機選擇tokens
                    perm = torch.randperm(len(expert_input))[:expert_capacity]
                    expert_input = expert_input[perm]
                    batch_indices = batch_indices[perm]
                    seq_indices = seq_indices[perm]
                
                # 專家處理
                expert_output = self.routed_experts[j](expert_input)
                
                # 獲取對應的權重
                current_token_weight = token_weight[batch_indices, seq_indices].unsqueeze(-1)
                
                # 應用權重並更新最終輸出
                final_output[batch_indices, seq_indices] += current_token_weight * expert_output
        
        # 5. 處理共享專家
        shared_weight = 1.0 / self.num_shared_experts
        for expert in self.shared_experts:
            final_output += shared_weight * expert(hidden_states)
        
        # 6. 最終輸出處理
        output = self.output(final_output)
        output = self.dropout(output)
        
        # 7. 殘差連接和正規化
        output = output * 0.5 + hidden_states * 0.5
        output = F.layer_norm(
            output,
            [output.size(-1)],
            eps=1e-6
        )
        
        return output, z_loss

class NSAConfig:
    def __init__(
        self,
        vocab_size: int = 32000,
        max_seq_length: int = 512,
        hidden_size: int = 768,
        num_attention_heads: int = 16,
        num_hidden_layers: int = 8,
        compress_ratio: int = 4,
        select_k: int = 16,
        window_size: int = 64,
        num_routed_experts: int = 4,
        num_shared_experts: int = 2,
        expert_capacity_factor: float = 1.25,
        router_z_loss_coef: float = 0.001,
        top_k: int = 2,
        # 【新增】动态稀疏度参数
        use_dynamic_sparsity: bool = False,  # 是否启用动态稀疏度
        min_select_k: int = 8,               # 最小选择数量
        max_select_k: int = 32,              # 最大选择数量
        min_compression_ratio: int = 2,      # 最小压缩比
        max_compression_ratio: int = 8,      # 最大压缩比
        entropy_factor: float = 0.5          # 熵调整因子
    ):
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers
        self.compress_ratio = compress_ratio
        self.select_k = select_k
        self.window_size = window_size
        
        # MoE相關配置
        self.num_routed_experts = num_routed_experts
        self.num_shared_experts = num_shared_experts
        self.expert_capacity_factor = expert_capacity_factor
        self.router_z_loss_coef = router_z_loss_coef
        self.top_k = top_k
        
        # 【新增】动态稀疏度控制参数
        self.use_dynamic_sparsity = use_dynamic_sparsity
        self.min_select_k = min_select_k
        self.max_select_k = max_select_k
        self.min_compression_ratio = min_compression_ratio
        self.max_compression_ratio = max_compression_ratio
        self.entropy_factor = entropy_factor

        print(f"vocab_size: {vocab_size}")
        print(f"max_seq_length: {max_seq_length}")
        print(f"hidden_size: {hidden_size}")
        print(f"num_attention_heads: {num_attention_heads}")
        print(f"vnum_hidden_layers: {num_hidden_layers}")

class NSAAttentionExtended(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.head_size = self.hidden_size // self.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.head_size
        self.scale = 1.0 / math.sqrt(self.head_size)
        
        # Query, Key, Value projections
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # Output projection
        self.output = nn.Linear(self.hidden_size , self.hidden_size)
        
        # Branch gates
        self.branch_gate = nn.Linear(self.hidden_size, 3)
        
        # Dropouts
        self.attention_dropout = nn.Dropout(0.1)
        self.output_dropout = nn.Dropout(0.1)
        
        # Additional parameters
        self.compress_ratio = config.compress_ratio
        self.select_k = config.select_k
        self.window_size = config.window_size
        
        # Compression layer
        self.compress = nn.Linear(config.hidden_size * config.compress_ratio, config.hidden_size)
        
        # Selection layer
        self.selection_score = nn.Linear(self.hidden_size, 1)
        
        # 【新增】动态稀疏度控制器
        self.use_dynamic_sparsity = getattr(config, 'use_dynamic_sparsity', False)
        if self.use_dynamic_sparsity:
            self.sparsity_controller = DynamicSparsityController(
                base_select_k=self.select_k,
                min_select_k=getattr(config, 'min_select_k', 8),
                max_select_k=getattr(config, 'max_select_k', 32),
                base_compression_ratio=self.compress_ratio,
                min_compression_ratio=getattr(config, 'min_compression_ratio', 2),
                max_compression_ratio=getattr(config, 'max_compression_ratio', 8),
                entropy_factor=getattr(config, 'entropy_factor', 0.5)
            )
            # 记录当前使用的动态参数
            self.current_select_k = self.select_k
            self.current_compress_ratio = self.compress_ratio
            self.sparsity_history = {"select_k": [], "compress_ratio": []}

    def _adjust_tensor_size(self, tensor, target_size, dim=1):
        """调整张量大小以匹配目标大小"""
        current_size = tensor.size(dim)
        if current_size == target_size:
            return tensor
            
        if current_size < target_size:
            # 需要填充
            pad_size = target_size - current_size
            padding = torch.zeros_like(tensor.narrow(dim, 0, 1)).repeat_interleave(pad_size, dim=dim)
            return torch.cat([tensor, padding], dim=dim)
        else:
            # 需要裁剪
            return tensor.narrow(dim, 0, target_size)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def compress_attention(self, hidden_states, attention_mask=None):
        """压缩注意力机制，支持真正的动态压缩比"""
        batch_size, seq_length, _ = hidden_states.size()
        
        # 获取初始化时的压缩比
        init_compress_ratio = self.config.compress_ratio
        expected_input_size = self.hidden_size * init_compress_ratio  # 线性层期望的输入维度
        
        # 动态调整压缩比
        compress_ratio = self.compress_ratio
        if self.use_dynamic_sparsity and hasattr(self, 'current_attention_scores'):
            _, dynamic_compress_ratio = self.sparsity_controller.compute_dynamic_sparsity(
                hidden_states, self.current_attention_scores
            )
            compress_ratio = dynamic_compress_ratio
            self.current_compress_ratio = compress_ratio
            self.sparsity_history["compress_ratio"].append(compress_ratio)
            
        # 调整序列长度为compress_ratio的倍数
        pad_length = (compress_ratio - seq_length % compress_ratio) % compress_ratio
        if pad_length > 0:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_length))
            if attention_mask is not None:
                attention_mask = nn.functional.pad(attention_mask, (0, pad_length))
        
        # 重塑为blocks
        new_seq_length = hidden_states.size(1)
        blocks = hidden_states.view(batch_size, -1, compress_ratio, self.hidden_size)
        blocks = blocks.reshape(batch_size, -1, compress_ratio * self.hidden_size)
        
        # 调整blocks的最后一个维度以匹配self.compress期望的输入维度
        current_input_size = compress_ratio * self.hidden_size  # 当前的输入维度
        
        if current_input_size != expected_input_size:
            # 如果维度不匹配，调整blocks的最后一个维度
            if current_input_size < expected_input_size:
                # 需要填充
                pad_size = expected_input_size - current_input_size
                padding = torch.zeros(batch_size, blocks.size(1), pad_size, device=blocks.device)
                blocks = torch.cat([blocks, padding], dim=2)
            else:
                # 需要裁剪
                blocks = blocks[:, :, :expected_input_size]
        
        # 压缩blocks
        compressed = self.compress(blocks)
        
        # 计算自注意力
        query = self.query(compressed)
        key = self.key(compressed)
        value = self.value(compressed)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        # 存储注意力分数用于动态稀疏度调整
        self.current_attention_scores = attention_scores
        
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        
        # 调整输出大小以匹配原始序列长度
        context = self._adjust_tensor_size(context, seq_length)
        
        return context

    def select_attention(self, hidden_states, attention_mask=None):
        """选择性注意力机制，支持动态选择数量"""
        batch_size, seq_length, _ = hidden_states.size()
        
        # 【新增】动态调整select_k
        effective_k = min(self.select_k, seq_length)
        if self.use_dynamic_sparsity and hasattr(self, 'current_attention_scores'):
            dynamic_k, _ = self.sparsity_controller.compute_dynamic_sparsity(
                hidden_states, self.current_attention_scores
            )
            effective_k = min(dynamic_k, seq_length)
            self.current_select_k = effective_k
            self.sparsity_history["select_k"].append(effective_k)
        
        # 计算选择分数
        scores = self.selection_score(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(~attention_mask.bool(), float('-inf'))
        
        # 选择top-k位置
        _, indices = torch.topk(scores, k=effective_k, dim=-1)
        indices = indices.sort(dim=-1)[0]
        
        # 收集选定的状态
        selected = torch.gather(
            hidden_states,
            dim=1,
            index=indices.unsqueeze(-1).expand(-1, -1, self.hidden_size)
        )
        
        # 计算自注意力
        query = self.query(selected)
        key = self.key(selected)
        value = self.value(selected)
        
        attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.attention_dropout(attention_probs)
        
        context = torch.matmul(attention_probs, value)
        
        # 调整输出大小以匹配原始序列长度
        context = self._adjust_tensor_size(context, seq_length)
        
        return context

    def window_attention(self, hidden_states, attention_mask=None):
        """滑动窗口注意力机制"""
        batch_size, seq_length, _ = hidden_states.size()
        
        # 计算有效窗口大小
        effective_window = min(self.window_size, seq_length)
        
        # 添加填充
        pad_length = (effective_window - seq_length % effective_window) % effective_window
        if pad_length > 0:
            hidden_states = nn.functional.pad(hidden_states, (0, 0, 0, pad_length))
            if attention_mask is not None:
                attention_mask = nn.functional.pad(attention_mask, (0, pad_length))
        
        # 创建滑动窗口
        windows = []
        for i in range(0, hidden_states.size(1) - effective_window + 1, effective_window // 2):
            windows.append(hidden_states[:, i:i + effective_window])
        
        # 处理每个窗口
        window_outputs = []
        for window in windows:
            # 计算自注意力
            query = self.query(window)
            key = self.key(window)
            value = self.value(window)
            
            attention_scores = torch.matmul(query, key.transpose(-1, -2)) * self.scale
            attention_probs = nn.functional.softmax(attention_scores, dim=-1)
            attention_probs = self.attention_dropout(attention_probs)
            
            context = torch.matmul(attention_probs, value)
            window_outputs.append(context)
        
        # 合并窗口输出
        output = torch.cat(window_outputs, dim=1)
        
        # 调整输出大小以匹配原始序列长度
        output = self._adjust_tensor_size(output, seq_length)
        
        return output

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_length, _ = hidden_states.size()
        
        # 计算不同注意力分支的输出
        compress_output = self.compress_attention(hidden_states, attention_mask)
        select_output = self.select_attention(hidden_states, attention_mask)
        window_output = self.window_attention(hidden_states, attention_mask)
        
        # 计算门控权重
        avg_features = hidden_states.mean(dim=1, keepdim=True)
        gate_logits = self.branch_gate(avg_features)
        gate_weights = F.softmax(gate_logits, dim=-1)
        
        # 合并输出
        outputs = [compress_output, select_output, window_output]
        combined_output = torch.zeros_like(hidden_states)
        for i, output in enumerate(outputs):
            combined_output += gate_weights[:, :, i:i+1] * output
        
        # 最终输出处理
        final_output = self.output(combined_output)
        final_output = self.output_dropout(final_output)
        
        # 残差连接和正规化
        final_output = final_output * 0.5 + hidden_states * 0.5
        final_output = F.layer_norm(
            final_output,
            [final_output.size(-1)],
            eps=1e-6
        )
        
        return final_output

class NSABlockExtended(nn.Module):
    def __init__(self, config):
        super().__init__()
        # 默认使用带路由的注意力机制
        self.attention = NSAAttentionExtendedWithRouting(config)
        self.has_z_loss = True  # 默认有z_loss
        
        self.intermediate = nn.Linear(config.hidden_size, config.hidden_size * 4)
        self.output = nn.Linear(config.hidden_size * 4, config.hidden_size)
        self.layernorm1 = nn.LayerNorm(config.hidden_size)
        self.layernorm2 = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.activation = nn.GELU()
        
        # 儲存配置以獲取z_loss係數
        self.config = config
        
    def forward(self, hidden_states, attention_mask=None):
        # 基于attention类型处理
        if self.has_z_loss:
            # 获取attention输出和z_loss
            attention_output, z_loss = self.attention(hidden_states, attention_mask)
            hidden_states = self.layernorm1(hidden_states + attention_output)
            
            intermediate_output = self.intermediate(hidden_states)
            intermediate_output = self.activation(intermediate_output)
            
            layer_output = self.output(intermediate_output)
            layer_output = self.dropout(layer_output)
            
            output = self.layernorm2(hidden_states + layer_output)
            
            # 返回输出和z_loss
            return output, z_loss
        else:
            # 处理无z_loss的情况
            attention_output = self.attention(hidden_states, attention_mask)
            hidden_states = self.layernorm1(hidden_states + attention_output)
            
            intermediate_output = self.intermediate(hidden_states)
            intermediate_output = self.activation(intermediate_output)
            
            layer_output = self.output(intermediate_output)
            layer_output = self.dropout(layer_output)
            
            output = self.layernorm2(hidden_states + layer_output)
            
            return output

class NSAModel(nn.Module):
    def __init__(self, config: NSAConfig):
        super().__init__()
        self.config = config
        
        # 原有的初始化代碼
        self.embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_seq_length,
            config.hidden_size
        )
        
        # 初始化embeddings
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.position_embeddings.weight, mean=0.0, std=0.02)
        
        # 【修改】根据配置选择使用标准注意力或带路由的注意力
        self.layers = nn.ModuleList()
        for _ in range(config.num_hidden_layers):
            if getattr(config, 'use_dynamic_sparsity', False):
                # 使用标准NSA和动态稀疏度
                block = NSABlockExtended(config)
                block.attention = NSAAttentionExtended(config)
                block.has_z_loss = False
                self.layers.append(block)
            else:
                # 使用带路由的NSA
                self.layers.append(NSABlockExtended(config))
        
        self.layernorm = nn.LayerNorm(config.hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size)
        
        # 初始化lm_head
        nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.lm_head.bias)
        
        # 【新增】动态稀疏度指标收集
        self.use_dynamic_sparsity = getattr(config, 'use_dynamic_sparsity', False)
        self.dynamic_sparsity_metrics = {"compression_ratios": [], "select_ks": [], "steps": []}
        self.step_counter = 0
        
    # 【新增】记录动态稀疏度指标
    def record_sparsity_metrics(self, layer_idx):
        """收集并记录动态稀疏度参数"""
        if not self.use_dynamic_sparsity:
            return
            
        layer = self.layers[layer_idx]
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'current_select_k'):
            self.dynamic_sparsity_metrics["select_ks"].append(layer.attention.current_select_k)
            
        if hasattr(layer, 'attention') and hasattr(layer.attention, 'current_compress_ratio'):
            self.dynamic_sparsity_metrics["compression_ratios"].append(layer.attention.current_compress_ratio)
            
        self.dynamic_sparsity_metrics["steps"].append(self.step_counter)
        self.step_counter += 1
            
    # 【新增】获取动态稀疏度指标
    def get_sparsity_metrics(self):
        """返回当前动态稀疏度指标的摘要统计"""
        if not self.use_dynamic_sparsity:
            return {"dynamic_sparsity_enabled": False}
            
        compr_ratios = self.dynamic_sparsity_metrics["compression_ratios"]
        select_ks = self.dynamic_sparsity_metrics["select_ks"]
        
        if not compr_ratios:
            return {"dynamic_sparsity_enabled": True, "no_records": True}
            
        return {
            "dynamic_sparsity_enabled": True,
            "compression_ratio": {
                "min": min(compr_ratios) if compr_ratios else None,
                "max": max(compr_ratios) if compr_ratios else None,
                "avg": sum(compr_ratios) / len(compr_ratios) if compr_ratios else None
            },
            "select_k": {
                "min": min(select_ks) if select_ks else None,
                "max": max(select_ks) if select_ks else None,
                "avg": sum(select_ks) / len(select_ks) if select_ks else None
            },
            "history": {
                "steps": self.dynamic_sparsity_metrics["steps"][-100:],
                "compression_ratios": self.dynamic_sparsity_metrics["compression_ratios"][-100:],
                "select_ks": self.dynamic_sparsity_metrics["select_ks"][-100:]
            }
        }
        
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # 獲取embeddings並加入梯度裁剪
        hidden_states = self.embeddings(input_ids)
        hidden_states = torch.clamp(hidden_states, min=-100, max=100)
        
        # 添加位置embeddings
        position_ids = torch.arange(
            input_ids.size(1), 
            dtype=torch.long, 
            device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        position_embeddings = torch.clamp(
            position_embeddings, 
            min=-100, 
            max=100
        )
        
        # 合併embeddings
        hidden_states = hidden_states + position_embeddings
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.layernorm(hidden_states)
        
        # 追蹤總的z_loss
        total_z_loss = 0.0
        
        # 通過transformer層處理
        for i, layer in enumerate(self.layers):
            # 检查该层是否有z_loss
            if hasattr(layer, 'has_z_loss') and layer.has_z_loss:
                layer_output, z_loss = layer(hidden_states, attention_mask)
                # 累積z_loss
                total_z_loss += z_loss
            else:
                layer_output = layer(hidden_states, attention_mask)
                
            # 添加縮放的殘差連接
            hidden_states = hidden_states * 0.5 + layer_output * 0.5
            # 裁剪值以防止爆炸
            hidden_states = torch.clamp(hidden_states, min=-100, max=100)
            
            # 如果开启了动态稀疏度，记录相关指标
            if self.use_dynamic_sparsity and i == 0:  # 只记录第一层
                self.record_sparsity_metrics(i)
        
        # 生成logits並小心縮放
        prediction_scores = self.lm_head(hidden_states)
        prediction_scores = prediction_scores / math.sqrt(self.config.hidden_size)
        prediction_scores = torch.clamp(prediction_scores, min=-100, max=100)
        
        # 如果提供了標籤，計算損失
        if labels is not None:
            # 主要的交叉熵損失
            loss_fct = nn.CrossEntropyLoss()
            ce_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), 
                labels.view(-1)
            )
            
            # 如果有z_loss，添加加權的z_loss到總損失
            if total_z_loss > 0:
                z_loss_weight = getattr(self.config, 'router_z_loss_coef', 0.001)
                avg_z_loss = total_z_loss / len(self.layers)
                total_loss = ce_loss + z_loss_weight * avg_z_loss
            else:
                total_loss = ce_loss
            
            return total_loss
            
        return prediction_scores
    
class ChineseTextDataset(Dataset):
    def __init__(
        self,
        data,  # 可以是文件路徑或數據列表
        tokenizer: BertTokenizer,
        max_length: int = 512
    ):
        if isinstance(data, str):
            # 如果是文件路徑，從文件讀取數據
            with open(data, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        else:
            # 如果是數據列表，直接使用
            self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        
    def __len__(self) -> int:
        return len(self.data)
        
    def __getitem__(self, idx: int) -> dict:
        text = self.data[idx]["text"]
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": encoding["input_ids"].squeeze()
        }

class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_dataset: Dataset,
        test_dataset: Optional[Dataset] = None,
        batch_size: int = 8,
        learning_rate: float = 5e-5,
        num_epochs: int = 3,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.model = model.to(device)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        self.test_loader = None
        if test_dataset:
            self.test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False
            )
        
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate
        )
        self.num_epochs = num_epochs
        self.device = device
        
    def train(self):
        self.model.train()
        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch_idx, batch in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Epoch {epoch+1}/{self.num_epochs}, Batch {batch_idx+1}/{len(self.train_loader)}, Loss: {loss.item():.12f}")
                
            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Average Loss: {avg_loss:.12f}")
            
            if self.test_loader:
                self.evaluate()
                
    def evaluate(self):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for batch in self.test_loader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                loss = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += loss.item()
                
        avg_loss = total_loss / len(self.test_loader)
        print(f"Validation Loss: {avg_loss:.12f}")
        
    def save_model(self, path: str):
        torch.save(self.model.state_dict(), path)
        
    def load_model(self, path: str):
        self.model.load_state_dict(torch.load(path))

class SparsityMonitor:
    """实时监控NSA模型动态稀疏度参数的变化"""
    
    def __init__(self, model_name="NSA模型", log_file="sparsity_metrics.json"):
        self.model_name = model_name
        self.log_file = log_file
        self.is_running = False
        
        # 历史数据
        self.steps = []
        self.compression_ratios = []
        self.select_ks = []
        
        # 初始化UI
        self.setup_ui()
        
    def setup_ui(self):
        """初始化监控器界面"""
        self.root = tk.Tk()
        self.root.title(f"{self.model_name} - 动态稀疏度监控器")
        self.root.geometry("900x600")
        
        # 控制面板
        control_frame = ttk.Frame(self.root, padding=10)
        control_frame.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="监控间隔 (秒):").pack(side=tk.LEFT, padx=5)
        self.interval_var = tk.StringVar(value="1.0")
        interval_entry = ttk.Entry(control_frame, textvariable=self.interval_var, width=5)
        interval_entry.pack(side=tk.LEFT, padx=5)
        
        self.start_button = ttk.Button(control_frame, text="开始监控", command=self.start_monitoring)
        self.start_button.pack(side=tk.LEFT, padx=15)
        
        self.stop_button = ttk.Button(control_frame, text="停止监控", command=self.stop_monitoring, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # 创建状态栏
        self.status_var = tk.StringVar(value="就绪")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # 当前值显示区域
        current_frame = ttk.LabelFrame(self.root, text="当前稀疏度参数", padding=10)
        current_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(current_frame, text="压缩比 (Compression Ratio):").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.current_cr_var = tk.StringVar(value="N/A")
        ttk.Label(current_frame, textvariable=self.current_cr_var, width=10).grid(row=0, column=1, sticky=tk.W, padx=5, pady=3)
        
        ttk.Label(current_frame, text="选择数量 (Select K):").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.current_sk_var = tk.StringVar(value="N/A")
        ttk.Label(current_frame, textvariable=self.current_sk_var, width=10).grid(row=1, column=1, sticky=tk.W, padx=5, pady=3)
        
        # 统计信息区域
        stats_frame = ttk.LabelFrame(current_frame, text="统计信息", padding=10)
        stats_frame.grid(row=0, column=2, rowspan=2, sticky=tk.W, padx=20)
        
        ttk.Label(stats_frame, text="压缩比 - 最小值:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=2)
        self.min_cr_var = tk.StringVar(value="N/A")
        ttk.Label(stats_frame, textvariable=self.min_cr_var, width=5).grid(row=0, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(stats_frame, text="压缩比 - 最大值:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=2)
        self.max_cr_var = tk.StringVar(value="N/A")
        ttk.Label(stats_frame, textvariable=self.max_cr_var, width=5).grid(row=1, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(stats_frame, text="压缩比 - 平均值:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=2)
        self.avg_cr_var = tk.StringVar(value="N/A")
        ttk.Label(stats_frame, textvariable=self.avg_cr_var, width=5).grid(row=2, column=1, sticky=tk.W, pady=2)
        
        ttk.Label(stats_frame, text="选择数 - 最小值:").grid(row=0, column=2, sticky=tk.W, padx=15, pady=2)
        self.min_sk_var = tk.StringVar(value="N/A")
        ttk.Label(stats_frame, textvariable=self.min_sk_var, width=5).grid(row=0, column=3, sticky=tk.W, pady=2)
        
        ttk.Label(stats_frame, text="选择数 - 最大值:").grid(row=1, column=2, sticky=tk.W, padx=15, pady=2)
        self.max_sk_var = tk.StringVar(value="N/A")
        ttk.Label(stats_frame, textvariable=self.max_sk_var, width=5).grid(row=1, column=3, sticky=tk.W, pady=2)
        
        ttk.Label(stats_frame, text="选择数 - 平均值:").grid(row=2, column=2, sticky=tk.W, padx=15, pady=2)
        self.avg_sk_var = tk.StringVar(value="N/A")
        ttk.Label(stats_frame, textvariable=self.avg_sk_var, width=5).grid(row=2, column=3, sticky=tk.W, pady=2)
        
        # 图表区域
        chart_frame = ttk.Frame(self.root)
        chart_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # 使用matplotlib创建图表
        self.fig, self.ax = plt.subplots(2, 1, figsize=(10, 6), dpi=80)
        
        # 压缩比图表
        self.compression_line, = self.ax[0].plot([], [], 'b-', label='压缩比')
        self.ax[0].set_title('动态压缩比变化')
        self.ax[0].set_xlabel('步骤')
        self.ax[0].set_ylabel('压缩比')
        self.ax[0].set_ylim(1, 9)
        self.ax[0].grid(True)
        self.ax[0].legend()
        
        # 选择数图表
        self.select_k_line, = self.ax[1].plot([], [], 'r-', label='选择数量')
        self.ax[1].set_title('动态选择数量变化')
        self.ax[1].set_xlabel('步骤')
        self.ax[1].set_ylabel('选择数量')
        self.ax[1].set_ylim(7, 33)
        self.ax[1].grid(True)
        self.ax[1].legend()
        
        self.fig.tight_layout()
        
        # 将图表嵌入到tkinter窗口中
        self.canvas = FigureCanvasTkAgg(self.fig, master=chart_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # 设置窗口关闭事件
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def start_monitoring(self):
        """开始监控动态稀疏度参数"""
        try:
            interval = float(self.interval_var.get())
            if interval <= 0:
                raise ValueError("监控间隔必须大于0")
                
            self.is_running = True
            self.start_button.config(state=tk.DISABLED)
            self.stop_button.config(state=tk.NORMAL)
            
            # 创建并启动监控线程
            self.monitor_thread = threading.Thread(target=self.monitor_loop, args=(interval,))
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            self.status_var.set("监控中...")
        except ValueError as e:
            self.status_var.set(f"错误: {str(e)}")
    
    def stop_monitoring(self):
        """停止监控"""
        self.is_running = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.status_var.set("监控已停止")
    
    def on_close(self):
        """窗口关闭事件处理"""
        self.is_running = False
        self.root.destroy()
    
    def monitor_loop(self, interval):
        """监控循环，定期读取参数变化"""
        while self.is_running:
            try:
                self.load_metrics()
                self.update_ui()
                time.sleep(interval)
            except Exception as e:
                self.status_var.set(f"监控错误: {str(e)}")
                time.sleep(interval)
    
    def load_metrics(self):
        """从日志文件加载指标数据"""
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, 'r') as f:
                    data = json.load(f)
                    
                if data.get("dynamic_sparsity_enabled", False):
                    history = data.get("history", {})
                    
                    # 更新历史数据
                    self.steps = history.get("steps", [])
                    self.compression_ratios = history.get("compression_ratios", [])
                    self.select_ks = history.get("select_ks", [])
                    
                    # 更新当前值
                    current_cr = self.compression_ratios[-1] if self.compression_ratios else None
                    current_sk = self.select_ks[-1] if self.select_ks else None
                    
                    self.current_cr_var.set(str(current_cr) if current_cr is not None else "N/A")
                    self.current_sk_var.set(str(current_sk) if current_sk is not None else "N/A")
                    
                    # 更新统计信息
                    cr_stats = data.get("compression_ratio", {})
                    sk_stats = data.get("select_k", {})
                    
                    self.min_cr_var.set(str(cr_stats.get("min", "N/A")))
                    self.max_cr_var.set(str(cr_stats.get("max", "N/A")))
                    self.avg_cr_var.set(f"{cr_stats.get('avg', 0):.2f}" if cr_stats.get("avg") else "N/A")
                    
                    self.min_sk_var.set(str(sk_stats.get("min", "N/A")))
                    self.max_sk_var.set(str(sk_stats.get("max", "N/A")))
                    self.avg_sk_var.set(f"{sk_stats.get('avg', 0):.2f}" if sk_stats.get("avg") else "N/A")
                    
                    self.status_var.set(f"已更新 | 当前步骤: {self.steps[-1] if self.steps else 'N/A'}")
                else:
                    self.status_var.set("动态稀疏度未启用")
        except Exception as e:
            self.status_var.set(f"加载指标失败: {str(e)}")
    
    def update_ui(self):
        """更新UI显示"""
        # 更新图表数据
        if self.steps and self.compression_ratios:
            self.compression_line.set_data(self.steps, self.compression_ratios)
            self.ax[0].relim()  # 重新计算轴范围
            self.ax[0].autoscale_view()  # 自动缩放视图
            
        if self.steps and self.select_ks:
            self.select_k_line.set_data(self.steps, self.select_ks)
            self.ax[1].relim()  # 重新计算轴范围
            self.ax[1].autoscale_view()  # 自动缩放视图
            
        # 保持X轴一致
        if self.steps:
            xlim = (min(self.steps) if self.steps else 0, 
                   max(self.steps) if self.steps else 10)
            self.ax[0].set_xlim(xlim)
            self.ax[1].set_xlim(xlim)
            
        # 重新绘制图表
        self.canvas.draw_idle()
    
    def run(self):
        """启动监控器"""
        self.root.mainloop()


# 模拟测试
# 创建模拟数据生成器
class DummyModel:
    def __init__(self):
        self.step = 0
        self.base_cr = 4
        self.base_sk = 16
            
    def get_sparsity_metrics(self):
        self.step += 1
            
        # 模拟压缩比在2-8之间波动
        cr_values = []
        sk_values = []
        steps = []
            
        for i in range(max(0, self.step - 100), self.step + 1):
            noise1 = np.sin(i/10) * 2 + np.random.normal(0, 0.5)
            cr = max(2, min(8, self.base_cr + noise1))
            cr_values.append(cr)
                
            noise2 = np.cos(i/8) * 6 + np.random.normal(0, 1)
            sk = max(8, min(32, self.base_sk + noise2))
            sk_values.append(int(sk))
                
            steps.append(i)
                
        return {
            "dynamic_sparsity_enabled": True,
            "compression_ratio": {
                "min": min(cr_values),
                "max": max(cr_values), 
                "avg": sum(cr_values) / len(cr_values)
            },
            "select_k": {
                "min": min(sk_values),
                "max": max(sk_values),
                "avg": sum(sk_values) / len(sk_values)
            },
            "history": {
                "steps": steps,
                "compression_ratios": cr_values,
                "select_ks": sk_values
            }
        }


# 数据记录器类
class SparsityLogger:
    """将动态稀疏度参数记录到文件"""
    
    def __init__(self, model, log_file="sparsity_metrics.json"):
        self.model = model
        self.log_file = log_file
        
    def log_metrics(self):
        """记录当前指标到文件"""
        if hasattr(self.model, 'get_sparsity_metrics'):
            metrics = self.model.get_sparsity_metrics()
            with open(self.log_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            return True
        return False


class TrainingVisualizer:
    """
    分析和可視化訓練記錄文件的工具
    """
    def __init__(self, log_dir="training_logs"):
        self.log_dir = log_dir
        self.data = None
        self.log_files = []
    
    def find_log_files(self, pattern="*.json"):
        """尋找所有符合模式的日誌文件"""
        self.log_files = glob.glob(os.path.join(self.log_dir, pattern))
        return self.log_files
    
    def load_log(self, log_file):
        """載入指定的日誌文件"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"成功載入日誌文件: {log_file}")
            return True
        except Exception as e:
            print(f"載入日誌文件時發生錯誤: {e}")
            return False
    
    def plot_training_loss(self, save_path=None, show=True):
        """繪製訓練和驗證損失曲線"""
        if not self.data or not self.data.get("epochs"):
            print("沒有足夠的數據來繪製損失曲線")
            return
        
        plt.figure(figsize=(12, 6))
        
        # 訓練損失
        epochs = self.data["epochs"]
        train_losses = self.data["train_losses"]
        plt.plot(epochs, train_losses, 'b-', marker='o', label='訓練損失', alpha=0.7)
        
        # 計算並繪製平滑線
        if len(train_losses) > 10:
            window_size = min(5, len(train_losses) // 5)
            smoothed = np.convolve(train_losses, np.ones(window_size)/window_size, mode='valid')
            valid_epochs = epochs[window_size-1:]
            if len(valid_epochs) == len(smoothed):
                plt.plot(valid_epochs, smoothed, 'b-', linewidth=2, label='平滑訓練損失')
        
        # 驗證損失
        if "val_losses" in self.data and self.data["val_losses"]:
            val_losses = self.data["val_losses"]
            plt.plot(epochs, val_losses, 'r-', marker='x', label='驗證損失', alpha=0.7)
            
            # 計算並繪製平滑線
            if len(val_losses) > 10:
                window_size = min(5, len(val_losses) // 5)
                smoothed = np.convolve(val_losses, np.ones(window_size)/window_size, mode='valid')
                valid_epochs = epochs[window_size-1:]
                if len(valid_epochs) == len(smoothed):
                    plt.plot(valid_epochs, smoothed, 'r-', linewidth=2, label='平滑驗證損失')
        
        plt.title(f'訓練與驗證損失 ({self.data.get("model_name", "未命名模型")})')
        plt.xlabel('Epoch')
        plt.ylabel('損失')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        # 設置x軸為整數
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        # 標註最佳點
        if "val_losses" in self.data and self.data["val_losses"]:
            best_idx = np.argmin(self.data["val_losses"])
            best_epoch = self.data["epochs"][best_idx]
            best_val_loss = self.data["val_losses"][best_idx]
            plt.annotate(f'最佳: {best_val_loss:.6f}',
                        xy=(best_epoch, best_val_loss),
                        xytext=(best_epoch, best_val_loss*1.1),
                        arrowprops=dict(facecolor='black', shrink=0.05, width=1.5, headwidth=8),
                        horizontalalignment='center', backgroundcolor='white')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"損失曲線已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_learning_rate(self, save_path=None, show=True):
        """繪製學習率變化曲線"""
        if not self.data or not self.data.get("learning_rates"):
            print("沒有學習率數據可用")
            return
        
        plt.figure(figsize=(12, 4))
        
        epochs = self.data["epochs"]
        lr_values = self.data["learning_rates"]
        
        plt.plot(epochs, lr_values, 'g-', marker='o', linewidth=2)
        plt.title(f'學習率變化 ({self.data.get("model_name", "未命名模型")})')
        plt.xlabel('Epoch')
        plt.ylabel('學習率')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 對數刻度更適合顯示學習率
        plt.yscale('log')
        
        # 設置x軸為整數
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"學習率曲線已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_dynamic_sparsity(self, save_path=None, show=True):
        """繪製動態稀疏度參數變化曲線"""
        if not self.data or not self.data.get("dynamic_sparsity", {}).get("enabled", False):
            print("沒有動態稀疏度數據可用或未啟用")
            return
        
        steps = self.data["dynamic_sparsity"]["steps"]
        if not steps:
            print("沒有步驟數據可用")
            return
        
        compression_ratios = self.data["dynamic_sparsity"]["compression_ratios"]
        select_ks = self.data["dynamic_sparsity"]["select_ks"]
        
        plt.figure(figsize=(12, 8))
        
        # 上圖: 壓縮比
        plt.subplot(2, 1, 1)
        plt.plot(steps, compression_ratios, 'b-', alpha=0.6)
        
        # 添加平滑線
        if len(compression_ratios) > 20:
            window_size = min(20, len(compression_ratios) // 10)
            smoothed = np.convolve(compression_ratios, np.ones(window_size)/window_size, mode='valid')
            valid_steps = steps[window_size-1:]
            if len(valid_steps) == len(smoothed):
                plt.plot(valid_steps, smoothed, 'b-', linewidth=2)
        
        plt.title(f'動態壓縮比變化 ({self.data.get("model_name", "未命名模型")})')
        plt.xlabel('步驟')
        plt.ylabel('壓縮比')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        # 下圖: 選擇數量
        plt.subplot(2, 1, 2)
        plt.plot(steps, select_ks, 'r-', alpha=0.6)
        
        # 添加平滑線
        if len(select_ks) > 20:
            window_size = min(20, len(select_ks) // 10)
            smoothed = np.convolve(select_ks, np.ones(window_size)/window_size, mode='valid')
            valid_steps = steps[window_size-1:]
            if len(valid_steps) == len(smoothed):
                plt.plot(valid_steps, smoothed, 'r-', linewidth=2)
        
        plt.title('動態選擇數量變化')
        plt.xlabel('步驟')
        plt.ylabel('選擇數量')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"動態稀疏度曲線已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_gpu_memory(self, save_path=None, show=True):
        """繪製GPU記憶體使用曲線"""
        if not self.data or not self.data.get("gpu_memory"):
            print("沒有GPU記憶體使用數據可用")
            return
        
        plt.figure(figsize=(12, 4))
        
        epochs = self.data["epochs"]
        gpu_mem = self.data["gpu_memory"]
        
        # 檢查GPU記憶體數據的格式
        if isinstance(gpu_mem[0], dict):
            # 繪製每個GPU的記憶體使用
            for gpu_id in gpu_mem[0].keys():
                mem_usage = [entry.get(gpu_id, 0) for entry in gpu_mem]
                plt.plot(epochs, mem_usage, marker='o', label=f'GPU {gpu_id}', alpha=0.7)
        else:
            # 繪製單一線條
            plt.plot(epochs, gpu_mem, 'c-', marker='o', linewidth=2)
        
        plt.title(f'GPU記憶體使用 ({self.data.get("model_name", "未命名模型")})')
        plt.xlabel('Epoch')
        plt.ylabel('使用記憶體 (GB)')
        plt.grid(True, linestyle='--', alpha=0.7)
        
        if isinstance(gpu_mem[0], dict) and len(gpu_mem[0]) > 1:
            plt.legend()
        
        # 設置x軸為整數
        plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"GPU記憶體使用曲線已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_batch_loss(self, save_path=None, show=True):
        """繪製批次訓練損失曲線"""
        if not self.data or not self.data.get("batches", {}).get("indices"):
            print("沒有批次損失數據可用")
            return
        
        plt.figure(figsize=(12, 4))
        
        indices = self.data["batches"]["indices"]
        losses = self.data["batches"]["losses"]
        
        plt.plot(indices, losses, 'b-', alpha=0.3)
        
        # 添加平滑線
        if len(losses) > 20:
            window_size = min(50, len(losses) // 20)
            smoothed = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
            valid_indices = indices[window_size-1:]
            if len(valid_indices) == len(smoothed):
                plt.plot(valid_indices, smoothed, 'r-', linewidth=2, label=f'移動平均 (窗口={window_size})')
        
        plt.title(f'批次訓練損失 ({self.data.get("model_name", "未命名模型")})')
        plt.xlabel('批次索引')
        plt.ylabel('損失')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"批次損失曲線已保存到: {save_path}")
        
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_all(self, output_dir=None, show=False):
        """繪製所有可用的圖表並保存"""
        if not self.data:
            print("沒有載入數據")
            return
        
        if not output_dir:
            output_dir = os.path.join(self.log_dir, "plots")
        
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = self.data.get("model_name", "model")
        
        # 繪製損失曲線
        save_path = os.path.join(output_dir, f"{model_name}_losses_{timestamp}.png")
        self.plot_training_loss(save_path=save_path, show=show)
        
        # 繪製學習率曲線
        if self.data.get("learning_rates"):
            save_path = os.path.join(output_dir, f"{model_name}_lr_{timestamp}.png")
            self.plot_learning_rate(save_path=save_path, show=show)
        
        # 繪製動態稀疏度曲線
        if self.data.get("dynamic_sparsity", {}).get("enabled", False):
            save_path = os.path.join(output_dir, f"{model_name}_sparsity_{timestamp}.png")
            self.plot_dynamic_sparsity(save_path=save_path, show=show)
        
        # 繪製GPU記憶體使用曲線
        if self.data.get("gpu_memory"):
            save_path = os.path.join(output_dir, f"{model_name}_gpu_mem_{timestamp}.png")
            self.plot_gpu_memory(save_path=save_path, show=show)
        
        # 繪製批次損失曲線
        if self.data.get("batches", {}).get("indices"):
            save_path = os.path.join(output_dir, f"{model_name}_batch_loss_{timestamp}.png")
            self.plot_batch_loss(save_path=save_path, show=show)
        
        print(f"所有圖表已保存到: {output_dir}")
    
    def print_stats(self):
        """打印訓練統計信息摘要"""
        if not self.data:
            print("沒有載入數據")
            return
        
        print("\n" + "="*60)
        print(f"訓練統計摘要: {self.data.get('model_name', '未命名模型')}")
        print(f"記錄時間: {self.data.get('timestamp', 'N/A')}")
        print("="*60)
        
        # 訓練和驗證損失
        if self.data.get("epochs"):
            epochs = self.data["epochs"]
            train_losses = self.data["train_losses"]
            
            print(f"\n訓練Epoch數: {len(epochs)}")
            print(f"訓練損失 - 開始: {train_losses[0]:.12f}, 結束: {train_losses[-1]:.12f}")
            print(f"訓練損失 - 最小值: {min(train_losses):.12f} (Epoch {epochs[np.argmin(train_losses)]})")
            
            if "val_losses" in self.data and self.data["val_losses"]:
                val_losses = self.data["val_losses"]
                print(f"驗證損失 - 開始: {val_losses[0]:.12f}, 結束: {val_losses[-1]:.12f}")
                print(f"驗證損失 - 最小值: {min(val_losses):.12f} (Epoch {epochs[np.argmin(val_losses)]})")
                
                # 計算訓練/驗證損失比率
                final_ratio = train_losses[-1] / val_losses[-1]
                print(f"最終訓練/驗證損失比: {final_ratio:.4f}")
                
                # 過擬合指標
                if final_ratio < 0.8:
                    print("過擬合風險: 低 (訓練損失顯著低於驗證損失)")
                elif final_ratio > 1.2:
                    print("過擬合風險: 高 (訓練損失顯著高於驗證損失)")
                else:
                    print("過擬合風險: 中等 (訓練和驗證損失相近)")
        
        # 學習率
        if self.data.get("learning_rates"):
            lr_values = self.data["learning_rates"]
            print(f"\n學習率 - 開始: {lr_values[0]:.10f}, 結束: {lr_values[-1]:.10f}")
            print(f"學習率 - 範圍: {min(lr_values):.10f} - {max(lr_values):.10f}")
        
        # 動態稀疏度
        if self.data.get("dynamic_sparsity", {}).get("enabled", False):
            print("\n動態稀疏度統計:")
            if self.data["dynamic_sparsity"]["compression_ratios"]:
                cr_values = self.data["dynamic_sparsity"]["compression_ratios"]
                print(f"  壓縮比 - 範圍: {min(cr_values):.2f} - {max(cr_values):.2f}")
                print(f"  壓縮比 - 平均值: {np.mean(cr_values):.2f}")
                print(f"  壓縮比 - 標準差: {np.std(cr_values):.2f}")
            
            if self.data["dynamic_sparsity"]["select_ks"]:
                sk_values = self.data["dynamic_sparsity"]["select_ks"]
                print(f"  選擇數量 - 範圍: {min(sk_values)} - {max(sk_values)}")
                print(f"  選擇數量 - 平均值: {np.mean(sk_values):.2f}")
                print(f"  選擇數量 - 標準差: {np.std(sk_values):.2f}")
        
        # 檢查點信息
        if self.data.get("checkpoints"):
            print("\n檢查點信息:")
            for i, cp in enumerate(self.data["checkpoints"]):
                is_best = cp.get("is_best", False)
                best_mark = " (最佳)" if is_best else ""
                
                print(f"  [{i+1}] Epoch {cp['epoch']}{best_mark}: {cp['path']}")
                if "val_loss" in cp:
                    print(f"      驗證損失: {cp['val_loss']:.12f}")
                print(f"      保存時間: {cp.get('timestamp', 'N/A')}")
        
        # 配置信息
        if self.data.get("config"):
            print("\n模型配置:")
            config = self.data["config"]
            for key in sorted(config.keys()):
                value = config[key]
                if isinstance(value, (int, float, str, bool)) or value is None:
                    print(f"  {key}: {value}")
            
            # 特別顯示重要參數
            important_params = [
                "hidden_size", "num_attention_heads", "num_hidden_layers",
                "compress_ratio", "select_k", "use_dynamic_sparsity"
            ]
            
            print("\n關鍵參數:")
            for param in important_params:
                if param in config:
                    print(f"  {param}: {config[param]}")
        
        print("\n" + "="*60)

class EnhancedSPTokenizer:
    def __init__(self, model_path=None, vocab_size=32000, model_type="unigram", 
                 max_length=512, special_tokens=None):
        """初始化分詞器
        
        Args:
            model_path: 預訓練模型路徑
            vocab_size: 詞彙表大小 
            model_type: 'bpe' 或 'unigram'
            max_length: 最大序列長度
            special_tokens: 特殊標記字典
        """
        self.max_length = max_length
        self.vocab_size = vocab_size
        self.model_type = model_type
        
        # 初始化特殊標記
        self.special_tokens = {
            "cls_token": "[CLS]",
            "sep_token": "[SEP]", 
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "mask_token": "[MASK]"
        }
        if special_tokens:
            self.special_tokens.update(special_tokens)
            
        # 設置標記ID
        self.cls_token = self.special_tokens["cls_token"]
        self.sep_token = self.special_tokens["sep_token"]
        self.pad_token = self.special_tokens["pad_token"]
        self.unk_token = self.special_tokens["unk_token"]
        
        self.cls_token_id = 0
        self.sep_token_id = 1
        self.pad_token_id = 2
        self.unk_token_id = 3
        
        # 載入已有模型
        if model_path:
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(model_path)
            self.vocab_size = self.sp_model.GetPieceSize()
        else:
            self.sp_model = None
    
    def train(self, texts, model_prefix="sp_model"):
        """使用給定文本訓練模型"""
        import tempfile
        import os
        
        # 計算實際數據量
        num_sentences = len(texts) 
        print(f"Training SentencePiece model with {num_sentences} sentences")

        # 將文本寫入臨時文件
        with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False) as f:
            for text in texts:
                if isinstance(text, dict) and 'text' in text:
                    f.write(text['text'] + '\n')
                elif isinstance(text, str):
                    f.write(text + '\n')
            temp_file = f.name
            
        # 計算平均文本長度和其他統計
        from collections import Counter
        total_chars = 0
        char_freq = Counter()
        word_freq = Counter()
        # 收集統計信息
        for text in texts:
            if isinstance(text, dict):
                text = text.get('text', '')
            if isinstance(text, str):
                total_chars += len(text)
                char_freq.update(text)
                words = text.split()
                word_freq.update(words)
       
        unique_chars = len(char_freq)
        print(f"- Unique characters: {unique_chars}")
        print(f"- self.vocab_size: {self.vocab_size}")
        

        try:
            # 計算建議的模型參數
            char_coverage = max(0.9999, unique_chars / self.vocab_size)

            # 訓練參數
            train_args = {
                'input': temp_file,
                'model_prefix': model_prefix,
                'vocab_size': self.vocab_size,
                'model_type': self.model_type,
                'character_coverage': char_coverage,
                'user_defined_symbols': list(self.special_tokens.values()),
                'pad_id': self.pad_token_id,
                'unk_id': self.unk_token_id,
                'bos_id': self.cls_token_id,
                'eos_id': self.sep_token_id,
                'input_sentence_size': num_sentences,
                'shuffle_input_sentence': True
            }
            
            print("Starting SentencePiece training with parameters:")
            for key, value in train_args.items():
                print(f"  {key}: {value}")
            # 訓練模型
            spm.SentencePieceTrainer.Train(**train_args)
            
            # 載入訓練好的模型
            self.sp_model = spm.SentencePieceProcessor()
            self.sp_model.Load(f"{model_prefix}.model")
            
            print(f"Successfully trained SentencePiece model with {self.vocab_size} vocab size")
            
        finally:
            # 清理臨時文件
            if os.path.exists(temp_file):
                os.remove(temp_file)
                
    def ensure_model_exists(self):
        """確保模型已經載入或訓練"""
        if self.sp_model is None:
            raise RuntimeError("No model loaded. Please either load a pre-trained model or train a new one using the train() method.")

    def tokenize(self, text):
        """將文本分詞為token列表"""
        self.ensure_model_exists()
        return self.sp_model.EncodeAsPieces(text)
    
    def convert_tokens_to_ids(self, tokens):
        """將token轉換為ID"""
        self.ensure_model_exists()
        return [self.sp_model.PieceToId(token) for token in tokens]
    
    def convert_ids_to_tokens(self, ids):
        """將ID轉換為token"""
        self.ensure_model_exists()
        return [self.sp_model.IdToPiece(id_) for id_ in ids]
    
    def decode(self, token_ids, skip_special_tokens=True):
        """將token ID序列解碼為文本"""
        self.ensure_model_exists()
        if skip_special_tokens:
            token_ids = [id_ for id_ in token_ids if id_ not in 
                        [self.cls_token_id, self.sep_token_id, self.pad_token_id]]
        tokens = self.convert_ids_to_tokens(token_ids)
        return self.sp_model.DecodePieces(tokens)
    
    def __call__(self, text, max_length=None, padding="max_length", 
                 truncation=True, return_tensors=None):
        """主要的分詞方法"""
        self.ensure_model_exists()
        import torch
        
        if max_length is None:
            max_length = self.max_length
            
        # 分詞
        tokens = self.tokenize(text)
        
        # 截斷
        if truncation and len(tokens) > max_length - 2:  # 為CLS和SEP保留空間
            tokens = tokens[:max_length - 2]
            
        # 添加特殊標記
        tokens = [self.cls_token] + tokens + [self.sep_token]
        
        # 轉換為ID
        input_ids = self.convert_tokens_to_ids(tokens)
        
        # 創建attention mask
        attention_mask = [1] * len(input_ids)
        
        # Padding
        if padding == "max_length":
            pad_length = max_length - len(input_ids)
            input_ids = input_ids + [self.pad_token_id] * pad_length
            attention_mask = attention_mask + [0] * pad_length
            
        # 轉換為張量
        if return_tensors == "pt":
            input_ids = torch.tensor([input_ids])
            attention_mask = torch.tensor([attention_mask])
            
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }

def compare_phases(phase1_log, phase2_log, output_dir="comparison_plots"):
    """比較不同階段的訓練結果"""
    # 創建輸出目錄
    os.makedirs(output_dir, exist_ok=True)
    
    # 載入第一階段數據
    viz1 = TrainingVisualizer()
    if not viz1.load_log(phase1_log):
        print(f"無法載入第一階段日誌: {phase1_log}")
        return
    
    # 載入第二階段數據
    viz2 = TrainingVisualizer()
    if not viz2.load_log(phase2_log):
        print(f"無法載入第二階段日誌: {phase2_log}")
        return
    
    # 比較訓練損失
    plt.figure(figsize=(12, 6))
    
    # 第一階段訓練損失
    epochs1 = viz1.data["epochs"]
    train_losses1 = viz1.data["train_losses"]
    plt.plot(epochs1, train_losses1, 'b-', marker='o', label='第一階段訓練損失', alpha=0.7)
    
    # 第一階段驗證損失
    if "val_losses" in viz1.data and viz1.data["val_losses"]:
        val_losses1 = viz1.data["val_losses"]
        plt.plot(epochs1, val_losses1, 'b--', marker='x', label='第一階段驗證損失', alpha=0.7)
    
    # 第二階段訓練損失 (需要偏移epoch數)
    offset = max(epochs1) if epochs1 else 0
    epochs2 = [e + offset for e in viz2.data["epochs"]]
    train_losses2 = viz2.data["train_losses"]
    plt.plot(epochs2, train_losses2, 'r-', marker='o', label='第二階段訓練損失', alpha=0.7)
    
    # 第二階段驗證損失
    if "val_losses" in viz2.data and viz2.data["val_losses"]:
        val_losses2 = viz2.data["val_losses"]
        plt.plot(epochs2, val_losses2, 'r--', marker='x', label='第二階段驗證損失', alpha=0.7)
    
    # 添加階段分隔線
    plt.axvline(x=offset, color='k', linestyle='--', alpha=0.5, label='階段分隔')
    
    plt.title('兩階段訓練損失比較')
    plt.xlabel('Epoch')
    plt.ylabel('損失')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存圖表
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "combined_loss_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 比較學習率變化
    if viz1.data.get("learning_rates") and viz2.data.get("learning_rates"):
        plt.figure(figsize=(12, 5))
        
        # 第一階段學習率
        lr1 = viz1.data["learning_rates"]
        plt.plot(epochs1, lr1, 'b-', marker='o', label='第一階段學習率', alpha=0.7)
        
        # 第二階段學習率
        lr2 = viz2.data["learning_rates"]
        plt.plot(epochs2, lr2, 'r-', marker='o', label='第二階段學習率', alpha=0.7)
        
        # 添加階段分隔線
        plt.axvline(x=offset, color='k', linestyle='--', alpha=0.5, label='階段分隔')
        
        plt.title('兩階段學習率比較')
        plt.xlabel('Epoch')
        plt.ylabel('學習率')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.yscale('log')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "combined_lr_comparison.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # 生成動態稀疏度比較（只有第二階段有）
    if viz2.data.get("dynamic_sparsity", {}).get("enabled", False):
        # 壓縮比
        plt.figure(figsize=(12, 5))
        steps = viz2.data["dynamic_sparsity"]["steps"]
        compression_ratios = viz2.data["dynamic_sparsity"]["compression_ratios"]
        plt.plot(steps, compression_ratios, 'b-', alpha=0.6)
        
        # 添加平滑線
        if len(compression_ratios) > 20:
            window_size = min(20, len(compression_ratios) // 10)
            smoothed = np.convolve(compression_ratios, np.ones(window_size)/window_size, mode='valid')
            valid_steps = steps[window_size-1:]
            if len(valid_steps) == len(smoothed):
                plt.plot(valid_steps, smoothed, 'r-', linewidth=2, label=f'移動平均 (窗口={window_size})')
        
        plt.title('第二階段動態壓縮比變化')
        plt.xlabel('步驟')
        plt.ylabel('壓縮比')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dynamic_compression_ratio.png"), dpi=300, bbox_inches='tight')
        plt.close()
        
        # 選擇數量
        plt.figure(figsize=(12, 5))
        select_ks = viz2.data["dynamic_sparsity"]["select_ks"]
        plt.plot(steps, select_ks, 'g-', alpha=0.6)
        
        # 添加平滑線
        if len(select_ks) > 20:
            window_size = min(20, len(select_ks) // 10)
            smoothed = np.convolve(select_ks, np.ones(window_size)/window_size, mode='valid')
            valid_steps = steps[window_size-1:]
            if len(valid_steps) == len(smoothed):
                plt.plot(valid_steps, smoothed, 'm-', linewidth=2, label=f'移動平均 (窗口={window_size})')
        
        plt.title('第二階段動態選擇數量變化')
        plt.xlabel('步驟')
        plt.ylabel('選擇數量')
        plt.grid(True, linestyle='--', alpha=0.7)
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "dynamic_select_k.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    print(f"比較圖表已保存到目錄: {output_dir}")

# 修改 main() 函數中讀取 parquet 檔案的部分
def read_parquet_files(data_dir, pattern="train-0001*-of-00322.parquet"):
    """讀取 parquet 檔案並處理編碼
    
    Args:
        data_dir: 數據目錄路徑
        pattern: 檔案匹配模式
    
    Returns:
        list: 處理後的文本數據列表
    """
    import pandas as pd
    import glob
    import os
    
    dataset = []
    parquet_files = glob.glob(os.path.join(data_dir, pattern))
    
    for file_path in parquet_files:
        try:
            # 使用 pandas 讀取 parquet 檔案
            df = pd.read_parquet(
                file_path,
                engine='pyarrow'  # 使用 pyarrow 引擎以更好地處理 UTF-8
            )
            
            # 檢查 df 中是否包含 'text' 欄位
            if 'text' in df.columns:
                # 確保文本是 UTF-8 編碼
                for text in df['text']:
                    if isinstance(text, str) and text.strip():
                        # 檢查並清理文本
                        try:
                            # 將文本轉換為 UTF-8 編碼並解碼
                            clean_text = text.encode('utf-8').decode('utf-8')
                            # 移除不可見字符和控制字符
                            clean_text = ''.join(char for char in clean_text 
                                              if char.isprintable() or char in ['\n', '\t'])
                            if clean_text.strip():
                                dataset.append({"text": clean_text})
                        except UnicodeError as e:
                            print(f"Warning: 跳過編碼有問題的文本: {str(e)}")
                            continue
            else:
                print(f"警告：{file_path} 中沒有找到 'text' 欄位")
                
        except Exception as e:
            print(f"讀取檔案 {file_path} 時發生錯誤: {str(e)}")
            continue
    
    print(f"成功載入 {len(dataset)} 條數據")
    print(f"已處理 {len(parquet_files)} 個 parquet 檔案")
    
    return dataset

def main():
    try:
        with open("紅樓夢.json", 'r', encoding='utf-8') as f:
            raw_data = json.load(f)
        
        # 將原始數據轉換為符合模型訓練需求的格式
        dataset = []
        for item in raw_data:
            if "text" in item and item["text"].strip():
                dataset.append({"text": item["text"]})
        
        print(f"成功載入 {len(dataset)} 條數據")
    except Exception as e:
        print(f"載入數據時出錯: {e}")
    
    print("初始化分詞器...")
    enhanced_tokenizer = EnhancedSPTokenizer(vocab_size=16000)
    
    # 使用數據集訓練分詞器
    print("訓練分詞器...")
    text_samples = [item["text"] for item in dataset if isinstance(item.get("text", ""), str)]
    if text_samples:
        enhanced_tokenizer.train(text_samples, model_prefix="chinese_sp_model")
    else:
        raise ValueError("No valid text samples found in dataset for tokenizer training")

    # 設定模型配置 - 初始配置（第一階段使用）
    print("初始化模型配置...")
    config = NSAConfig(
        # MoE相關參數調整
        num_routed_experts=4,
        num_shared_experts=1,
        expert_capacity_factor=1.5,
        router_z_loss_coef=0.0001,
        top_k=2,
        
        # 模型結構參數
        vocab_size=enhanced_tokenizer.vocab_size,
        max_seq_length=512,
        hidden_size=896,
        num_attention_heads=8,
        num_hidden_layers=8,
        compress_ratio=4,
        select_k=16,
        window_size=64,
        
        # 第一階段禁用動態稀疏度
        use_dynamic_sparsity=False
    )
    
    # 創建模型
    print("創建模型...")
    model = NSAModel(config)
    
    # 準備數據集
    print("準備訓練和測試數據集...")
    train_size = int(0.9 * len(dataset))
    indices = torch.randperm(len(dataset))
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]
    
    train_data = [dataset[i] for i in train_indices]
    test_data = [dataset[i] for i in test_indices]
    
    train_dataset = ChineseTextDataset(train_data, enhanced_tokenizer)
    test_dataset = ChineseTextDataset(test_data, enhanced_tokenizer)
    
    #---------- 第一階段訓練：固定稀疏度 ----------#
    print("\n" + "="*80)
    print("第一階段訓練開始：禁用動態稀疏度")
    print("="*80 + "\n")

    # 創建第一階段訓練器（使用增強版的訓練器，包含記錄功能）
    trainer_phase1 = create_trainer_with_warmup_and_logging(
        phase=1,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    
    # 開始第一階段訓練
    trainer_phase1.train()
    
    # 保存第一階段模型
    print("保存第一階段模型...")
    trainer_phase1.save_model("nsa_chinese_model_phase1.pth")
    
    # 清理GPU記憶體
    trainer_phase1._clean_gpu_memory()
    
    #---------- 第二階段訓練：啟用動態稀疏度 ----------#
    print("\n" + "="*80)
    print("第二階段訓練開始：啟用動態稀疏度")
    print("="*80 + "\n")
    
    # 修改配置以啟用動態稀疏度
    config.use_dynamic_sparsity = True
    config.min_select_k = 8
    config.max_select_k = 28
    config.min_compression_ratio = 2
    config.max_compression_ratio = 6
    config.entropy_factor = 0.4
    
    # 重新初始化層以啟用動態稀疏度控制
    print("更新模型配置，啟用動態稀疏度...")
    for i, layer in enumerate(model.layers):
        if hasattr(layer, 'attention') and isinstance(layer.attention, NSAAttentionExtended):
            print(f"  正在為層 {i} 啟用動態稀疏度...")
            # 啟用動態稀疏度
            layer.attention.use_dynamic_sparsity = True
            # 創建稀疏度控制器（如果不存在）
            if not hasattr(layer.attention, 'sparsity_controller'):
                layer.attention.sparsity_controller = DynamicSparsityController(
                    base_select_k=config.select_k,
                    min_select_k=config.min_select_k,
                    max_select_k=config.max_select_k,
                    base_compression_ratio=config.compress_ratio,
                    min_compression_ratio=config.min_compression_ratio,
                    max_compression_ratio=config.max_compression_ratio,
                    entropy_factor=config.entropy_factor
                )
            layer.attention.current_select_k = config.select_k
            layer.attention.current_compress_ratio = config.compress_ratio
            layer.attention.sparsity_history = {"select_k": [], "compress_ratio": []}
    
    # 更新模型的動態稀疏度標誌
    model.use_dynamic_sparsity = True
    model.dynamic_sparsity_metrics = {"compression_ratios": [], "select_ks": [], "steps": []}
    model.step_counter = 0
    
    # 創建第二階段訓練器（使用增強版的訓練器，包含記錄功能）
    trainer_phase2 = create_trainer_with_warmup_and_logging(
        phase=2,
        model=model,
        train_dataset=train_dataset,
        test_dataset=test_dataset
    )
    
    # 開始第二階段訓練
    trainer_phase2.train()
    
    # 保存最終模型
    print("保存最終模型...")
    trainer_phase2.save_model("nsa_chinese_model_final.pth")

    # 顯示訓練完成後的稀疏度統計
    print("\n" + "="*40)
    print("動態稀疏度訓練統計")
    print("="*40)
    
    metrics = model.get_sparsity_metrics()
    
    if metrics.get("dynamic_sparsity_enabled", False):
        cr_stats = metrics.get("compression_ratio", {})
        sk_stats = metrics.get("select_k", {})
        
        #print(f"壓縮比 - 最小值: {cr_stats.get('min', 'N/A')}")
        #print(f"壓縮比 - 最大值: {cr_stats.get('max', 'N/A')}")
        #print(f"壓縮比 - 平均值: {cr_stats.get('avg', 'N/A'):.2f}")
        
        #print(f"選擇數 - 最小值: {sk_stats.get('min', 'N/A')}")
        #print(f"選擇數 - 最大值: {sk_stats.get('max', 'N/A')}")
        #print(f"選擇數 - 平均值: {sk_stats.get('avg', 'N/A'):.2f}")
    else:
        print("動態稀疏度未啟用或未收集到指標數據")
    
    # 使用可視化工具分析訓練結果
    print("\n" + "="*40)
    print("生成訓練過程可視化圖表")
    print("="*40)
    
    try:
        # 為第一階段訓練生成圖表
        visualizer1 = TrainingVisualizer(log_dir="training_logs_phase1")
        log_files1 = visualizer1.find_log_files()
        if log_files1:
            log_files1.sort(key=os.path.getmtime, reverse=True)
            visualizer1.load_log(log_files1[0])
            visualizer1.plot_all(output_dir="plots_phase1", show=False)
            visualizer1.print_stats()
        
        # 為第二階段訓練生成圖表
        visualizer2 = TrainingVisualizer(log_dir="training_logs_phase2")
        log_files2 = visualizer2.find_log_files()
        if log_files2:
            log_files2.sort(key=os.path.getmtime, reverse=True)
            visualizer2.load_log(log_files2[0])
            visualizer2.plot_all(output_dir="plots_phase2", show=False)
            visualizer2.print_stats()
        
        # 比較兩個階段的訓練結果
        if log_files1 and log_files2:
            compare_phases(log_files1[0], log_files2[0], output_dir="comparison_plots")
            print("兩階段訓練結果比較圖表已生成，可在 comparison_plots 目錄查看")
        
        print("圖表生成完成，可在 plots_phase1 和 plots_phase2 目錄查看")
    except Exception as e:
        print(f"生成圖表時發生錯誤: {e}")
    
    print("\n訓練完成！")

if __name__ == "__main__":
    main()