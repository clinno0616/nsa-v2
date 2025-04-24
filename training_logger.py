import os
import json
import time
import datetime
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimSun', 'FangSong']  # 嘗試多種字體
plt.rcParams['axes.unicode_minus'] = False  # 解決負號顯示問題
import numpy as np
import threading


class TrainingLogger:
    """
    記錄和可視化訓練過程中的各種指標
    """
    def __init__(self, log_dir='logs', model_name='nsa_model'):
        # 創建日誌目錄
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.model_name = model_name
        self.timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f"{model_name}_{self.timestamp}.json")
        self.plot_dir = os.path.join(log_dir, 'plots')
        os.makedirs(self.plot_dir, exist_ok=True)
        
        # 初始化記錄數據
        self.data = {
            "model_name": model_name,
            "timestamp": self.timestamp,
            "epochs": [],
            "train_losses": [],
            "val_losses": [],
            "learning_rates": [],
            "gpu_memory": [],
            "time_elapsed": [],
            "dynamic_sparsity": {
                "enabled": False,
                "compression_ratios": [],
                "select_ks": [],
                "steps": []
            },
            "checkpoints": [],
            "config": {}
        }
        
        # 是否正在進行記錄
        self.is_recording = False
        self.record_thread = None
        
    def start_recording(self, interval=5.0):
        """開始定期記錄，間隔為interval秒"""
        self.is_recording = True
        
        def record_loop():
            while self.is_recording:
                try:
                    self.save_log()
                    time.sleep(interval)
                except Exception as e:
                    print(f"記錄過程發生錯誤: {e}")
                    time.sleep(interval)
        
        self.record_thread = threading.Thread(target=record_loop)
        self.record_thread.daemon = True
        self.record_thread.start()
        print(f"訓練記錄器已啟動，每 {interval} 秒記錄一次")
        
    def stop_recording(self):
        """停止記錄"""
        self.is_recording = False
        if self.record_thread:
            self.record_thread.join(timeout=1.0)
        self.save_log()
        print("訓練記錄器已停止")
        
    def record_config(self, config):
        """記錄模型配置"""
        if isinstance(config, object) and not isinstance(config, dict):
            # 如果是對象，嘗試獲取其字典表示
            config_dict = {}
            for key in dir(config):
                if not key.startswith('_'):
                    value = getattr(config, key)
                    if not callable(value):
                        config_dict[key] = value
            self.data["config"] = config_dict
        else:
            # 如果已經是字典或其他可序列化對象
            self.data["config"] = config
            
    def record_epoch(self, epoch, train_loss, val_loss=None, lr=None, gpu_memory=None, time_elapsed=None):
        """記錄每個epoch的訓練數據"""
        # 確保 epoch 不會重複添加（如果多次調用）
        if epoch not in self.data["epochs"]:
            self.data["epochs"].append(epoch)
            self.data["train_losses"].append(float(train_loss))
            
            # 如果提供了驗證損失，也進行記錄
            if val_loss is not None:
                # 如果驗證損失數組長度與 epochs 不一致，可能需要補充
                while len(self.data["val_losses"]) < len(self.data["epochs"]) - 1:
                    # 填充 None 以保持對齊
                    self.data["val_losses"].append(None)
                self.data["val_losses"].append(float(val_loss))
            
            # 記錄學習率
            if lr is not None:
                self.data["learning_rates"].append(float(lr))
            
            # 記錄 GPU 記憶體使用情況
            if gpu_memory is not None:
                # 確保 GPU 記憶體數據是可序列化的
                if isinstance(gpu_memory, dict):
                    self.data["gpu_memory"].append({str(k): float(v) for k, v in gpu_memory.items()})
                else:
                    self.data["gpu_memory"].append(float(gpu_memory))
            
            # 記錄時間
            if time_elapsed is not None:
                if isinstance(time_elapsed, datetime.timedelta):
                    self.data["time_elapsed"].append(time_elapsed.total_seconds())
                else:
                    self.data["time_elapsed"].append(float(time_elapsed))
        else:
            # 如果 epoch 已存在，則更新相應的值
            idx = self.data["epochs"].index(epoch)
            
            # 更新訓練損失
            if idx < len(self.data["train_losses"]):
                self.data["train_losses"][idx] = float(train_loss)
            else:
                self.data["train_losses"].append(float(train_loss))
            
            # 更新驗證損失
            if val_loss is not None:
                if len(self.data["val_losses"]) > idx:
                    self.data["val_losses"][idx] = float(val_loss)
                else:
                    # 如果需要，用 None 填充間隔
                    while len(self.data["val_losses"]) < idx:
                        self.data["val_losses"].append(None)
                    self.data["val_losses"].append(float(val_loss))
            
            # 更新學習率
            if lr is not None and len(self.data["learning_rates"]) > idx:
                self.data["learning_rates"][idx] = float(lr)
                
            # 更新 GPU 記憶體
            if gpu_memory is not None and len(self.data["gpu_memory"]) > idx:
                if isinstance(gpu_memory, dict):
                    self.data["gpu_memory"][idx] = {str(k): float(v) for k, v in gpu_memory.items()}
                else:
                    self.data["gpu_memory"][idx] = float(gpu_memory)
                    
            # 更新時間
            if time_elapsed is not None and len(self.data["time_elapsed"]) > idx:
                if isinstance(time_elapsed, datetime.timedelta):
                    self.data["time_elapsed"][idx] = time_elapsed.total_seconds()
                else:
                    self.data["time_elapsed"][idx] = float(time_elapsed)
                
    def record_batch(self, batch_idx, loss, lr=None):
        """記錄每個batch的訓練數據 (簡化版)"""
        if "batches" not in self.data:
            self.data["batches"] = {"indices": [], "losses": [], "learning_rates": []}
            
        self.data["batches"]["indices"].append(batch_idx)
        self.data["batches"]["losses"].append(float(loss))
        
        if lr is not None:
            self.data["batches"]["learning_rates"].append(float(lr))
            
    def record_sparsity_metrics(self, is_enabled, compression_ratio=None, select_k=None, step=None):
        """記錄動態稀疏度指標"""
        self.data["dynamic_sparsity"]["enabled"] = bool(is_enabled)
        
        if compression_ratio is not None:
            self.data["dynamic_sparsity"]["compression_ratios"].append(float(compression_ratio))
            
        if select_k is not None:
            self.data["dynamic_sparsity"]["select_ks"].append(int(select_k))
            
        if step is not None:
            self.data["dynamic_sparsity"]["steps"].append(int(step))
            
    def record_checkpoint(self, checkpoint_path, epoch, val_loss=None, is_best=False):
        """記錄模型檢查點信息"""
        checkpoint_info = {
            "path": checkpoint_path,
            "epoch": epoch,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_best": is_best
        }
        
        if val_loss is not None:
            checkpoint_info["val_loss"] = float(val_loss)
            
        self.data["checkpoints"].append(checkpoint_info)
        
    def save_log(self):
        """將記錄數據保存到JSON文件"""
        try:
            with open(self.log_file, 'w', encoding='utf-8') as f:
                json.dump(self.data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"保存日誌文件時發生錯誤: {e}")
            
    def generate_plots(self):
        """根據記錄的數據生成圖表"""
        if not self.data["epochs"]:
            print("沒有足夠的數據來生成圖表")
            return
            
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 1. 訓練和驗證損失圖
        plt.figure(figsize=(12, 6))
        epochs = self.data["epochs"]
        train_losses = self.data["train_losses"]
        
        # 確保訓練損失和 epochs 長度一致
        if len(epochs) == len(train_losses):
            plt.plot(epochs, train_losses, 'b-', label='訓練損失')
        else:
            print(f"警告: epochs長度({len(epochs)})與train_losses長度({len(train_losses)})不一致")
            min_len = min(len(epochs), len(train_losses))
            plt.plot(epochs[:min_len], train_losses[:min_len], 'b-', label='訓練損失')
        
        # 處理驗證損失
        if self.data["val_losses"]:
            val_losses = self.data["val_losses"]
            
            # 檢查驗證損失數組長度
            if len(val_losses) == len(epochs):
                # 長度一致，直接繪製
                plt.plot(epochs, val_losses, 'r-', label='驗證損失')
            else:
                print(f"警告: epochs長度({len(epochs)})與val_losses長度({len(val_losses)})不一致")
                
                # 情況1: 驗證頻率小於訓練頻率（例如每2個epoch驗證一次）
                if len(epochs) > len(val_losses) and len(epochs) % len(val_losses) == 0:
                    # 計算驗證頻率
                    validation_frequency = len(epochs) // len(val_losses)
                    # 創建對應的epoch索引
                    val_epochs = epochs[::validation_frequency]
                    # 如果長度還是不一致，取最小長度
                    min_len = min(len(val_epochs), len(val_losses))
                    plt.plot(val_epochs[:min_len], val_losses[:min_len], 'r-', label='驗證損失')
                else:
                    # 情況2: 無法確定規律，僅使用前 min_len 個數據點
                    min_len = min(len(epochs), len(val_losses))
                    plt.plot(epochs[:min_len], val_losses[:min_len], 'r-', label='驗證損失')
        
        plt.title('訓練與驗證損失')
        plt.xlabel('Epoch')
        plt.ylabel('損失')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.plot_dir, f"{self.model_name}_losses_{timestamp}.png"), dpi=200)
        
        # 2. 學習率變化圖
        if self.data["learning_rates"]:
            plt.figure(figsize=(12, 4))
            
            # 確保學習率和 epochs 長度一致
            lr_values = self.data["learning_rates"]
            if len(epochs) == len(lr_values):
                plt.plot(epochs, lr_values, 'g-')
            else:
                print(f"警告: epochs長度({len(epochs)})與learning_rates長度({len(lr_values)})不一致")
                min_len = min(len(epochs), len(lr_values))
                plt.plot(epochs[:min_len], lr_values[:min_len], 'g-')
                
            plt.title('學習率變化')
            plt.xlabel('Epoch')
            plt.ylabel('學習率')
            plt.grid(True)
            plt.yscale('log')  # 使用對數刻度以便更好地可視化學習率的變化
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f"{self.model_name}_lr_{timestamp}.png"), dpi=200)
            
        # 3. 動態稀疏度參數圖
        if self.data["dynamic_sparsity"]["enabled"] and self.data["dynamic_sparsity"]["steps"]:
            plt.figure(figsize=(12, 8))
            
            steps = self.data["dynamic_sparsity"]["steps"]
            compression_ratios = self.data["dynamic_sparsity"]["compression_ratios"]
            select_ks = self.data["dynamic_sparsity"]["select_ks"]
            
            # 檢查數組長度
            if len(steps) != len(compression_ratios):
                print(f"警告: steps長度({len(steps)})與compression_ratios長度({len(compression_ratios)})不一致")
                min_len = min(len(steps), len(compression_ratios))
                steps_cr = steps[:min_len]
                compression_ratios = compression_ratios[:min_len]
            else:
                steps_cr = steps
                
            if len(steps) != len(select_ks):
                print(f"警告: steps長度({len(steps)})與select_ks長度({len(select_ks)})不一致")
                min_len = min(len(steps), len(select_ks))
                steps_sk = steps[:min_len]
                select_ks = select_ks[:min_len]
            else:
                steps_sk = steps
            
            plt.subplot(2, 1, 1)
            plt.plot(steps_cr, compression_ratios, 'b-')
            plt.title('動態壓縮比變化')
            plt.xlabel('步驟')
            plt.ylabel('壓縮比')
            plt.grid(True)
            
            plt.subplot(2, 1, 2)
            plt.plot(steps_sk, select_ks, 'r-')
            plt.title('動態選擇數量變化')
            plt.xlabel('步驟')
            plt.ylabel('選擇數量')
            plt.grid(True)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f"{self.model_name}_sparsity_{timestamp}.png"), dpi=200)
            
        # 4. GPU記憶體使用圖
        if self.data["gpu_memory"]:
            plt.figure(figsize=(12, 4))
            
            gpu_memory = self.data["gpu_memory"]
            
            # 確保GPU記憶體和epochs長度一致
            if len(epochs) != len(gpu_memory):
                print(f"警告: epochs長度({len(epochs)})與gpu_memory長度({len(gpu_memory)})不一致")
                min_len = min(len(epochs), len(gpu_memory))
                plot_epochs = epochs[:min_len]
                gpu_memory = gpu_memory[:min_len]
            else:
                plot_epochs = epochs
            
            # 檢查GPU記憶體數據的格式
            if gpu_memory and isinstance(gpu_memory[0], dict):
                # 如果是字典形式，繪製每個GPU的記憶體使用
                for gpu_id in gpu_memory[0].keys():
                    try:
                        mem_usage = [entry.get(gpu_id, 0) for entry in gpu_memory]
                        plt.plot(plot_epochs, mem_usage, label=f'GPU {gpu_id}')
                    except Exception as e:
                        print(f"繪製GPU {gpu_id}記憶體使用時出錯: {e}")
            else:
                # 否則繪製單一線條
                plt.plot(plot_epochs, gpu_memory, 'c-')
                
            plt.title('GPU記憶體使用')
            plt.xlabel('Epoch')
            plt.ylabel('使用記憶體 (GB)')
            plt.grid(True)
            if gpu_memory and isinstance(gpu_memory[0], dict) and len(gpu_memory[0]) > 1:
                plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f"{self.model_name}_gpu_mem_{timestamp}.png"), dpi=200)
            
        # 5. 批次損失圖（如果有記錄）
        if "batches" in self.data and self.data["batches"]["indices"]:
            plt.figure(figsize=(12, 4))
            batch_indices = self.data["batches"]["indices"]
            batch_losses = self.data["batches"]["losses"]
            
            # 檢查長度是否一致
            if len(batch_indices) != len(batch_losses):
                print(f"警告: batch_indices長度({len(batch_indices)})與batch_losses長度({len(batch_losses)})不一致")
                min_len = min(len(batch_indices), len(batch_losses))
                batch_indices = batch_indices[:min_len]
                batch_losses = batch_losses[:min_len]
                
            plt.plot(batch_indices, batch_losses, 'b-', alpha=0.5)
            
            # 添加平滑線
            if len(batch_losses) > 10:
                smooth_window = min(50, len(batch_losses) // 10)
                try:
                    smoothed = np.convolve(batch_losses, np.ones(smooth_window)/smooth_window, mode='valid')
                    valid_indices = batch_indices[smooth_window-1:len(smoothed)+smooth_window-1]
                    if len(valid_indices) == len(smoothed):
                        plt.plot(valid_indices, smoothed, 'r-', linewidth=2)
                    else:
                        print(f"警告: 平滑後索引長度({len(valid_indices)})與數據長度({len(smoothed)})不一致")
                except Exception as e:
                    print(f"計算平滑批次損失時出錯: {e}")
                
            plt.title('批次訓練損失')
            plt.xlabel('批次索引')
            plt.ylabel('損失')
            plt.grid(True)
            plt.tight_layout()
            plt.savefig(os.path.join(self.plot_dir, f"{self.model_name}_batch_loss_{timestamp}.png"), dpi=200)
            
        plt.close('all')
        print(f"圖表已保存到: {self.plot_dir}")
        
    def load_log(self, log_file):
        """載入已存在的日誌文件進行分析"""
        try:
            with open(log_file, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"成功載入日誌文件: {log_file}")
            return True
        except Exception as e:
            print(f"載入日誌文件時發生錯誤: {e}")
            return False
            
    def print_summary(self):
        """打印訓練過程摘要"""
        print("\n" + "="*50)
        print(f"訓練摘要: {self.model_name}")
        print("="*50)
        
        if self.data["epochs"]:
            total_epochs = max(self.data["epochs"])
            min_train_loss = min(self.data["train_losses"])
            min_train_epoch = self.data["epochs"][self.data["train_losses"].index(min_train_loss)]
            
            print(f"總訓練Epoch: {total_epochs}")
            print(f"最低訓練損失: {min_train_loss:.12f} (Epoch {min_train_epoch})")
            
            if self.data["val_losses"]:
                min_val_loss = min(self.data["val_losses"])
                min_val_epoch = self.data["epochs"][self.data["val_losses"].index(min_val_loss)]
                print(f"最低驗證損失: {min_val_loss:.12f} (Epoch {min_val_epoch})")
                
        if self.data["checkpoints"]:
            best_checkpoints = [cp for cp in self.data["checkpoints"] if cp.get("is_best", False)]
            if best_checkpoints:
                print("\n最佳檢查點:")
                for cp in best_checkpoints:
                    print(f"  Epoch {cp['epoch']}: {cp['path']}")
                    if "val_loss" in cp:
                        print(f"  驗證損失: {cp['val_loss']:.12f}")
                    print(f"  保存時間: {cp['timestamp']}")
                    
        if self.data["dynamic_sparsity"]["enabled"] and self.data["dynamic_sparsity"]["compression_ratios"]:
            cr_values = self.data["dynamic_sparsity"]["compression_ratios"]
            sk_values = self.data["dynamic_sparsity"]["select_ks"]
            
            print("\n動態稀疏度統計:")
            print(f"  壓縮比 - 範圍: {min(cr_values):.2f} - {max(cr_values):.2f}, 平均: {np.mean(cr_values):.2f}")
            print(f"  選擇數 - 範圍: {min(sk_values)} - {max(sk_values)}, 平均: {np.mean(sk_values):.2f}")
        
        print("="*50 + "\n")