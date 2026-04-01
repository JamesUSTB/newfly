"""
PIRCN (Physics-Informed Residual Correction Network)
物理引导残差修正网络
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, Tuple, Optional
import joblib


class ResidualCorrectionNetwork(nn.Module):
    """残差修正网络"""
    
    def __init__(self, input_dim: int, hidden_dims: list = [64, 32, 16]):
        """
        初始化网络
        
        Args:
            input_dim: 输入维度
            hidden_dims: 隐藏层维度列表
        """
        super(ResidualCorrectionNetwork, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.2))
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class PhysicsConstraints:
    """物理约束计算"""
    
    @staticmethod
    def dynamic_amplification_constraint(features: np.ndarray, 
                                        speeds: np.ndarray,
                                        peak_idx: int) -> float:
        """
        约束1：动力放大系数（峰值应随速度增加）
        
        Args:
            features: 特征矩阵
            speeds: 速度数组
            peak_idx: 峰值特征索引
            
        Returns:
            约束损失
        """
        # 提取峰值特征
        peaks = features[:, peak_idx]
        
        # 计算峰值对速度的梯度
        # 使用有限差分近似
        gradients = []
        for i in range(len(speeds) - 1):
            if speeds[i+1] != speeds[i]:
                grad = (peaks[i+1] - peaks[i]) / (speeds[i+1] - speeds[i])
                gradients.append(grad)
        
        if not gradients:
            return 0
        
        # 惩罚负梯度（峰值应随速度增加）
        negative_grads = [g for g in gradients if g < 0]
        loss = np.sum(np.abs(negative_grads)) if negative_grads else 0
        
        return loss
    
    @staticmethod
    def frequency_speed_constraint(features: np.ndarray,
                                   speeds: np.ndarray,
                                   freq_idx: int,
                                   axle_distance: float = 3.5) -> float:
        """
        约束2：主频-速度关系 f ∝ v/L
        
        Args:
            features: 特征矩阵
            speeds: 速度数组
            freq_idx: 主频特征索引
            axle_distance: 轴距 (m)
            
        Returns:
            约束损失
        """
        # 提取主频特征
        freqs = features[:, freq_idx]
        
        # 理论频率 = 速度 / 轴距
        speeds_ms = speeds / 3.6  # km/h to m/s
        theoretical_freqs = speeds_ms / axle_distance
        
        # 计算相关系数（应该是正相关）
        if len(freqs) > 1:
            correlation = np.corrcoef(freqs, theoretical_freqs)[0, 1]
            loss = max(0, 1 - correlation)  # 惩罚弱相关
        else:
            loss = 0
        
        return loss
    
    @staticmethod
    def lte_boundary_constraint(features: np.ndarray,
                               lte_indices: list) -> float:
        """
        约束4：传荷能力边界 0 < LTE < 1
        
        Args:
            features: 特征矩阵
            lte_indices: LTE特征索引列表
            
        Returns:
            约束损失
        """
        loss = 0
        for idx in lte_indices:
            lte_values = features[:, idx]
            
            # 惩罚超出边界的值
            below_zero = lte_values < 0
            above_one = lte_values > 1
            
            loss += np.sum(np.abs(lte_values[below_zero]))
            loss += np.sum(lte_values[above_one] - 1)
        
        return loss


class PIRCNModel:
    """PIRCN模型（LightGBM + 物理约束残差修正）"""
    
    def __init__(self, base_predictor, config: Dict):
        """
        初始化PIRCN模型
        
        Args:
            base_predictor: 基础预测器（LightGBM）
            config: 配置参数
        """
        self.base_predictor = base_predictor
        self.config = config
        self.residual_network = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _extract_physics_features(self, X: np.ndarray, 
                                  base_predictions: np.ndarray) -> np.ndarray:
        """
        提取物理约束特征
        
        Args:
            X: 原始特征
            base_predictions: 基础模型预测
            
        Returns:
            增强特征矩阵
        """
        # 组合原始特征和基础预测
        physics_features = np.column_stack([
            X,
            base_predictions.reshape(-1, 1)
        ])
        
        return physics_features
    
    def train(self, X: np.ndarray, y: np.ndarray,
             X_val: Optional[np.ndarray] = None,
             y_val: Optional[np.ndarray] = None) -> Dict[str, list]:
        """
        训练残差修正网络
        
        Args:
            X: 训练特征
            y: 训练标签
            X_val: 验证特征
            y_val: 验证标签
            
        Returns:
            训练历史
        """
        # 获取基础预测
        y_base = self.base_predictor.predict(X)
        residuals = y - y_base
        
        # 提取物理特征
        X_physics = self._extract_physics_features(X, y_base)
        
        # 初始化网络
        input_dim = X_physics.shape[1]
        hidden_dims = self.config.get('hidden_dims', [64, 32, 16])
        self.residual_network = ResidualCorrectionNetwork(input_dim, hidden_dims).to(self.device)
        
        # 准备数据
        X_tensor = torch.FloatTensor(X_physics).to(self.device)
        y_tensor = torch.FloatTensor(residuals).reshape(-1, 1).to(self.device)
        
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, 
                               batch_size=self.config.get('batch_size', 32),
                               shuffle=True)
        
        # 优化器
        optimizer = optim.Adam(self.residual_network.parameters(),
                              lr=self.config.get('learning_rate', 0.001))
        
        # 训练
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.get('epochs', 100)):
            # 训练模式
            self.residual_network.train()
            epoch_loss = 0
            
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()
                
                # 前向传播
                predictions = self.residual_network(batch_X)
                
                # MSE损失
                mse_loss = nn.MSELoss()(predictions, batch_y)
                
                # 物理约束损失
                physics_loss = self._compute_physics_loss(X, y)
                
                # 平滑约束
                smooth_loss = self._compute_smooth_loss(predictions)
                
                # 总损失
                total_loss = (mse_loss + 
                            self.config.get('physics_weight', 0.1) * physics_loss +
                            self.config.get('smooth_weight', 0.01) * smooth_loss)
                
                # 反向传播
                total_loss.backward()
                optimizer.step()
                
                epoch_loss += total_loss.item()
            
            avg_loss = epoch_loss / len(dataloader)
            history['train_loss'].append(avg_loss)
            
            # 验证
            if X_val is not None and y_val is not None:
                val_loss = self._validate(X_val, y_val)
                history['val_loss'].append(val_loss)
                
                # 早停
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                
                if patience_counter >= self.config.get('early_stopping_patience', 10):
                    print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return history
    
    def _compute_physics_loss(self, X: np.ndarray, y: np.ndarray) -> torch.Tensor:
        """计算物理约束损失"""
        # 这里实现简化版的物理约束
        # 实际应用中需要根据具体特征索引计算
        return torch.tensor(0.0, device=self.device)
    
    def _compute_smooth_loss(self, predictions: torch.Tensor) -> torch.Tensor:
        """计算平滑约束损失（相邻预测不应有剧烈变化）"""
        if len(predictions) < 2:
            return torch.tensor(0.0, device=self.device)
        
        diff = predictions[1:] - predictions[:-1]
        smooth_loss = torch.mean(diff ** 2)
        return smooth_loss
    
    def _validate(self, X_val: np.ndarray, y_val: np.ndarray) -> float:
        """验证"""
        self.residual_network.eval()
        
        with torch.no_grad():
            y_base_val = self.base_predictor.predict(X_val)
            residuals_val = y_val - y_base_val
            
            X_physics_val = self._extract_physics_features(X_val, y_base_val)
            X_tensor = torch.FloatTensor(X_physics_val).to(self.device)
            y_tensor = torch.FloatTensor(residuals_val).reshape(-1, 1).to(self.device)
            
            predictions = self.residual_network(X_tensor)
            loss = nn.MSELoss()(predictions, y_tensor)
        
        return loss.item()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测结果
        """
        # 基础预测
        y_base = self.base_predictor.predict(X)
        
        # 残差修正
        if self.residual_network is not None:
            self.residual_network.eval()
            
            X_physics = self._extract_physics_features(X, y_base)
            X_tensor = torch.FloatTensor(X_physics).to(self.device)
            
            with torch.no_grad():
                residual_correction = self.residual_network(X_tensor).cpu().numpy().flatten()
            
            # 最终预测
            y_final = y_base + residual_correction
        else:
            y_final = y_base
        
        return y_final
    
    def save(self, model_path: str):
        """保存模型"""
        if self.residual_network is not None:
            torch.save(self.residual_network.state_dict(), model_path)
    
    def load(self, model_path: str, input_dim: int):
        """加载模型"""
        hidden_dims = self.config.get('hidden_dims', [64, 32, 16])
        self.residual_network = ResidualCorrectionNetwork(input_dim, hidden_dims).to(self.device)
        self.residual_network.load_state_dict(torch.load(model_path))
