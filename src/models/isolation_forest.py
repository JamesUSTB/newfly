"""
孤立森林健康状态判别模块
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple
import joblib


class HealthStateDetector:
    """基于孤立森林的健康状态检测器"""
    
    def __init__(self, config: Dict):
        """
        初始化检测器
        
        Args:
            config: 配置参数
        """
        self.config = config
        self.model = IsolationForest(
            n_estimators=config.get('n_estimators', 100),
            contamination=config.get('contamination', 0.1),
            max_samples=config.get('max_samples', 'auto'),
            random_state=config.get('random_state', 42)
        )
        self.scaler = StandardScaler()
        self.thresholds = config.get('thresholds', {
            'healthy': 0.3,
            'mild': 0.5,
            'moderate': 0.7
        })
        
    def select_features(self, X: np.ndarray, feature_names: List[str]) -> Tuple[np.ndarray, List[str]]:
        """
        选择关键特征（跨缝特征为主）
        
        Args:
            X: 完整特征矩阵
            feature_names: 特征名称列表
            
        Returns:
            (选择的特征矩阵, 选择的特征名)
        """
        # 优先选择跨缝特征
        priority_keywords = [
            'LTE', 'energy_transfer', 'xcorr', 'phase_diff',
            'time_lag', 'amplitude_ratio', 'coherence',
            'kurtosis', 'dominant_freq'
        ]
        
        selected_indices = []
        selected_names = []
        
        for i, name in enumerate(feature_names):
            for keyword in priority_keywords:
                if keyword in name:
                    selected_indices.append(i)
                    selected_names.append(name)
                    break
        
        # 如果选择的特征太少，添加更多特征
        if len(selected_indices) < 10 and len(feature_names) > 10:
            remaining_indices = [i for i in range(len(feature_names)) if i not in selected_indices]
            additional_count = min(10 - len(selected_indices), len(remaining_indices))
            selected_indices.extend(remaining_indices[:additional_count])
            selected_names.extend([feature_names[i] for i in remaining_indices[:additional_count]])
        
        X_selected = X[:, selected_indices]
        
        return X_selected, selected_names
    
    def fit(self, X: np.ndarray, feature_names: Optional[List[str]] = None):
        """
        训练模型（无监督）
        
        Args:
            X: 特征矩阵
            feature_names: 特征名称列表
        """
        # 特征选择
        if feature_names:
            X_selected, self.selected_features = self.select_features(X, feature_names)
        else:
            X_selected = X
            self.selected_features = [f'feature_{i}' for i in range(X.shape[1])]
        
        # 标准化
        X_scaled = self.scaler.fit_transform(X_selected)
        
        # 训练孤立森林
        self.model.fit(X_scaled)
    
    def compute_anomaly_score(self, X: np.ndarray) -> np.ndarray:
        """
        计算异常分数
        
        Args:
            X: 特征矩阵
            
        Returns:
            异常分数数组 (0-1, 值越大越异常)
        """
        # 标准化
        X_scaled = self.scaler.transform(X)
        
        # 获取异常分数（decision_function返回负值，越小越异常）
        raw_scores = self.model.decision_function(X_scaled)
        
        # 转换为0-1范围（使用sigmoid变换）
        # 归一化到[0, 1]
        scores_normalized = 1 / (1 + np.exp(raw_scores))
        
        return scores_normalized
    
    def predict_labels(self, X: np.ndarray) -> np.ndarray:
        """
        预测标签（-1: 异常, 1: 正常）
        
        Args:
            X: 特征矩阵
            
        Returns:
            标签数组
        """
        X_scaled = self.scaler.transform(X)
        labels = self.model.predict(X_scaled)
        return labels
    
    def classify_health_state(self, anomaly_score: float) -> str:
        """
        根据异常分数分类健康状态
        
        Args:
            anomaly_score: 异常分数
            
        Returns:
            健康状态字符串
        """
        if anomaly_score < self.thresholds['healthy']:
            return 'Healthy'
        elif anomaly_score < self.thresholds['mild']:
            return 'Mild Anomaly'
        elif anomaly_score < self.thresholds['moderate']:
            return 'Moderate Anomaly'
        else:
            return 'Severe Anomaly'
    
    def analyze_batch(self, X: np.ndarray) -> Dict:
        """
        批量分析
        
        Args:
            X: 特征矩阵
            
        Returns:
            分析结果字典
        """
        # 计算异常分数
        anomaly_scores = self.compute_anomaly_score(X)
        
        # 分类健康状态
        health_states = [self.classify_health_state(score) for score in anomaly_scores]
        
        # 统计
        unique_states, counts = np.unique(health_states, return_counts=True)
        state_distribution = dict(zip(unique_states, counts))
        
        results = {
            'anomaly_scores': anomaly_scores,
            'health_states': health_states,
            'state_distribution': state_distribution,
            'mean_score': np.mean(anomaly_scores),
            'std_score': np.std(anomaly_scores),
            'max_score': np.max(anomaly_scores),
            'min_score': np.min(anomaly_scores)
        }
        
        return results
    
    def identify_anomalous_features(self, X: np.ndarray, 
                                   sample_idx: int,
                                   top_k: int = 5) -> List[Tuple[str, float]]:
        """
        识别导致异常的关键特征
        
        Args:
            X: 特征矩阵
            sample_idx: 样本索引
            top_k: 返回前k个特征
            
        Returns:
            (特征名, 重要性分数)列表
        """
        # 获取该样本的特征值
        sample = X[sample_idx:sample_idx+1]
        
        # 计算每个特征对异常分数的贡献
        # 通过逐个屏蔽特征来评估影响
        base_score = self.compute_anomaly_score(sample)[0]
        
        feature_importance = []
        for i in range(sample.shape[1]):
            # 创建副本并屏蔽该特征（设为均值）
            sample_masked = sample.copy()
            sample_masked[:, i] = np.mean(X[:, i])
            
            # 计算新的异常分数
            masked_score = self.compute_anomaly_score(sample_masked)[0]
            
            # 重要性 = 原始分数 - 屏蔽后分数
            importance = base_score - masked_score
            
            feature_name = self.selected_features[i] if hasattr(self, 'selected_features') else f'feature_{i}'
            feature_importance.append((feature_name, importance))
        
        # 按重要性排序
        feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
        
        return feature_importance[:top_k]
    
    def save(self, model_path: str, scaler_path: str):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
            scaler_path: 标准化器保存路径
        """
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        # 保存选择的特征名
        if hasattr(self, 'selected_features'):
            joblib.dump(self.selected_features, model_path + '.features')
    
    def load(self, model_path: str, scaler_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型路径
            scaler_path: 标准化器路径
        """
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        
        # 加载选择的特征名
        features_path = model_path + '.features'
        try:
            self.selected_features = joblib.load(features_path)
        except:
            pass


# 添加缺失的导入
from typing import Optional
