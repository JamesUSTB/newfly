"""
时域特征提取模块
"""

import numpy as np
from scipy import stats
from typing import Dict


class TimeDomainFeatures:
    """时域特征提取器"""
    
    @staticmethod
    def compute_peak(signal: np.ndarray) -> float:
        """峰值响应"""
        return np.max(np.abs(signal))
    
    @staticmethod
    def compute_peak_to_peak(signal: np.ndarray) -> float:
        """峰峰值"""
        return np.max(signal) - np.min(signal)
    
    @staticmethod
    def compute_rms(signal: np.ndarray) -> float:
        """均方根值"""
        return np.sqrt(np.mean(signal ** 2))
    
    @staticmethod
    def compute_mean(signal: np.ndarray) -> float:
        """均值"""
        return np.mean(signal)
    
    @staticmethod
    def compute_std(signal: np.ndarray) -> float:
        """标准差"""
        return np.std(signal)
    
    @staticmethod
    def compute_kurtosis(signal: np.ndarray) -> float:
        """峭度（冲击性）"""
        return stats.kurtosis(signal)
    
    @staticmethod
    def compute_skewness(signal: np.ndarray) -> float:
        """偏度（对称性）"""
        return stats.skew(signal)
    
    @staticmethod
    def compute_crest_factor(signal: np.ndarray) -> float:
        """峰值因子"""
        rms = TimeDomainFeatures.compute_rms(signal)
        if rms == 0:
            return 0
        peak = TimeDomainFeatures.compute_peak(signal)
        return peak / rms
    
    @staticmethod
    def compute_shape_factor(signal: np.ndarray) -> float:
        """波形因子"""
        mean_abs = np.mean(np.abs(signal))
        if mean_abs == 0:
            return 0
        rms = TimeDomainFeatures.compute_rms(signal)
        return rms / mean_abs
    
    @staticmethod
    def compute_impulse_factor(signal: np.ndarray) -> float:
        """脉冲因子"""
        mean_abs = np.mean(np.abs(signal))
        if mean_abs == 0:
            return 0
        peak = TimeDomainFeatures.compute_peak(signal)
        return peak / mean_abs
    
    @staticmethod
    def compute_clearance_factor(signal: np.ndarray) -> float:
        """裕度因子"""
        mean_sqrt = np.mean(np.sqrt(np.abs(signal))) ** 2
        if mean_sqrt == 0:
            return 0
        peak = TimeDomainFeatures.compute_peak(signal)
        return peak / mean_sqrt
    
    @staticmethod
    def compute_duration(signal: np.ndarray, fs: float, 
                        threshold_ratio: float = 0.1) -> float:
        """
        有效持续时间（响应衰减特性）
        
        Args:
            signal: 信号
            fs: 采样频率
            threshold_ratio: 阈值比例（相对于峰值）
            
        Returns:
            持续时间（秒）
        """
        envelope = np.abs(signal)
        peak = np.max(envelope)
        threshold = peak * threshold_ratio
        
        # 找到超过阈值的样本
        above_threshold = envelope > threshold
        if not np.any(above_threshold):
            return 0
        
        # 计算首次和末次超过阈值的位置
        indices = np.where(above_threshold)[0]
        duration_samples = indices[-1] - indices[0] + 1
        duration = duration_samples / fs
        
        return duration
    
    @staticmethod
    def compute_zero_crossing_rate(signal: np.ndarray) -> float:
        """过零率"""
        zero_crossings = np.sum(np.abs(np.diff(np.sign(signal)))) / 2
        return zero_crossings / len(signal)
    
    @staticmethod
    def compute_energy(signal: np.ndarray) -> float:
        """信号能量"""
        return np.sum(signal ** 2)
    
    @staticmethod
    def extract_all(signal: np.ndarray, fs: float, 
                   prefix: str = '') -> Dict[str, float]:
        """
        提取所有时域特征
        
        Args:
            signal: 信号
            fs: 采样频率
            prefix: 特征名前缀
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基本统计特征
        features[f'{prefix}peak'] = TimeDomainFeatures.compute_peak(signal)
        features[f'{prefix}peak_to_peak'] = TimeDomainFeatures.compute_peak_to_peak(signal)
        features[f'{prefix}rms'] = TimeDomainFeatures.compute_rms(signal)
        features[f'{prefix}mean'] = TimeDomainFeatures.compute_mean(signal)
        features[f'{prefix}std'] = TimeDomainFeatures.compute_std(signal)
        
        # 高阶统计特征
        features[f'{prefix}kurtosis'] = TimeDomainFeatures.compute_kurtosis(signal)
        features[f'{prefix}skewness'] = TimeDomainFeatures.compute_skewness(signal)
        
        # 形状因子
        features[f'{prefix}crest_factor'] = TimeDomainFeatures.compute_crest_factor(signal)
        features[f'{prefix}shape_factor'] = TimeDomainFeatures.compute_shape_factor(signal)
        features[f'{prefix}impulse_factor'] = TimeDomainFeatures.compute_impulse_factor(signal)
        features[f'{prefix}clearance_factor'] = TimeDomainFeatures.compute_clearance_factor(signal)
        
        # 时间特性
        features[f'{prefix}duration'] = TimeDomainFeatures.compute_duration(signal, fs)
        features[f'{prefix}zero_crossing_rate'] = TimeDomainFeatures.compute_zero_crossing_rate(signal)
        features[f'{prefix}energy'] = TimeDomainFeatures.compute_energy(signal)
        
        return features
