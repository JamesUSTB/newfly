"""
跨缝特征提取模块（传荷能力、能量传递、互相关）
"""

import numpy as np
from scipy import signal
from typing import Dict, Tuple


class CrossJointFeatures:
    """跨缝特征提取器"""
    
    @staticmethod
    def compute_lte(signal_left: np.ndarray, signal_right: np.ndarray) -> float:
        """
        传荷能力（Load Transfer Efficiency）
        LTE = peak(right) / peak(left)
        
        Args:
            signal_left: 左板（加载侧）信号
            signal_right: 右板（非加载侧）信号
            
        Returns:
            传荷能力 (0-1范围，理论上)
        """
        peak_left = np.max(np.abs(signal_left))
        peak_right = np.max(np.abs(signal_right))
        
        if peak_left == 0:
            return 0
        
        lte = peak_right / peak_left
        return lte
    
    @staticmethod
    def compute_energy_transfer(signal_left: np.ndarray, 
                               signal_right: np.ndarray) -> float:
        """
        能量传递率
        
        Args:
            signal_left: 左板信号
            signal_right: 右板信号
            
        Returns:
            能量传递率
        """
        energy_left = np.sum(signal_left ** 2)
        energy_right = np.sum(signal_right ** 2)
        
        if energy_left == 0:
            return 0
        
        transfer_ratio = energy_right / energy_left
        return transfer_ratio
    
    @staticmethod
    def compute_cross_correlation(signal_left: np.ndarray,
                                  signal_right: np.ndarray) -> float:
        """
        互相关系数（波形相似度）
        
        Args:
            signal_left: 左板信号
            signal_right: 右板信号
            
        Returns:
            最大互相关系数
        """
        # 归一化信号
        left_norm = (signal_left - np.mean(signal_left)) / (np.std(signal_left) + 1e-10)
        right_norm = (signal_right - np.mean(signal_right)) / (np.std(signal_right) + 1e-10)
        
        # 计算互相关
        correlation = np.correlate(left_norm, right_norm, mode='full')
        
        # 归一化
        max_corr = np.max(np.abs(correlation)) / len(signal_left)
        
        return max_corr
    
    @staticmethod
    def compute_phase_difference(signal_left: np.ndarray,
                                 signal_right: np.ndarray,
                                 fs: float) -> float:
        """
        相位差（波传播延迟）
        
        Args:
            signal_left: 左板信号
            signal_right: 右板信号
            fs: 采样频率
            
        Returns:
            相位差（度）
        """
        # 归一化信号
        left_norm = (signal_left - np.mean(signal_left)) / (np.std(signal_left) + 1e-10)
        right_norm = (signal_right - np.mean(signal_right)) / (np.std(signal_right) + 1e-10)
        
        # 计算互相关
        correlation = np.correlate(left_norm, right_norm, mode='full')
        
        # 找到最大相关位置
        max_idx = np.argmax(np.abs(correlation))
        lag = max_idx - (len(signal_left) - 1)
        
        # 计算时间延迟
        time_delay = lag / fs
        
        # 估计主频
        fft_left = np.fft.rfft(signal_left)
        freqs = np.fft.rfftfreq(len(signal_left), 1/fs)
        dominant_freq = freqs[np.argmax(np.abs(fft_left))]
        
        if dominant_freq == 0:
            return 0
        
        # 计算相位差（度）
        phase_diff = (time_delay * dominant_freq * 360) % 360
        
        return phase_diff
    
    @staticmethod
    def compute_coherence(signal_left: np.ndarray,
                         signal_right: np.ndarray,
                         fs: float) -> Tuple[float, float]:
        """
        相干函数（频域相关性）
        
        Args:
            signal_left: 左板信号
            signal_right: 右板信号
            fs: 采样频率
            
        Returns:
            (平均相干性, 峰值相干性)
        """
        # 计算相干性
        freqs, coherence = signal.coherence(signal_left, signal_right, fs=fs)
        
        mean_coherence = np.mean(coherence)
        peak_coherence = np.max(coherence)
        
        return mean_coherence, peak_coherence
    
    @staticmethod
    def compute_time_lag(signal_left: np.ndarray,
                        signal_right: np.ndarray,
                        fs: float) -> float:
        """
        时间延迟（秒）
        
        Args:
            signal_left: 左板信号
            signal_right: 右板信号
            fs: 采样频率
            
        Returns:
            时间延迟（秒）
        """
        # 归一化信号
        left_norm = (signal_left - np.mean(signal_left)) / (np.std(signal_left) + 1e-10)
        right_norm = (signal_right - np.mean(signal_right)) / (np.std(signal_right) + 1e-10)
        
        # 计算互相关
        correlation = np.correlate(left_norm, right_norm, mode='full')
        
        # 找到最大相关位置
        max_idx = np.argmax(np.abs(correlation))
        lag = max_idx - (len(signal_left) - 1)
        
        # 转换为时间
        time_lag = lag / fs
        
        return time_lag
    
    @staticmethod
    def compute_amplitude_ratio(signal_left: np.ndarray,
                               signal_right: np.ndarray) -> float:
        """
        幅值比（RMS比）
        
        Args:
            signal_left: 左板信号
            signal_right: 右板信号
            
        Returns:
            幅值比
        """
        rms_left = np.sqrt(np.mean(signal_left ** 2))
        rms_right = np.sqrt(np.mean(signal_right ** 2))
        
        if rms_left == 0:
            return 0
        
        return rms_right / rms_left
    
    @staticmethod
    def extract_all(signal_left: np.ndarray,
                   signal_right: np.ndarray,
                   fs: float,
                   depth_label: str = '') -> Dict[str, float]:
        """
        提取所有跨缝特征
        
        Args:
            signal_left: 左板信号
            signal_right: 右板信号
            fs: 采样频率
            depth_label: 深度标签（如'5cm'或'3.5cm'）
            
        Returns:
            特征字典
        """
        features = {}
        prefix = f'{depth_label}_' if depth_label else ''
        
        # 传荷能力
        features[f'{prefix}LTE'] = CrossJointFeatures.compute_lte(signal_left, signal_right)
        
        # 能量传递率
        features[f'{prefix}energy_transfer'] = CrossJointFeatures.compute_energy_transfer(
            signal_left, signal_right
        )
        
        # 互相关系数
        features[f'{prefix}xcorr'] = CrossJointFeatures.compute_cross_correlation(
            signal_left, signal_right
        )
        
        # 相位差
        features[f'{prefix}phase_diff'] = CrossJointFeatures.compute_phase_difference(
            signal_left, signal_right, fs
        )
        
        # 时间延迟
        features[f'{prefix}time_lag'] = CrossJointFeatures.compute_time_lag(
            signal_left, signal_right, fs
        )
        
        # 幅值比
        features[f'{prefix}amplitude_ratio'] = CrossJointFeatures.compute_amplitude_ratio(
            signal_left, signal_right
        )
        
        # 相干性
        mean_coh, peak_coh = CrossJointFeatures.compute_coherence(
            signal_left, signal_right, fs
        )
        features[f'{prefix}coherence_mean'] = mean_coh
        features[f'{prefix}coherence_peak'] = peak_coh
        
        return features
