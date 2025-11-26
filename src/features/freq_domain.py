"""
频域特征提取模块
"""

import numpy as np
from scipy import signal as sig
from scipy.fft import fft, fftfreq
from typing import Dict, List, Tuple


class FrequencyDomainFeatures:
    """频域特征提取器"""
    
    @staticmethod
    def compute_psd(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        计算功率谱密度
        
        Args:
            signal: 信号
            fs: 采样频率
            
        Returns:
            (频率, PSD)
        """
        freqs, psd = sig.welch(signal, fs=fs, nperseg=min(256, len(signal)))
        return freqs, psd
    
    @staticmethod
    def compute_dominant_freq(signal: np.ndarray, fs: float) -> float:
        """
        主频（最大能量频率）
        
        Args:
            signal: 信号
            fs: 采样频率
            
        Returns:
            主频 (Hz)
        """
        freqs, psd = FrequencyDomainFeatures.compute_psd(signal, fs)
        dominant_idx = np.argmax(psd)
        return freqs[dominant_idx]
    
    @staticmethod
    def compute_centroid_freq(signal: np.ndarray, fs: float) -> float:
        """
        质心频率
        
        Args:
            signal: 信号
            fs: 采样频率
            
        Returns:
            质心频率 (Hz)
        """
        freqs, psd = FrequencyDomainFeatures.compute_psd(signal, fs)
        total_power = np.sum(psd)
        if total_power == 0:
            return 0
        centroid = np.sum(freqs * psd) / total_power
        return centroid
    
    @staticmethod
    def compute_bandwidth(signal: np.ndarray, fs: float) -> float:
        """
        带宽（频谱宽度）
        
        Args:
            signal: 信号
            fs: 采样频率
            
        Returns:
            带宽 (Hz)
        """
        freqs, psd = FrequencyDomainFeatures.compute_psd(signal, fs)
        centroid = FrequencyDomainFeatures.compute_centroid_freq(signal, fs)
        
        total_power = np.sum(psd)
        if total_power == 0:
            return 0
        
        bandwidth = np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / total_power)
        return bandwidth
    
    @staticmethod
    def compute_band_energy(signal: np.ndarray, fs: float, 
                           freq_band: Tuple[float, float]) -> float:
        """
        计算频带能量占比
        
        Args:
            signal: 信号
            fs: 采样频率
            freq_band: 频带范围 (f_low, f_high)
            
        Returns:
            能量占比 (0-1)
        """
        freqs, psd = FrequencyDomainFeatures.compute_psd(signal, fs)
        
        # 选择频带内的PSD
        band_mask = (freqs >= freq_band[0]) & (freqs <= freq_band[1])
        band_power = np.sum(psd[band_mask])
        total_power = np.sum(psd)
        
        if total_power == 0:
            return 0
        
        return band_power / total_power
    
    @staticmethod
    def compute_spectral_entropy(signal: np.ndarray, fs: float) -> float:
        """
        谱熵
        
        Args:
            signal: 信号
            fs: 采样频率
            
        Returns:
            谱熵
        """
        freqs, psd = FrequencyDomainFeatures.compute_psd(signal, fs)
        
        # 归一化PSD
        psd_norm = psd / np.sum(psd)
        
        # 计算熵
        psd_norm = psd_norm[psd_norm > 0]  # 避免log(0)
        entropy = -np.sum(psd_norm * np.log2(psd_norm))
        
        return entropy
    
    @staticmethod
    def compute_spectral_flatness(signal: np.ndarray, fs: float) -> float:
        """
        谱平坦度（噪声度量）
        
        Args:
            signal: 信号
            fs: 采样频率
            
        Returns:
            谱平坦度 (0-1)
        """
        freqs, psd = FrequencyDomainFeatures.compute_psd(signal, fs)
        
        # 避免log(0)
        psd = psd[psd > 0]
        if len(psd) == 0:
            return 0
        
        geometric_mean = np.exp(np.mean(np.log(psd)))
        arithmetic_mean = np.mean(psd)
        
        if arithmetic_mean == 0:
            return 0
        
        return geometric_mean / arithmetic_mean
    
    @staticmethod
    def compute_spectral_rolloff(signal: np.ndarray, fs: float, 
                                 rolloff_ratio: float = 0.85) -> float:
        """
        谱滚降点（能量累积到指定比例的频率）
        
        Args:
            signal: 信号
            fs: 采样频率
            rolloff_ratio: 滚降比例
            
        Returns:
            滚降频率 (Hz)
        """
        freqs, psd = FrequencyDomainFeatures.compute_psd(signal, fs)
        
        cumsum_psd = np.cumsum(psd)
        threshold = cumsum_psd[-1] * rolloff_ratio
        
        rolloff_idx = np.where(cumsum_psd >= threshold)[0]
        if len(rolloff_idx) == 0:
            return freqs[-1]
        
        return freqs[rolloff_idx[0]]
    
    @staticmethod
    def extract_all(signal: np.ndarray, fs: float, 
                   freq_bands: List[Tuple[float, float]],
                   prefix: str = '') -> Dict[str, float]:
        """
        提取所有频域特征
        
        Args:
            signal: 信号
            fs: 采样频率
            freq_bands: 频带列表
            prefix: 特征名前缀
            
        Returns:
            特征字典
        """
        features = {}
        
        # 基本频域特征
        features[f'{prefix}dominant_freq'] = FrequencyDomainFeatures.compute_dominant_freq(signal, fs)
        features[f'{prefix}centroid_freq'] = FrequencyDomainFeatures.compute_centroid_freq(signal, fs)
        features[f'{prefix}bandwidth'] = FrequencyDomainFeatures.compute_bandwidth(signal, fs)
        
        # 频带能量
        for i, band in enumerate(freq_bands, 1):
            energy = FrequencyDomainFeatures.compute_band_energy(signal, fs, band)
            features[f'{prefix}band_energy_{i}'] = energy
        
        # 谱特性
        features[f'{prefix}spectral_entropy'] = FrequencyDomainFeatures.compute_spectral_entropy(signal, fs)
        features[f'{prefix}spectral_flatness'] = FrequencyDomainFeatures.compute_spectral_flatness(signal, fs)
        features[f'{prefix}spectral_rolloff'] = FrequencyDomainFeatures.compute_spectral_rolloff(signal, fs)
        
        return features
