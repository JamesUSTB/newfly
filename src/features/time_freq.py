"""
时频特征提取模块（小波分析）
"""

import numpy as np
import pywt
from typing import Dict, List


class TimeFrequencyFeatures:
    """时频特征提取器"""
    
    @staticmethod
    def compute_wavelet_decomposition(signal: np.ndarray, 
                                     wavelet: str = 'db4',
                                     level: int = 5) -> List[np.ndarray]:
        """
        小波分解
        
        Args:
            signal: 信号
            wavelet: 小波基
            level: 分解层数
            
        Returns:
            系数列表 [cA, cD_n, ..., cD_1]
        """
        coeffs = pywt.wavedec(signal, wavelet, level=level)
        return coeffs
    
    @staticmethod
    def compute_wavelet_energy(coeffs: List[np.ndarray]) -> List[float]:
        """
        计算各层小波能量
        
        Args:
            coeffs: 小波系数列表
            
        Returns:
            能量列表
        """
        energies = []
        for coeff in coeffs:
            energy = np.sum(coeff ** 2)
            energies.append(energy)
        return energies
    
    @staticmethod
    def compute_wavelet_entropy(signal: np.ndarray, 
                               wavelet: str = 'db4',
                               level: int = 5) -> float:
        """
        小波能量熵（信号复杂度）
        
        Args:
            signal: 信号
            wavelet: 小波基
            level: 分解层数
            
        Returns:
            小波熵
        """
        coeffs = TimeFrequencyFeatures.compute_wavelet_decomposition(signal, wavelet, level)
        energies = TimeFrequencyFeatures.compute_wavelet_energy(coeffs)
        
        # 归一化能量
        total_energy = sum(energies)
        if total_energy == 0:
            return 0
        
        probs = [e / total_energy for e in energies]
        
        # 计算熵
        entropy = 0
        for p in probs:
            if p > 0:
                entropy -= p * np.log2(p)
        
        return entropy
    
    @staticmethod
    def compute_detail_energy_ratio(signal: np.ndarray,
                                   wavelet: str = 'db4',
                                   level: int = 5) -> List[float]:
        """
        各层细节系数能量占比
        
        Args:
            signal: 信号
            wavelet: 小波基
            level: 分解层数
            
        Returns:
            能量占比列表
        """
        coeffs = TimeFrequencyFeatures.compute_wavelet_decomposition(signal, wavelet, level)
        energies = TimeFrequencyFeatures.compute_wavelet_energy(coeffs)
        
        total_energy = sum(energies)
        if total_energy == 0:
            return [0] * len(energies)
        
        ratios = [e / total_energy for e in energies]
        return ratios
    
    @staticmethod
    def compute_approximate_energy_ratio(signal: np.ndarray,
                                        wavelet: str = 'db4',
                                        level: int = 5) -> float:
        """
        近似系数能量占比
        
        Args:
            signal: 信号
            wavelet: 小波基
            level: 分解层数
            
        Returns:
            近似能量占比
        """
        ratios = TimeFrequencyFeatures.compute_detail_energy_ratio(signal, wavelet, level)
        return ratios[0] if ratios else 0
    
    @staticmethod
    def compute_wavelet_variance(coeffs: List[np.ndarray]) -> List[float]:
        """
        各层小波系数方差
        
        Args:
            coeffs: 小波系数列表
            
        Returns:
            方差列表
        """
        variances = []
        for coeff in coeffs:
            variance = np.var(coeff)
            variances.append(variance)
        return variances
    
    @staticmethod
    def compute_wavelet_std(coeffs: List[np.ndarray]) -> List[float]:
        """
        各层小波系数标准差
        
        Args:
            coeffs: 小波系数列表
            
        Returns:
            标准差列表
        """
        stds = []
        for coeff in coeffs:
            std = np.std(coeff)
            stds.append(std)
        return stds
    
    @staticmethod
    def extract_all(signal: np.ndarray, 
                   wavelet: str = 'db4',
                   level: int = 5,
                   prefix: str = '') -> Dict[str, float]:
        """
        提取所有时频特征
        
        Args:
            signal: 信号
            wavelet: 小波基
            level: 分解层数
            prefix: 特征名前缀
            
        Returns:
            特征字典
        """
        features = {}
        
        # 小波分解
        coeffs = TimeFrequencyFeatures.compute_wavelet_decomposition(signal, wavelet, level)
        
        # 小波熵
        features[f'{prefix}wavelet_entropy'] = TimeFrequencyFeatures.compute_wavelet_entropy(
            signal, wavelet, level
        )
        
        # 各层能量
        energies = TimeFrequencyFeatures.compute_wavelet_energy(coeffs)
        for i, energy in enumerate(energies):
            if i == 0:
                features[f'{prefix}wavelet_energy_a{level}'] = energy
            else:
                features[f'{prefix}wavelet_energy_d{level - i + 1}'] = energy
        
        # 能量占比
        ratios = TimeFrequencyFeatures.compute_detail_energy_ratio(signal, wavelet, level)
        for i, ratio in enumerate(ratios):
            if i == 0:
                features[f'{prefix}wavelet_energy_ratio_a{level}'] = ratio
            else:
                features[f'{prefix}wavelet_energy_ratio_d{level - i + 1}'] = ratio
        
        # 各层方差
        variances = TimeFrequencyFeatures.compute_wavelet_variance(coeffs)
        for i, var in enumerate(variances):
            if i == 0:
                features[f'{prefix}wavelet_var_a{level}'] = var
            else:
                features[f'{prefix}wavelet_var_d{level - i + 1}'] = var
        
        return features
