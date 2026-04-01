"""
MFCC特征提取模块（适配振动信号）
"""

import numpy as np
from scipy.fft import dct
from scipy.signal import get_window
from typing import Dict, Tuple


class MFCCFeatures:
    """MFCC特征提取器"""
    
    def __init__(self, fs: float, n_mfcc: int = 13, 
                 n_filters: int = 26, fft_size: int = 256,
                 hop_length: int = 128, freq_range: Tuple[float, float] = (0, 500)):
        """
        初始化MFCC提取器
        
        Args:
            fs: 采样频率
            n_mfcc: MFCC系数数量
            n_filters: Mel滤波器数量
            fft_size: FFT窗口大小
            hop_length: 帧移
            freq_range: 频率范围 (f_min, f_max)
        """
        self.fs = fs
        self.n_mfcc = n_mfcc
        self.n_filters = n_filters
        self.fft_size = fft_size
        self.hop_length = hop_length
        self.freq_range = freq_range
        
        # 创建Mel滤波器组
        self.mel_filterbank = self._create_mel_filterbank()
    
    def _hz_to_mel(self, hz: float) -> float:
        """Hz转Mel"""
        return 2595 * np.log10(1 + hz / 700)
    
    def _mel_to_hz(self, mel: float) -> float:
        """Mel转Hz"""
        return 700 * (10 ** (mel / 2595) - 1)
    
    def _create_mel_filterbank(self) -> np.ndarray:
        """
        创建Mel滤波器组
        
        Returns:
            滤波器组矩阵 (n_filters, fft_size//2 + 1)
        """
        f_min, f_max = self.freq_range
        
        # 转换到Mel尺度
        mel_min = self._hz_to_mel(f_min)
        mel_max = self._hz_to_mel(f_max)
        
        # 创建等间隔的Mel点
        mel_points = np.linspace(mel_min, mel_max, self.n_filters + 2)
        hz_points = np.array([self._mel_to_hz(m) for m in mel_points])
        
        # 转换为FFT bin
        bin_points = np.floor((self.fft_size + 1) * hz_points / self.fs).astype(int)
        
        # 创建滤波器组
        filterbank = np.zeros((self.n_filters, self.fft_size // 2 + 1))
        
        for i in range(1, self.n_filters + 1):
            left = bin_points[i - 1]
            center = bin_points[i]
            right = bin_points[i + 1]
            
            # 上升沿
            for j in range(left, center):
                if center > left:
                    filterbank[i - 1, j] = (j - left) / (center - left)
            
            # 下降沿
            for j in range(center, right):
                if right > center:
                    filterbank[i - 1, j] = (right - j) / (right - center)
        
        return filterbank
    
    def _stft(self, signal: np.ndarray) -> np.ndarray:
        """
        短时傅里叶变换
        
        Args:
            signal: 输入信号
            
        Returns:
            STFT幅度谱 (n_frames, fft_size//2 + 1)
        """
        # 计算帧数
        n_frames = 1 + (len(signal) - self.fft_size) // self.hop_length
        
        # 创建窗函数
        window = get_window('hann', self.fft_size)
        
        # 逐帧计算FFT
        spectrogram = []
        for i in range(n_frames):
            start = i * self.hop_length
            end = start + self.fft_size
            
            if end > len(signal):
                frame = np.pad(signal[start:], (0, end - len(signal)), mode='constant')
            else:
                frame = signal[start:end]
            
            # 加窗
            windowed = frame * window
            
            # FFT
            fft_result = np.fft.rfft(windowed, n=self.fft_size)
            magnitude = np.abs(fft_result)
            
            spectrogram.append(magnitude)
        
        return np.array(spectrogram)
    
    def compute_mfcc(self, signal: np.ndarray) -> np.ndarray:
        """
        计算MFCC特征
        
        Args:
            signal: 输入信号
            
        Returns:
            MFCC矩阵 (n_frames, n_mfcc)
        """
        # STFT
        spectrogram = self._stft(signal)
        
        # 应用Mel滤波器组
        mel_spectrogram = np.dot(spectrogram, self.mel_filterbank.T)
        
        # 取对数
        log_mel = np.log(mel_spectrogram + 1e-10)
        
        # DCT
        mfcc = dct(log_mel, type=2, axis=1, norm='ortho')[:, :self.n_mfcc]
        
        return mfcc
    
    def compute_statistics(self, mfcc: np.ndarray) -> Dict[str, float]:
        """
        计算MFCC统计特征
        
        Args:
            mfcc: MFCC矩阵
            
        Returns:
            统计特征字典
        """
        features = {}
        
        # 对每个系数计算统计量
        for i in range(mfcc.shape[1]):
            coeff = mfcc[:, i]
            features[f'mfcc_{i+1}_mean'] = np.mean(coeff)
            features[f'mfcc_{i+1}_std'] = np.std(coeff)
            features[f'mfcc_{i+1}_max'] = np.max(coeff)
            features[f'mfcc_{i+1}_min'] = np.min(coeff)
        
        return features
    
    def extract_all(self, signal: np.ndarray, prefix: str = '') -> Dict[str, float]:
        """
        提取所有MFCC特征
        
        Args:
            signal: 输入信号
            prefix: 特征名前缀
            
        Returns:
            特征字典
        """
        # 计算MFCC
        mfcc = self.compute_mfcc(signal)
        
        # 计算统计特征（使用均值作为代表）
        features = {}
        for i in range(mfcc.shape[1]):
            coeff_mean = np.mean(mfcc[:, i])
            features[f'{prefix}mfcc_{i+1}'] = coeff_mean
        
        return features
