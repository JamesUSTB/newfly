"""
可视化工具模块
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple
from matplotlib.gridspec import GridSpec


class Visualizer:
    """可视化工具类"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 6), dpi: int = 300):
        """
        初始化
        
        Args:
            figsize: 图形大小
            dpi: 分辨率
        """
        self.figsize = figsize
        self.dpi = dpi
        plt.style.use('seaborn-v0_8-darkgrid')
    
    def plot_time_series(self, time: np.ndarray, signals: Dict[str, np.ndarray],
                        title: str = 'Time Series', save_path: Optional[str] = None):
        """
        绘制时域波形
        
        Args:
            time: 时间轴
            signals: 信号字典 {名称: 信号数组}
            title: 标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(len(signals), 1, figsize=(self.figsize[0], self.figsize[1] * len(signals) / 2))
        
        if len(signals) == 1:
            axes = [axes]
        
        for ax, (name, signal) in zip(axes, signals.items()):
            ax.plot(time, signal, linewidth=0.5)
            ax.set_ylabel('Acceleration (g)')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
        
        axes[-1].set_xlabel('Time (s)')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_frequency_spectrum(self, freqs: np.ndarray, spectra: Dict[str, np.ndarray],
                               title: str = 'Frequency Spectrum', 
                               save_path: Optional[str] = None):
        """
        绘制频谱
        
        Args:
            freqs: 频率轴
            spectra: 频谱字典 {名称: 频谱数组}
            title: 标题
            save_path: 保存路径
        """
        fig, axes = plt.subplots(len(spectra), 1, figsize=(self.figsize[0], self.figsize[1] * len(spectra) / 2))
        
        if len(spectra) == 1:
            axes = [axes]
        
        for ax, (name, spectrum) in zip(axes, spectra.items()):
            ax.semilogy(freqs, spectrum, linewidth=1)
            ax.set_ylabel('PSD')
            ax.set_title(name)
            ax.grid(True, alpha=0.3)
            ax.set_xlim([0, 200])  # 关注0-200Hz
        
        axes[-1].set_xlabel('Frequency (Hz)')
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_heatmap(self, features: pd.DataFrame, 
                            title: str = 'Feature Heatmap',
                            save_path: Optional[str] = None):
        """
        绘制特征热图
        
        Args:
            features: 特征DataFrame
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(self.figsize[0], self.figsize[1] * 1.5))
        
        # 标准化特征
        features_normalized = (features - features.mean()) / features.std()
        
        sns.heatmap(features_normalized.T, cmap='RdBu_r', center=0, 
                   cbar_kws={'label': 'Normalized Value'},
                   ax=ax, linewidths=0.5, linecolor='gray')
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Sample Index')
        ax.set_ylabel('Features')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_confusion_matrix(self, cm: np.ndarray, labels: List[str],
                             title: str = 'Confusion Matrix',
                             save_path: Optional[str] = None):
        """
        绘制混淆矩阵
        
        Args:
            cm: 混淆矩阵
            labels: 类别标签
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=(8, 6))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels, ax=ax)
        
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel('Predicted Label')
        ax.set_ylabel('True Label')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, importance_dict: Dict[str, float],
                               top_k: int = 20,
                               title: str = 'Feature Importance',
                               save_path: Optional[str] = None):
        """
        绘制特征重要性
        
        Args:
            importance_dict: 特征重要性字典
            top_k: 显示前k个特征
            title: 标题
            save_path: 保存路径
        """
        # 选择前k个特征
        sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_k]
        features, importances = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(10, max(6, top_k * 0.3)))
        
        y_pos = np.arange(len(features))
        ax.barh(y_pos, importances, align='center')
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='x', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]],
                             metric: str = 'accuracy',
                             title: str = 'Model Comparison',
                             save_path: Optional[str] = None):
        """
        绘制模型对比
        
        Args:
            results: 结果字典 {模型名: {指标名: 值}}
            metric: 对比指标
            title: 标题
            save_path: 保存路径
        """
        models = list(results.keys())
        values = [results[model].get(metric, 0) for model in models]
        
        fig, ax = plt.subplots(figsize=(8, 6))
        
        bars = ax.bar(models, values, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, axis='y', alpha=0.3)
        
        # 在柱子上显示数值
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}',
                   ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_anomaly_distribution(self, anomaly_scores: np.ndarray,
                                  health_states: List[str],
                                  title: str = 'Anomaly Score Distribution',
                                  save_path: Optional[str] = None):
        """
        绘制异常分数分布
        
        Args:
            anomaly_scores: 异常分数数组
            health_states: 健康状态列表
            title: 标题
            save_path: 保存路径
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(self.figsize[0], self.figsize[1] * 0.7))
        
        # 直方图
        ax1.hist(anomaly_scores, bins=30, edgecolor='black', alpha=0.7)
        ax1.set_xlabel('Anomaly Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('Score Distribution')
        ax1.grid(True, alpha=0.3)
        
        # 健康状态饼图
        unique_states, counts = np.unique(health_states, return_counts=True)
        colors = {'Healthy': '#2ca02c', 'Mild Anomaly': '#ff7f0e', 
                 'Moderate Anomaly': '#d62728', 'Severe Anomaly': '#8c564b'}
        pie_colors = [colors.get(state, '#7f7f7f') for state in unique_states]
        
        ax2.pie(counts, labels=unique_states, autopct='%1.1f%%', colors=pie_colors)
        ax2.set_title('Health State Distribution')
        
        fig.suptitle(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
    
    def plot_training_history(self, history: Dict[str, List[float]],
                             title: str = 'Training History',
                             save_path: Optional[str] = None):
        """
        绘制训练历史
        
        Args:
            history: 训练历史字典
            title: 标题
            save_path: 保存路径
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        for key, values in history.items():
            ax.plot(values, label=key, linewidth=2)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()
