"""
主程序入口
纵向切缝路面动力响应特征分析
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# 导入配置
import config

# 导入数据处理模块
from data.loader import DataLoader
from data.preprocessor import SignalPreprocessor, EventDetector

# 导入特征提取模块
from features.time_domain import TimeDomainFeatures
from features.freq_domain import FrequencyDomainFeatures
from features.time_freq import TimeFrequencyFeatures
from features.mfcc import MFCCFeatures
from features.cross_joint import CrossJointFeatures

# 导入模型模块
from models.lightgbm_model import LightGBMSpeedPredictor
from models.pircn import PIRCNModel
from models.isolation_forest import HealthStateDetector

# 导入工具模块
from utils.visualization import Visualizer
from utils.evaluation import ModelEvaluator


class PavementAnalysisPipeline:
    """路面动力响应分析流程"""
    
    def __init__(self, data_dir: str, output_dir: str):
        """
        初始化分析流程
        
        Args:
            data_dir: 数据目录
            output_dir: 输出目录
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 初始化组件
        self.loader = DataLoader(data_dir)
        self.preprocessor = SignalPreprocessor(config.FS, vars(config))
        self.detector = EventDetector(config.FS, vars(config))
        self.visualizer = Visualizer(config.FIG_SIZE, config.DPI)
        self.evaluator = ModelEvaluator()
        
        # MFCC提取器
        self.mfcc_extractor = MFCCFeatures(
            fs=config.FS,
            n_mfcc=config.MFCC_N_COEFFS,
            n_filters=config.MFCC_N_FILTERS,
            fft_size=config.MFCC_FFT_SIZE,
            hop_length=config.MFCC_HOP_LENGTH,
            freq_range=config.MFCC_FREQ_RANGE
        )
        
        # 存储结果
        self.features_df = None
        self.labels_df = None
        
    def load_and_preprocess_data(self, speed: int, condition: str) -> pd.DataFrame:
        """
        加载和预处理数据
        
        Args:
            speed: 速度
            condition: 工况
            
        Returns:
            预处理后的DataFrame
        """
        print(f"Loading data: speed={speed}, condition={condition}")
        
        # 加载数据
        df = self.loader.load_experiment_data(speed, condition)
        
        # 获取所有传感器列
        sensor_columns = [col for col in df.columns if 'acce' in col.lower()]
        
        # 验证数据
        if not self.loader.validate_data(df, sensor_columns):
            print("Warning: Data validation failed")
        
        # 预处理
        print("Preprocessing signals...")
        df_processed = self.preprocessor.preprocess_dataframe(df, sensor_columns)
        
        return df_processed
    
    def extract_features_from_segment(self, segment: pd.DataFrame, 
                                     sensor_columns: list) -> Dict:
        """
        从事件片段提取特征
        
        Args:
            segment: 事件片段DataFrame
            sensor_columns: 传感器列名
            
        Returns:
            特征字典
        """
        features = {}
        
        # 对每个传感器提取特征
        for sensor in sensor_columns:
            if sensor not in segment.columns:
                continue
            
            signal = segment[sensor].values
            prefix = f'{sensor}_'
            
            # 时域特征
            time_features = TimeDomainFeatures.extract_all(signal, config.FS, prefix)
            features.update(time_features)
            
            # 频域特征
            freq_features = FrequencyDomainFeatures.extract_all(
                signal, config.FS, config.FREQ_BANDS, prefix
            )
            features.update(freq_features)
            
            # 时频特征
            timefreq_features = TimeFrequencyFeatures.extract_all(
                signal, config.WAVELET_NAME, config.WAVELET_LEVEL, prefix
            )
            features.update(timefreq_features)
            
            # MFCC特征
            mfcc_features = self.mfcc_extractor.extract_all(signal, prefix)
            features.update(mfcc_features)
        
        # 跨缝特征
        if 'acce6' in segment.columns and 'acce7' in segment.columns:
            cross_features_5cm = CrossJointFeatures.extract_all(
                segment['acce6'].values,
                segment['acce7'].values,
                config.FS,
                '5cm'
            )
            features.update(cross_features_5cm)
        
        if 'acce16' in segment.columns and 'acce17' in segment.columns:
            cross_features_35cm = CrossJointFeatures.extract_all(
                segment['acce16'].values,
                segment['acce17'].values,
                config.FS,
                '3.5cm'
            )
            features.update(cross_features_35cm)
        
        return features
    
    def build_feature_dataset(self):
        """构建完整特征数据集"""
        print("\n" + "="*60)
        print("Building Feature Dataset")
        print("="*60)
        
        all_features = []
        all_labels = []
        
        # 遍历所有实验条件
        for speed in config.SPEEDS:
            for condition in config.CONDITIONS:
                try:
                    # 加载和预处理
                    df = self.load_and_preprocess_data(speed, condition)
                    
                    # 获取传感器列
                    sensor_columns = [col for col in df.columns if 'acce' in col.lower()]
                    
                    # 事件检测和片段提取
                    print(f"Detecting events...")
                    segments = self.detector.extract_segments(
                        df, sensor_columns, config.EVENT_DURATION
                    )
                    
                    print(f"Found {len(segments)} events")
                    
                    # 提取特征
                    for segment, peak_pos in segments:
                        features = self.extract_features_from_segment(segment, sensor_columns)
                        all_features.append(features)
                        
                        # 标签
                        label = {
                            'speed': speed,
                            'condition': condition,
                            'speed_class': config.SPEEDS.index(speed)
                        }
                        all_labels.append(label)
                    
                except Exception as e:
                    print(f"Error processing speed={speed}, condition={condition}: {e}")
                    continue
        
        # 转换为DataFrame
        self.features_df = pd.DataFrame(all_features)
        self.labels_df = pd.DataFrame(all_labels)
        
        print(f"\nDataset built: {len(self.features_df)} samples, {len(self.features_df.columns)} features")
        
        # 保存
        self.features_df.to_csv(self.output_dir / 'features.csv', index=False)
        self.labels_df.to_csv(self.output_dir / 'labels.csv', index=False)
        
        return self.features_df, self.labels_df
    
    def train_lightgbm_models(self):
        """训练LightGBM模型（分类和回归）"""
        print("\n" + "="*60)
        print("Training LightGBM Models")
        print("="*60)
        
        if self.features_df is None:
            print("Error: Feature dataset not built")
            return
        
        X = self.features_df.values
        y_class = self.labels_df['speed_class'].values
        y_reg = self.labels_df['speed'].values
        
        results = {}
        
        # 分类任务
        print("\n--- Classification Task ---")
        clf_model = LightGBMSpeedPredictor(
            task='classification',
            params=config.LGBM_PARAMS_CLASSIFICATION
        )
        clf_metrics = clf_model.train(X, y_class)
        print(f"Classification Accuracy: {clf_metrics['accuracy']:.4f}")
        results['classification'] = clf_model
        
        # 回归任务
        print("\n--- Regression Task ---")
        reg_model = LightGBMSpeedPredictor(
            task='regression',
            params=config.LGBM_PARAMS_REGRESSION
        )
        reg_metrics = reg_model.train(X, y_reg)
        print(f"Regression RMSE: {reg_metrics['rmse']:.4f}")
        results['regression'] = reg_model
        
        # 保存模型
        clf_model.save(
            str(self.output_dir / 'lgbm_classification.txt'),
            str(self.output_dir / 'scaler_classification.pkl')
        )
        reg_model.save(
            str(self.output_dir / 'lgbm_regression.txt'),
            str(self.output_dir / 'scaler_regression.pkl')
        )
        
        return results
    
    def train_pircn_model(self, base_model):
        """训练PIRCN模型"""
        print("\n" + "="*60)
        print("Training PIRCN Model")
        print("="*60)
        
        if self.features_df is None:
            print("Error: Feature dataset not built")
            return
        
        X = self.features_df.values
        y_reg = self.labels_df['speed'].values
        
        # 创建PIRCN模型
        pircn = PIRCNModel(base_model, config.PIRCN_PARAMS)
        
        # 训练
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y_reg, test_size=0.2, random_state=42
        )
        
        history = pircn.train(X_train, y_train, X_val, y_val)
        
        # 评估
        y_pred = pircn.predict(X_val)
        metrics = self.evaluator.evaluate_regression(y_val, y_pred)
        print(f"PIRCN RMSE: {metrics['rmse']:.4f}")
        
        # 保存
        pircn.save(str(self.output_dir / 'pircn_residual.pth'))
        
        return pircn, history
    
    def train_isolation_forest(self):
        """训练孤立森林模型"""
        print("\n" + "="*60)
        print("Training Isolation Forest")
        print("="*60)
        
        if self.features_df is None:
            print("Error: Feature dataset not built")
            return
        
        X = self.features_df.values
        feature_names = list(self.features_df.columns)
        
        # 创建健康状态检测器
        detector = HealthStateDetector(config.IFOREST_PARAMS)
        
        # 训练
        detector.fit(X, feature_names)
        
        # 分析
        results = detector.analyze_batch(X)
        
        print(f"\nAnomaly Score Statistics:")
        print(f"Mean: {results['mean_score']:.4f}")
        print(f"Std:  {results['std_score']:.4f}")
        print(f"\nHealth State Distribution:")
        for state, count in results['state_distribution'].items():
            print(f"{state}: {count}")
        
        # 保存
        detector.save(
            str(self.output_dir / 'isolation_forest.pkl'),
            str(self.output_dir / 'if_scaler.pkl')
        )
        
        return detector, results
    
    def run_full_pipeline(self):
        """运行完整流程"""
        print("\n" + "="*60)
        print("Pavement Dynamics Analysis Pipeline")
        print("="*60)
        
        # 1. 构建特征数据集
        self.build_feature_dataset()
        
        # 2. 训练LightGBM模型
        lgbm_models = self.train_lightgbm_models()
        
        # 3. 训练PIRCN模型
        if 'regression' in lgbm_models:
            pircn_model, history = self.train_pircn_model(lgbm_models['regression'])
        
        # 4. 训练孤立森林
        if_detector, if_results = self.train_isolation_forest()
        
        print("\n" + "="*60)
        print("Pipeline Completed Successfully!")
        print("="*60)


def main():
    """主函数"""
    # 配置路径
    data_dir = './data'  # 数据目录
    output_dir = './output'  # 输出目录
    
    # 创建并运行流程
    pipeline = PavementAnalysisPipeline(data_dir, output_dir)
    pipeline.run_full_pipeline()


if __name__ == '__main__':
    main()
