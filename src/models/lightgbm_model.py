"""
LightGBM模型模块（分类和回归）
"""

import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, mean_absolute_error, mean_squared_error, r2_score
from typing import Dict, Tuple, Optional
import joblib


class LightGBMSpeedPredictor:
    """基于LightGBM的速度预测器"""
    
    def __init__(self, task: str = 'classification', params: Optional[Dict] = None):
        """
        初始化预测器
        
        Args:
            task: 任务类型 ('classification' 或 'regression')
            params: LightGBM参数
        """
        self.task = task
        self.params = params or {}
        self.model = None
        self.scaler = StandardScaler()
        self.feature_importance = None
        
    def train(self, X: np.ndarray, y: np.ndarray, 
             validation_split: float = 0.2) -> Dict[str, float]:
        """
        训练模型
        
        Args:
            X: 特征矩阵
            y: 标签
            validation_split: 验证集比例
            
        Returns:
            训练指标字典
        """
        # 分割数据
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=validation_split, random_state=42, stratify=y if self.task == 'classification' else None
        )
        
        # 标准化
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_val_scaled = self.scaler.transform(X_val)
        
        # 创建数据集
        train_data = lgb.Dataset(X_train_scaled, label=y_train)
        val_data = lgb.Dataset(X_val_scaled, label=y_val, reference=train_data)
        
        # 训练模型
        self.model = lgb.train(
            self.params,
            train_data,
            valid_sets=[train_data, val_data],
            valid_names=['train', 'valid'],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50),
                lgb.log_evaluation(period=0)
            ]
        )
        
        # 保存特征重要性
        self.feature_importance = self.model.feature_importance(importance_type='gain')
        
        # 评估
        metrics = self.evaluate(X_val, y_val)
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        预测
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测结果
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        predictions = self.model.predict(X_scaled)
        
        if self.task == 'classification':
            # 返回类别
            predictions = np.argmax(predictions, axis=1)
        
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        预测概率（仅分类任务）
        
        Args:
            X: 特征矩阵
            
        Returns:
            预测概率矩阵
        """
        if self.task != 'classification':
            raise ValueError("predict_proba only available for classification task")
        
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        X_scaled = self.scaler.transform(X)
        probabilities = self.model.predict(X_scaled)
        
        return probabilities
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            X: 特征矩阵
            y: 真实标签
            
        Returns:
            评估指标字典
        """
        predictions = self.predict(X)
        metrics = {}
        
        if self.task == 'classification':
            metrics['accuracy'] = accuracy_score(y, predictions)
            metrics['f1_macro'] = f1_score(y, predictions, average='macro')
            metrics['f1_weighted'] = f1_score(y, predictions, average='weighted')
            
            # 混淆矩阵
            cm = confusion_matrix(y, predictions)
            metrics['confusion_matrix'] = cm
            
        else:  # regression
            metrics['mae'] = mean_absolute_error(y, predictions)
            metrics['rmse'] = np.sqrt(mean_squared_error(y, predictions))
            metrics['r2'] = r2_score(y, predictions)
            
            # 平均相对误差
            mape = np.mean(np.abs((y - predictions) / (y + 1e-10))) * 100
            metrics['mape'] = mape
        
        return metrics
    
    def get_feature_importance(self, feature_names: Optional[list] = None) -> Dict[str, float]:
        """
        获取特征重要性
        
        Args:
            feature_names: 特征名称列表
            
        Returns:
            特征重要性字典
        """
        if self.feature_importance is None:
            raise ValueError("Model not trained yet")
        
        if feature_names is None:
            feature_names = [f'feature_{i}' for i in range(len(self.feature_importance))]
        
        importance_dict = dict(zip(feature_names, self.feature_importance))
        
        # 按重要性排序
        importance_dict = dict(sorted(importance_dict.items(), key=lambda x: x[1], reverse=True))
        
        return importance_dict
    
    def save(self, model_path: str, scaler_path: str):
        """
        保存模型
        
        Args:
            model_path: 模型保存路径
            scaler_path: 标准化器保存路径
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.save_model(model_path)
        joblib.dump(self.scaler, scaler_path)
    
    def load(self, model_path: str, scaler_path: str):
        """
        加载模型
        
        Args:
            model_path: 模型路径
            scaler_path: 标准化器路径
        """
        self.model = lgb.Booster(model_file=model_path)
        self.scaler = joblib.load(scaler_path)
    
    def cross_validate(self, X: np.ndarray, y: np.ndarray, cv: int = 5) -> Dict[str, float]:
        """
        交叉验证
        
        Args:
            X: 特征矩阵
            y: 标签
            cv: 折数
            
        Returns:
            交叉验证指标
        """
        X_scaled = self.scaler.fit_transform(X)
        
        if self.task == 'classification':
            model = lgb.LGBMClassifier(**self.params)
            scoring = 'accuracy'
        else:
            model = lgb.LGBMRegressor(**self.params)
            scoring = 'neg_mean_squared_error'
        
        scores = cross_val_score(model, X_scaled, y, cv=cv, scoring=scoring)
        
        return {
            'mean_score': np.mean(scores),
            'std_score': np.std(scores),
            'scores': scores
        }
