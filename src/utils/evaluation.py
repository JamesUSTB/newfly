"""
评估指标模块
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report,
    mean_absolute_error, mean_squared_error, r2_score
)
from typing import Dict, Optional


class ModelEvaluator:
    """模型评估器"""
    
    @staticmethod
    def evaluate_classification(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               labels: Optional[list] = None) -> Dict:
        """
        评估分类模型
        
        Args:
            y_true: 真实标签
            y_pred: 预测标签
            labels: 类别标签列表
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 基本指标
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0)
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 混淆矩阵
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        
        # 分类报告
        if labels:
            report = classification_report(y_true, y_pred, target_names=labels, output_dict=True, zero_division=0)
        else:
            report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
        metrics['classification_report'] = report
        
        return metrics
    
    @staticmethod
    def evaluate_regression(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
        """
        评估回归模型
        
        Args:
            y_true: 真实值
            y_pred: 预测值
            
        Returns:
            评估指标字典
        """
        metrics = {}
        
        # 基本指标
        metrics['mae'] = mean_absolute_error(y_true, y_pred)
        metrics['mse'] = mean_squared_error(y_true, y_pred)
        metrics['rmse'] = np.sqrt(mean_squared_error(y_true, y_pred))
        metrics['r2'] = r2_score(y_true, y_pred)
        
        # 平均绝对百分比误差
        mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-10))) * 100
        metrics['mape'] = mape
        
        # 最大误差
        metrics['max_error'] = np.max(np.abs(y_true - y_pred))
        
        # 中位数绝对误差
        metrics['median_ae'] = np.median(np.abs(y_true - y_pred))
        
        return metrics
    
    @staticmethod
    def compare_models(results: Dict[str, Dict]) -> Dict:
        """
        比较多个模型
        
        Args:
            results: 模型结果字典 {模型名: 评估指标}
            
        Returns:
            比较结果
        """
        comparison = {}
        
        # 收集所有指标
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        
        # 排除非数值指标
        numeric_metrics = [m for m in all_metrics 
                         if not isinstance(list(results.values())[0].get(m), (dict, np.ndarray, list))]
        
        # 对每个指标进行比较
        for metric in numeric_metrics:
            comparison[metric] = {}
            for model_name, model_results in results.items():
                if metric in model_results:
                    comparison[metric][model_name] = model_results[metric]
            
            # 找到最佳模型
            if comparison[metric]:
                if metric in ['mae', 'mse', 'rmse', 'mape', 'max_error']:
                    # 越小越好
                    best_model = min(comparison[metric].items(), key=lambda x: x[1])[0]
                else:
                    # 越大越好
                    best_model = max(comparison[metric].items(), key=lambda x: x[1])[0]
                
                comparison[metric]['best_model'] = best_model
        
        return comparison
    
    @staticmethod
    def compute_speed_prediction_accuracy(y_true: np.ndarray, 
                                         y_pred: np.ndarray,
                                         tolerance: float = 5.0) -> Dict:
        """
        计算速度预测精度（针对回归任务）
        
        Args:
            y_true: 真实速度
            y_pred: 预测速度
            tolerance: 容许误差 (km/h)
            
        Returns:
            精度指标
        """
        errors = np.abs(y_true - y_pred)
        
        metrics = {}
        metrics['within_tolerance'] = np.mean(errors <= tolerance) * 100  # 百分比
        metrics['mean_error'] = np.mean(errors)
        metrics['std_error'] = np.std(errors)
        metrics['median_error'] = np.median(errors)
        metrics['percentile_95_error'] = np.percentile(errors, 95)
        
        return metrics
    
    @staticmethod
    def print_evaluation_report(metrics: Dict, task: str = 'classification'):
        """
        打印评估报告
        
        Args:
            metrics: 评估指标字典
            task: 任务类型
        """
        print("\n" + "="*50)
        print(f"{'Evaluation Report':^50}")
        print("="*50)
        
        if task == 'classification':
            print(f"\nAccuracy:          {metrics['accuracy']:.4f}")
            print(f"Precision (macro): {metrics['precision_macro']:.4f}")
            print(f"Recall (macro):    {metrics['recall_macro']:.4f}")
            print(f"F1-Score (macro):  {metrics['f1_macro']:.4f}")
            
            print("\nConfusion Matrix:")
            print(metrics['confusion_matrix'])
            
        elif task == 'regression':
            print(f"\nMAE:   {metrics['mae']:.4f}")
            print(f"RMSE:  {metrics['rmse']:.4f}")
            print(f"R²:    {metrics['r2']:.4f}")
            print(f"MAPE:  {metrics['mape']:.2f}%")
        
        print("\n" + "="*50 + "\n")
    
    @staticmethod
    def compute_cross_validation_summary(cv_results: Dict) -> Dict:
        """
        汇总交叉验证结果
        
        Args:
            cv_results: 交叉验证结果
            
        Returns:
            汇总统计
        """
        summary = {
            'mean': cv_results['mean_score'],
            'std': cv_results['std_score'],
            'min': np.min(cv_results['scores']),
            'max': np.max(cv_results['scores']),
            'scores': cv_results['scores']
        }
        
        return summary
