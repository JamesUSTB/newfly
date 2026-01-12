# 纵向切缝路面动力响应特征分析系统

## 项目概述

本项目实现了一套完整的路面动力响应特征分析系统，用于：
1. **特征分析**：量化纵向切缝对车载激励下路面振动响应的影响
2. **移动荷载识别**：基于PIRCN修正的LightGBM模型进行速度预测
3. **健康状态判别**：基于孤立森林的无监督异常检测

## 系统架构

```
src/
├── config.py                 # 全局配置参数
├── data/
│   ├── loader.py            # 数据加载（多文件拼接）
│   └── preprocessor.py      # 预处理（降噪、去趋势、平滑）
├── features/
│   ├── time_domain.py       # 时域特征
│   ├── freq_domain.py       # 频域特征
│   ├── time_freq.py         # 时频特征（小波）
│   ├── mfcc.py              # MFCC特征
│   └── cross_joint.py       # 跨缝特征
├── models/
│   ├── lightgbm_model.py    # LightGBM模型
│   ├── pircn.py             # PIRCN残差修正网络
│   └── isolation_forest.py  # 孤立森林模型
├── utils/
│   ├── visualization.py     # 可视化工具
│   └── evaluation.py        # 评估指标
└── main.py                   # 主程序入口
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 数据准备

### 数据格式要求

1. **文件格式**：CSV文件
2. **命名规范**：文件名应包含速度和工况信息，例如：
   - `data_speed40_conditionA_20231120_153045.csv`
   - `data_60kmh_B_001.csv`

3. **数据列**：必须包含传感器列（acce1-acce20）

### 数据目录结构

```
data/
├── speed40_conditionA_001.csv
├── speed40_conditionA_002.csv
├── speed40_conditionA_003.csv
├── speed60_conditionB_001.csv
├── ...
```

## 使用方法

### 1. 基本使用

```python
from src.main import PavementAnalysisPipeline

# 创建分析流程
pipeline = PavementAnalysisPipeline(
    data_dir='./data',
    output_dir='./output'
)

# 运行完整流程
pipeline.run_full_pipeline()
```

### 2. 逐步执行

```python
# 构建特征数据集
features_df, labels_df = pipeline.build_feature_dataset()

# 训练LightGBM模型
lgbm_models = pipeline.train_lightgbm_models()

# 训练PIRCN模型
pircn_model, history = pipeline.train_pircn_model(lgbm_models['regression'])

# 训练孤立森林
if_detector, if_results = pipeline.train_isolation_forest()
```

### 3. 自定义配置

修改 `src/config.py` 中的参数：

```python
# 事件检测参数
EVENT_DURATION = 1.0  # 事件片段长度（秒）
DETECTION_THRESHOLD_K = 5.0  # 阈值系数

# 频带划分
FREQ_BANDS = [(0, 10), (10, 50), (50, 150), (150, 500)]

# LightGBM参数
LGBM_PARAMS_CLASSIFICATION = {
    'learning_rate': 0.05,
    'num_leaves': 31,
    'n_estimators': 200
}
```

## 特征说明

### 时域特征（每通道）
- `peak`: 峰值响应
- `peak_to_peak`: 峰峰值
- `kurtosis`: 峭度（冲击性）
- `skewness`: 偏度（对称性）
- `rms`: 有效值
- `crest_factor`: 峰值因子
- `duration`: 有效持续时间

### 频域特征（每通道）
- `dominant_freq`: 主频
- `centroid_freq`: 质心频率
- `band_energy_1-4`: 各频带能量占比
- `spectral_entropy`: 谱熵
- `spectral_flatness`: 谱平坦度

### 时频特征（每通道）
- `wavelet_entropy`: 小波能量熵
- `wavelet_energy_d1-d5`: 各层细节系数能量
- `wavelet_energy_ratio_*`: 能量占比

### MFCC特征（每通道）
- `mfcc_1-13`: 13个MFCC系数

### 跨缝特征（核心指标）
- `LTE_5cm/3.5cm`: 传荷能力（5cm和3.5cm深度）
- `energy_transfer_*`: 能量传递率
- `xcorr_*`: 互相关系数（波形相似度）
- `phase_diff_*`: 相位差（波传播延迟）
- `time_lag_*`: 时间延迟
- `coherence_*`: 相干性

## 模型说明

### 1. LightGBM基础模型

**分类任务**：预测速度类别（40/60/80/100 km/h）
- 输出：4类速度等级
- 评估指标：准确率、F1分数、混淆矩阵

**回归任务**：预测连续速度值
- 输出：速度值（km/h）
- 评估指标：MAE、RMSE、R²、MAPE

### 2. PIRCN物理引导残差修正网络

结构：
```
输入特征 → LightGBM预测 → 残差计算 → 物理特征提取 → 残差修正网络 → 最终预测
```

物理约束：
1. 动力放大系数约束（峰值随速度增加）
2. 主频-速度关系约束（f ∝ v/L）
3. 传荷能力边界约束（0 < LTE < 1）

### 3. 孤立森林健康状态判别

**异常分数**：0-1范围，值越大越异常

**健康状态分级**：
- **健康**：异常分数 < 0.3
- **轻度异常**：0.3 ≤ 异常分数 < 0.5
- **中度异常**：0.5 ≤ 异常分数 < 0.7
- **重度异常**：异常分数 ≥ 0.7

## 输出结果

运行完成后，`output/` 目录包含：

```
output/
├── features.csv                    # 特征矩阵
├── labels.csv                      # 标签
├── lgbm_classification.txt         # LightGBM分类模型
├── lgbm_regression.txt            # LightGBM回归模型
├── scaler_classification.pkl       # 分类标准化器
├── scaler_regression.pkl          # 回归标准化器
├── pircn_residual.pth             # PIRCN残差网络
├── isolation_forest.pkl           # 孤立森林模型
└── if_scaler.pkl                  # 孤立森林标准化器
```

## 可视化

```python
from src.utils.visualization import Visualizer

viz = Visualizer()

# 绘制时域波形
viz.plot_time_series(time, signals, title='Time Series')

# 绘制频谱
viz.plot_frequency_spectrum(freqs, spectra, title='Frequency Spectrum')

# 绘制特征热图
viz.plot_feature_heatmap(features_df, title='Feature Heatmap')

# 绘制混淆矩阵
viz.plot_confusion_matrix(cm, labels, title='Confusion Matrix')

# 绘制特征重要性
viz.plot_feature_importance(importance_dict, top_k=20)

# 绘制异常分数分布
viz.plot_anomaly_distribution(anomaly_scores, health_states)
```

## 性能优化建议

1. **数据量大时**：使用批处理和并行计算
2. **内存不足时**：逐批处理数据文件
3. **训练慢时**：调整模型参数（减少n_estimators等）
4. **GPU加速**：PIRCN模型自动使用GPU（如果可用）

## 常见问题

### Q1: 找不到数据文件
**A**: 检查文件命名是否包含速度和工况信息，或修改 `loader.py` 中的文件匹配模式

### Q2: 特征提取失败
**A**: 确保信号长度足够（至少1秒数据），检查传感器列名是否正确

### Q3: 模型训练报错
**A**: 检查特征中是否有NaN或Inf值，可以在 `config.py` 中调整参数

### Q4: 异常检测结果异常
**A**: 调整 `contamination` 参数或重新选择特征

## 实验参数说明

- **采样频率**：1000 Hz
- **载重**：2t（标准两轴试验车）
- **速度等级**：40、60、80、100 km/h
- **工况**：A（左板）、B（右板）、C（横跨）
- **测点**：K6
- **埋深**：5cm和3.5cm两个深度

## 引用

如果使用本系统发表论文，请引用：

```
[待补充引用信息]
```

## 许可证

[待补充许可证信息]

## 联系方式

[待补充联系方式]

## 更新日志

### v1.0.0 (2024)
- 初始版本发布
- 实现完整的数据处理流程
- 实现LightGBM、PIRCN、孤立森林模型
- 实现可视化和评估工具
