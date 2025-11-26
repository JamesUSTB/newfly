# 快速开始指南

## 1. 环境准备

### 安装依赖

```bash
pip install -r requirements.txt
```

## 2. 生成示例数据（用于测试）

如果你没有真实数据，可以生成示例数据：

```bash
python generate_sample_data.py
```

这将在 `data/` 目录下生成模拟的振动数据文件。

## 3. 运行完整分析流程

```bash
cd src
python main.py
```

或者在Python中：

```python
from src.main import PavementAnalysisPipeline

pipeline = PavementAnalysisPipeline(
    data_dir='./data',
    output_dir='./output'
)

pipeline.run_full_pipeline()
```

## 4. 查看结果

结果将保存在 `output/` 目录：

- `features.csv` - 提取的特征矩阵
- `labels.csv` - 对应的标签
- `lgbm_*.txt` - LightGBM模型
- `pircn_residual.pth` - PIRCN残差修正网络
- `isolation_forest.pkl` - 孤立森林健康检测模型

## 5. 自定义配置

编辑 `src/config.py` 修改参数：

```python
# 例如：修改事件检测窗口长度
EVENT_DURATION = 1.5  # 从1.0秒改为1.5秒

# 修改频带划分
FREQ_BANDS = [(0, 15), (15, 60), (60, 200), (200, 500)]

# 修改LightGBM参数
LGBM_PARAMS_CLASSIFICATION['n_estimators'] = 300
```

## 6. 分步执行

如果想要更细粒度的控制：

```python
from src.main import PavementAnalysisPipeline

pipeline = PavementAnalysisPipeline('./data', './output')

# 步骤1: 构建特征数据集
features_df, labels_df = pipeline.build_feature_dataset()
print(f"Features shape: {features_df.shape}")

# 步骤2: 训练LightGBM
lgbm_models = pipeline.train_lightgbm_models()

# 步骤3: 训练PIRCN（可选）
if 'regression' in lgbm_models:
    pircn, history = pipeline.train_pircn_model(lgbm_models['regression'])

# 步骤4: 健康状态检测
detector, results = pipeline.train_isolation_forest()
print(f"Anomaly detection: {results['state_distribution']}")
```

## 7. 使用已训练模型

```python
from src.models.lightgbm_model import LightGBMSpeedPredictor
import numpy as np

# 加载模型
model = LightGBMSpeedPredictor(task='regression')
model.load('./output/lgbm_regression.txt', './output/scaler_regression.pkl')

# 预测
X_new = np.random.randn(10, n_features)  # 新数据
predictions = model.predict(X_new)
print(f"Predicted speeds: {predictions}")
```

## 8. 可视化

```python
from src.utils.visualization import Visualizer
import pandas as pd

viz = Visualizer()

# 加载特征
features = pd.read_csv('./output/features.csv')

# 绘制特征热图
viz.plot_feature_heatmap(features.iloc[:50, :20], 
                         title='Feature Heatmap',
                         save_path='./output/heatmap.png')
```

## 常见问题

### Q: 如何处理真实数据？

确保CSV文件包含：
- 文件名包含速度和工况信息
- 列名包含传感器标识（acce1-acce20）
- 采样频率为1000Hz

### Q: 如何调整内存使用？

如果内存不足，可以：
1. 减少 `EVENT_DURATION`（处理更短的片段）
2. 减少特征数量（注释掉某些特征提取代码）
3. 批量处理数据文件

### Q: 如何使用GPU加速？

PIRCN模型会自动检测并使用GPU。确保安装了正确的PyTorch版本：

```bash
# CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu118

# CUDA 12.1
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

## 进阶使用

### 特征选择

```python
from sklearn.feature_selection import SelectKBest, f_classif

# 选择最重要的k个特征
selector = SelectKBest(f_classif, k=50)
X_selected = selector.fit_transform(X, y)
```

### 超参数调优

```python
from sklearn.model_selection import GridSearchCV
import lightgbm as lgb

param_grid = {
    'num_leaves': [20, 31, 50],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300]
}

model = lgb.LGBMClassifier()
grid_search = GridSearchCV(model, param_grid, cv=5)
grid_search.fit(X, y)
print(f"Best params: {grid_search.best_params_}")
```

### 数据增强

```python
# 在config.py中启用
AUGMENTATION = {
    'enabled': True,
    'noise_level': 0.02,
    'time_stretch_factors': [0.8, 0.9, 1.1, 1.2],
    'window_shift': 0.2
}
```

## 支持

如遇问题，请检查：
1. 依赖包版本是否正确
2. 数据格式是否符合要求
3. 配置参数是否合理

更多信息请参考完整的 `README.md` 文档。
