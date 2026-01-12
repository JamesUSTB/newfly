"""
全局配置参数
纵向切缝路面动力响应特征分析
"""

# 采样参数
FS = 1000.0  # 采样频率 (Hz)

# 传感器配置
SENSOR_CONFIG = {
    'left_5cm': ['acce6'],
    'left_3.5cm': ['acce16'],
    'right_5cm': ['acce7', 'acce8', 'acce9', 'acce10', 'acce1', 'acce2', 'acce3', 'acce4', 'acce5'],
    'right_3.5cm': ['acce17', 'acce18', 'acce19', 'acce20', 'acce11', 'acce12', 'acce13', 'acce14', 'acce15'],
    'cross_joint_5cm': ('acce6', 'acce7'),
    'cross_joint_3.5cm': ('acce16', 'acce17')
}

# 实验参数
SPEEDS = [40, 60, 80, 100]  # km/h
WEIGHT = 2000  # kg
CONDITIONS = ['A', 'B', 'C']  # 左板、右板、横跨

# 预处理参数
DENOISE_WAVELET = 'db4'      # 降噪小波基
DENOISE_LEVEL = 5            # 分解层数
HIGHPASS_CUTOFF = 0.5        # 高通截止频率 (Hz)
DETREND_ORDER = 2            # 去趋势多项式阶数
SMOOTH_WINDOW = 11           # 平滑窗口长度
SMOOTH_ORDER = 3             # 平滑多项式阶数

# 事件检测参数
EVENT_DURATION = 1.0         # 事件片段长度 (秒) ★可配置★
ENERGY_WINDOW = 0.05         # 能量窗口 (秒)
DETECTION_THRESHOLD_K = 5.0  # 阈值系数 (k倍标准差)

# 特征参数
FREQ_BANDS = [(0, 10), (10, 50), (50, 150), (150, 500)]  # 频带划分 (Hz)
MFCC_FREQ_RANGE = (0, 500)   # MFCC频率范围 (Hz)
MFCC_N_COEFFS = 13           # MFCC系数数量
MFCC_N_FILTERS = 26          # Mel滤波器数量
MFCC_FFT_SIZE = 256          # FFT窗口大小
MFCC_HOP_LENGTH = 128        # 帧移

# 小波分析参数
WAVELET_NAME = 'db4'
WAVELET_LEVEL = 5

# 模型参数
LGBM_PARAMS_CLASSIFICATION = {
    'objective': 'multiclass',
    'num_class': 4,
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'n_estimators': 200,
    'metric': 'multi_logloss',
    'verbose': -1,
    'random_state': 42
}

LGBM_PARAMS_REGRESSION = {
    'objective': 'regression',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': -1,
    'n_estimators': 200,
    'metric': 'rmse',
    'verbose': -1,
    'random_state': 42
}

PIRCN_PARAMS = {
    'hidden_dims': [64, 32, 16],
    'learning_rate': 0.001,
    'epochs': 100,
    'batch_size': 32,
    'physics_weight': 0.1,      # λ1: 物理约束权重
    'smooth_weight': 0.01,      # λ2: 平滑约束权重
    'early_stopping_patience': 10,
    'random_state': 42
}

IFOREST_PARAMS = {
    'n_estimators': 100,
    'contamination': 0.1,
    'max_samples': 'auto',
    'random_state': 42
}

# 健康状态阈值
HEALTH_THRESHOLDS = {
    'healthy': 0.3,
    'mild': 0.5,
    'moderate': 0.7
}

# 数据增强参数
AUGMENTATION = {
    'enabled': True,
    'noise_level': 0.01,
    'time_stretch_factors': [0.9, 1.1],
    'window_shift': 0.1  # 秒
}

# 训练参数
TRAIN_TEST_SPLIT = 0.2
VALIDATION_SPLIT = 0.1
RANDOM_STATE = 42

# 可视化参数
FIG_SIZE = (12, 6)
DPI = 300
COLORMAP = 'viridis'
