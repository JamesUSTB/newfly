"""
生成示例数据用于测试
Generate sample data for testing purposes
"""

import numpy as np
import pandas as pd
from pathlib import Path
import os


def generate_vibration_signal(duration=10.0, fs=1000.0, speed=40, depth='5cm', position='left'):
    """
    生成模拟振动信号
    
    Args:
        duration: 持续时间（秒）
        fs: 采样频率
        speed: 速度 (km/h)
        depth: 埋深
        position: 位置（left/right）
        
    Returns:
        信号数组
    """
    n_samples = int(duration * fs)
    t = np.arange(n_samples) / fs
    
    # 基础噪声
    signal = np.random.normal(0, 0.01, n_samples)
    
    # 模拟车辆通过事件（在中间位置）
    event_time = duration / 2
    event_samples = int(fs * 1.0)  # 1秒事件
    event_start = int((event_time - 0.5) * fs)
    
    if event_start >= 0 and event_start + event_samples < n_samples:
        # 车辆通过频率（基于速度和轴距）
        vehicle_freq = (speed / 3.6) / 3.5  # 假设轴距3.5m
        
        # 主频响应（基于速度和深度）
        dominant_freq = vehicle_freq * (1 + np.random.uniform(-0.2, 0.2))
        
        # 幅值（受速度和位置影响）
        if position == 'right':
            amplitude = 0.3 * (speed / 40) * np.random.uniform(0.3, 0.7)  # 传荷效应
        else:
            amplitude = 0.5 * (speed / 40)
        
        if depth == '3.5cm':
            amplitude *= 1.5  # 浅埋深度响应更大
        
        # 生成事件信号
        event_t = np.arange(event_samples) / fs
        
        # 组合多个频率成分
        event_signal = (
            amplitude * np.exp(-event_t * 2) * np.sin(2 * np.pi * dominant_freq * event_t) +
            amplitude * 0.3 * np.sin(2 * np.pi * dominant_freq * 2 * event_t) +
            amplitude * 0.2 * np.sin(2 * np.pi * dominant_freq * 0.5 * event_t)
        )
        
        # 添加到信号中
        signal[event_start:event_start + event_samples] += event_signal
    
    return signal


def generate_experiment_data(speed, condition, file_idx, output_dir):
    """
    生成单个实验数据文件
    
    Args:
        speed: 速度
        condition: 工况
        file_idx: 文件序号
        output_dir: 输出目录
    """
    # 确定传感器配置
    sensors_5cm = ['acce6', 'acce7', 'acce8', 'acce9', 'acce10', 
                   'acce1', 'acce2', 'acce3', 'acce4', 'acce5']
    sensors_35cm = ['acce16', 'acce17', 'acce18', 'acce19', 'acce20',
                    'acce11', 'acce12', 'acce13', 'acce14', 'acce15']
    
    # 生成数据
    data = {}
    
    # 根据工况确定主要响应位置
    if condition == 'A':  # 左板
        primary_position = 'left'
        secondary_position = 'right'
    elif condition == 'B':  # 右板
        primary_position = 'right'
        secondary_position = 'right'
    else:  # 横跨
        primary_position = 'left'
        secondary_position = 'right'
    
    # 生成5cm深度传感器数据
    data['acce6'] = generate_vibration_signal(10.0, 1000.0, speed, '5cm', 'left')
    for sensor in sensors_5cm[1:]:
        data[sensor] = generate_vibration_signal(10.0, 1000.0, speed, '5cm', secondary_position)
    
    # 生成3.5cm深度传感器数据
    data['acce16'] = generate_vibration_signal(10.0, 1000.0, speed, '3.5cm', 'left')
    for sensor in sensors_35cm[1:]:
        data[sensor] = generate_vibration_signal(10.0, 1000.0, speed, '3.5cm', secondary_position)
    
    # 创建DataFrame
    df = pd.DataFrame(data)
    
    # 生成文件名（包含时间戳）
    timestamp = f"202311{20 + file_idx:02d}_{150000 + file_idx * 100:06d}"
    filename = f"data_speed{speed}_condition{condition}_{timestamp}.csv"
    
    # 保存
    filepath = output_dir / filename
    df.to_csv(filepath, index=False)
    
    print(f"Generated: {filename}")


def main():
    """主函数"""
    print("="*60)
    print("Generating Sample Data for Pavement Analysis")
    print("="*60)
    
    # 创建输出目录
    output_dir = Path('./data')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 实验参数
    speeds = [40, 60, 80, 100]
    conditions = ['A', 'B', 'C']
    n_files_per_condition = 3  # 每个工况生成3个文件
    
    # 生成数据
    for speed in speeds:
        for condition in conditions:
            for file_idx in range(n_files_per_condition):
                generate_experiment_data(speed, condition, file_idx, output_dir)
    
    print("\n" + "="*60)
    print("Sample Data Generation Completed!")
    print(f"Total files: {len(speeds) * len(conditions) * n_files_per_condition}")
    print(f"Output directory: {output_dir.absolute()}")
    print("="*60)


if __name__ == '__main__':
    main()
