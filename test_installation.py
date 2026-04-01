"""
测试安装和环境配置
Test installation and environment setup
"""

import sys
import importlib

def test_import(module_name, package_name=None):
    """测试模块导入"""
    try:
        importlib.import_module(module_name)
        print(f"✓ {package_name or module_name} imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Failed to import {package_name or module_name}: {e}")
        return False

def test_installations():
    """测试所有依赖包"""
    print("="*60)
    print("Testing Package Installations")
    print("="*60)
    
    packages = [
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('scipy', 'SciPy'),
        ('sklearn', 'Scikit-learn'),
        ('lightgbm', 'LightGBM'),
        ('torch', 'PyTorch'),
        ('pywt', 'PyWavelets'),
        ('matplotlib', 'Matplotlib'),
        ('seaborn', 'Seaborn'),
        ('joblib', 'Joblib')
    ]
    
    results = []
    for module, name in packages:
        results.append(test_import(module, name))
    
    print("\n" + "="*60)
    if all(results):
        print("✓ All packages installed successfully!")
    else:
        print("✗ Some packages are missing. Please run:")
        print("  pip install -r requirements.txt")
    print("="*60)
    
    return all(results)

def test_project_structure():
    """测试项目结构"""
    print("\n" + "="*60)
    print("Testing Project Structure")
    print("="*60)
    
    from pathlib import Path
    
    required_files = [
        'src/config.py',
        'src/main.py',
        'src/data/loader.py',
        'src/data/preprocessor.py',
        'src/features/time_domain.py',
        'src/features/freq_domain.py',
        'src/features/time_freq.py',
        'src/features/mfcc.py',
        'src/features/cross_joint.py',
        'src/models/lightgbm_model.py',
        'src/models/pircn.py',
        'src/models/isolation_forest.py',
        'src/utils/visualization.py',
        'src/utils/evaluation.py'
    ]
    
    all_exist = True
    for file_path in required_files:
        path = Path(file_path)
        if path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ Missing: {file_path}")
            all_exist = False
    
    print("\n" + "="*60)
    if all_exist:
        print("✓ Project structure is complete!")
    else:
        print("✗ Some files are missing")
    print("="*60)
    
    return all_exist

def test_imports():
    """测试项目模块导入"""
    print("\n" + "="*60)
    print("Testing Project Module Imports")
    print("="*60)
    
    sys.path.insert(0, '.')
    
    modules = [
        'src.config',
        'src.data.loader',
        'src.data.preprocessor',
        'src.features.time_domain',
        'src.features.freq_domain',
        'src.features.time_freq',
        'src.features.mfcc',
        'src.features.cross_joint',
        'src.models.lightgbm_model',
        'src.models.pircn',
        'src.models.isolation_forest',
        'src.utils.visualization',
        'src.utils.evaluation'
    ]
    
    results = []
    for module in modules:
        results.append(test_import(module))
    
    print("\n" + "="*60)
    if all(results):
        print("✓ All project modules imported successfully!")
    else:
        print("✗ Some modules failed to import")
    print("="*60)
    
    return all(results)

def test_basic_functionality():
    """测试基本功能"""
    print("\n" + "="*60)
    print("Testing Basic Functionality")
    print("="*60)
    
    try:
        import numpy as np
        from src.features.time_domain import TimeDomainFeatures
        from src.features.freq_domain import FrequencyDomainFeatures
        
        # 生成测试信号
        fs = 1000.0
        t = np.linspace(0, 1, int(fs))
        signal = np.sin(2 * np.pi * 10 * t) + 0.1 * np.random.randn(len(t))
        
        # 测试时域特征
        time_features = TimeDomainFeatures.extract_all(signal, fs)
        print(f"✓ Time domain features: {len(time_features)} features extracted")
        
        # 测试频域特征
        freq_bands = [(0, 10), (10, 50), (50, 150), (150, 500)]
        freq_features = FrequencyDomainFeatures.extract_all(signal, fs, freq_bands)
        print(f"✓ Frequency domain features: {len(freq_features)} features extracted")
        
        print("\n" + "="*60)
        print("✓ Basic functionality test passed!")
        print("="*60)
        return True
        
    except Exception as e:
        print(f"\n✗ Basic functionality test failed: {e}")
        print("="*60)
        return False

def main():
    """主测试函数"""
    print("\n" + "="*60)
    print("Pavement Analysis System - Installation Test")
    print("="*60)
    
    results = []
    
    # 测试包安装
    results.append(test_installations())
    
    # 测试项目结构
    results.append(test_project_structure())
    
    # 测试模块导入
    results.append(test_imports())
    
    # 测试基本功能
    results.append(test_basic_functionality())
    
    # 总结
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    if all(results):
        print("✓✓✓ All tests passed! ✓✓✓")
        print("\nYou can now run the analysis pipeline:")
        print("  python generate_sample_data.py  # Generate test data")
        print("  cd src && python main.py        # Run analysis")
    else:
        print("✗✗✗ Some tests failed ✗✗✗")
        print("\nPlease fix the issues above before proceeding.")
    
    print("="*60)

if __name__ == '__main__':
    main()
