"""
检查机器学习环境是否配置正确
"""

import sys

def check_environment():
    """检查环境配置"""
    print("=" * 70)
    print("环境检查")
    print("=" * 70)
    
    all_ok = True
    
    # 检查Python版本
    print(f"\n1. Python版本: {sys.version}")
    if sys.version_info < (3, 8):
        print("   ✗ Python版本过低，需要3.8+")
        all_ok = False
    else:
        print("   ✓ Python版本符合要求")
    
    # 检查基础库
    print("\n2. 基础库检查:")
    
    libraries = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'matplotlib': 'matplotlib',
        'sklearn': 'scikit-learn',
        'torch': 'torch'
    }
    
    for module_name, package_name in libraries.items():
        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'unknown')
            print(f"   ✓ {package_name}: {version}")
        except ImportError:
            print(f"   ✗ {package_name}: 未安装")
            print(f"     安装命令: pip install {package_name}")
            if package_name == 'torch':
                all_ok = False
    
    # 检查PyTorch和CUDA
    print("\n3. PyTorch和CUDA检查:")
    try:
        import torch
        print(f"   ✓ PyTorch版本: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"   ✓ CUDA可用")
            print(f"     GPU设备: {torch.cuda.get_device_name(0)}")
            print(f"     CUDA版本: {torch.version.cuda}")
            print(f"     显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            print("   ⚠ CUDA不可用，将使用CPU训练（速度较慢）")
            print("     如果有NVIDIA显卡，可以安装GPU版本的PyTorch")
            print("     访问 https://pytorch.org/ 获取安装命令")
    except ImportError:
        print("   ✗ PyTorch未安装")
        print("     安装命令: pip install torch")
        all_ok = False
    
    # 检查项目文件
    print("\n4. 项目文件检查:")
    import os
    required_files = [
        'trend_analyzer.py',
        'ml_models.py',
        'trend_based_predictor.py',
        'ml_training_pipeline.py'
    ]
    
    for file in required_files:
        if os.path.exists(file):
            print(f"   ✓ {file}")
        else:
            print(f"   ✗ {file}: 文件不存在")
            all_ok = False
    
    # 总结
    print("\n" + "=" * 70)
    if all_ok:
        print("✓ 环境检查通过！可以开始训练模型")
        print("\n下一步:")
        print("  1. 准备数据文件（CSV格式）")
        print("  2. 运行: python ml_training_pipeline.py --data your_data.csv")
        print("  或运行: python quick_start_ml.py")
    else:
        print("✗ 环境检查未通过，请先安装缺失的依赖")
        print("\n安装命令:")
        print("  pip install -r requirements_ml.txt")
    print("=" * 70)
    
    return all_ok

if __name__ == "__main__":
    check_environment()
