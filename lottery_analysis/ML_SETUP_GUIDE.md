# 机器学习环境配置指南

## 一、硬件要求

### 最低配置（CPU训练）
- **CPU**: 4核心以上（推荐8核心）
- **内存**: 8GB RAM（推荐16GB）
- **存储**: 至少10GB可用空间
- **训练时间**: 较慢（可能需要数小时）

### 推荐配置（GPU训练）
- **CPU**: 6核心以上
- **GPU**: NVIDIA显卡，支持CUDA（如GTX 1060, RTX 2060等）
- **显存**: 4GB以上（推荐8GB+）
- **内存**: 16GB RAM（推荐32GB）
- **存储**: 至少20GB可用空间（用于数据和模型）
- **训练时间**: 快（通常几分钟到半小时）

### 云平台选项（无需本地硬件）
- **Google Colab**: 免费GPU（有限制）
- **Kaggle**: 免费GPU（每周30小时）
- **AWS/GCP/Azure**: 付费GPU实例

## 二、软件环境

### 1. Python版本
- **Python 3.8+**（推荐3.9或3.10）

### 2. 必需库
```bash
# 基础库
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.4.0
scikit-learn>=0.24.0

# 深度学习库
torch>=1.9.0          # PyTorch
torchvision>=0.10.0   # 图像处理（如果使用CNN）
```

### 3. 可选库（用于高级功能）
```bash
tensorboard>=2.7.0    # 训练可视化
tqdm>=4.62.0          # 进度条
```

## 三、安装步骤

### 方式1：使用pip（推荐）

```bash
# 1. 创建虚拟环境（推荐）
python -m venv lottery_ml_env
source lottery_ml_env/bin/activate  # Linux/Mac
# 或
lottery_ml_env\Scripts\activate  # Windows

# 2. 安装基础库
pip install pandas numpy matplotlib scikit-learn

# 3. 安装PyTorch（根据你的系统选择）
# CPU版本（所有系统）
pip install torch torchvision

# GPU版本（NVIDIA显卡，需要CUDA）
# 访问 https://pytorch.org/ 获取正确的安装命令
# 例如（CUDA 11.8）:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其他依赖
pip install tqdm tensorboard
```

### 方式2：使用conda

```bash
# 1. 创建环境
conda create -n lottery_ml python=3.9
conda activate lottery_ml

# 2. 安装基础库
conda install pandas numpy matplotlib scikit-learn

# 3. 安装PyTorch（GPU版本）
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# 4. 安装其他依赖
pip install tqdm tensorboard
```

### 方式3：使用requirements文件

```bash
# 安装所有依赖
pip install -r requirements_ml.txt
```

## 四、验证安装

运行以下Python代码验证环境：

```python
import torch
import pandas as pd
import numpy as np
import sklearn

print(f"PyTorch版本: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU设备: {torch.cuda.get_device_name(0)}")
    print(f"CUDA版本: {torch.version.cuda}")

print(f"Pandas版本: {pd.__version__}")
print(f"NumPy版本: {np.__version__}")
print(f"Scikit-learn版本: {sklearn.__version__}")
```

## 五、数据准备

### 数据格式要求
CSV文件，包含以下列：
```
期号,前区1,前区2,前区3,前区4,前区5,后区1,后区2
2023001,1,5,12,23,35,3,8
2023002,2,7,15,28,33,1,9
...
```

### 数据量要求
- **最少**: 100期（基础训练）
- **推荐**: 300期以上（更好的效果）
- **理想**: 500期以上（最佳效果）

## 六、训练配置建议

### CPU训练配置
```python
config = {
    'batch_size': 16,        # 较小的批次
    'epochs': 30,            # 较少的轮数
    'learning_rate': 0.001,
    'hidden_size': 32,       # 较小的隐藏层
    'num_layers': 1          # 较少的层数
}
```

### GPU训练配置
```python
config = {
    'batch_size': 32,        # 较大的批次
    'epochs': 50,            # 更多的轮数
    'learning_rate': 0.001,
    'hidden_size': 64,       # 较大的隐藏层
    'num_layers': 2          # 更多的层数
}
```

## 七、常见问题

### Q1: 没有GPU怎么办？
A: 可以使用CPU训练，只是速度较慢。或者使用Google Colab等免费GPU平台。

### Q2: 内存不足怎么办？
A: 
- 减小batch_size
- 减少lookback期数
- 使用更小的模型（hidden_size）

### Q3: 训练太慢怎么办？
A:
- 使用GPU加速
- 减少训练数据量（先测试）
- 减少模型复杂度
- 使用云平台（Colab/Kaggle）

### Q4: 如何检查GPU是否可用？
A: 运行 `python -c "import torch; print(torch.cuda.is_available())"`

## 八、云平台使用

### Google Colab
1. 访问 https://colab.research.google.com/
2. 新建笔记本
3. 运行时 -> 更改运行时类型 -> 选择GPU
4. 上传数据和代码
5. 运行训练

### Kaggle
1. 访问 https://www.kaggle.com/
2. 新建Notebook
3. 设置 -> Accelerator -> GPU
4. 上传数据和代码
5. 运行训练

## 九、下一步

安装完成后，运行：
```bash
python ml_training_pipeline.py
```

这将自动：
1. 检查环境
2. 加载数据
3. 训练模型
4. 保存模型
5. 进行预测
