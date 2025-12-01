# 机器学习完整指南

## 📋 目录

1. [硬件要求](#硬件要求)
2. [软件环境](#软件环境)
3. [安装步骤](#安装步骤)
4. [数据准备](#数据准备)
5. [训练模型](#训练模型)
6. [使用模型预测](#使用模型预测)
7. [代码生成说明](#代码生成说明)
8. [常见问题](#常见问题)

---

## 一、硬件要求

### 🖥️ CPU训练（最低配置）

**适用场景**: 没有GPU或想先测试

- **CPU**: 4核心以上（推荐8核心）
- **内存**: 8GB RAM（推荐16GB）
- **存储**: 至少10GB可用空间
- **训练时间**: 
  - 100期数据: 10-30分钟
  - 300期数据: 1-3小时
  - 500期数据: 2-5小时

**优点**: 
- 无需额外硬件
- 适合小规模测试

**缺点**: 
- 训练速度慢
- 无法训练复杂模型

### 🎮 GPU训练（推荐配置）

**适用场景**: 有NVIDIA显卡

- **CPU**: 6核心以上
- **GPU**: NVIDIA显卡，支持CUDA
  - 最低: GTX 1050 Ti (4GB显存)
  - 推荐: RTX 2060 / GTX 1660 (6GB+显存)
  - 理想: RTX 3060+ (8GB+显存)
- **显存**: 4GB以上（推荐8GB+）
- **内存**: 16GB RAM（推荐32GB）
- **存储**: 至少20GB可用空间
- **训练时间**: 
  - 100期数据: 2-5分钟
  - 300期数据: 10-30分钟
  - 500期数据: 20-60分钟

**优点**: 
- 训练速度快（10-20倍）
- 可以训练复杂模型
- 支持更大的batch size

**缺点**: 
- 需要NVIDIA显卡
- 需要安装CUDA

### ☁️ 云平台（无需硬件）

**适用场景**: 没有GPU或不想配置环境

#### Google Colab（推荐）
- **免费**: 是
- **GPU**: 免费GPU（有限制）
- **使用**: 
  1. 访问 https://colab.research.google.com/
  2. 新建笔记本
  3. 运行时 -> 更改运行时类型 -> GPU
  4. 上传代码和数据
  5. 运行训练

#### Kaggle
- **免费**: 是
- **GPU**: 免费GPU（每周30小时）
- **使用**: 
  1. 访问 https://www.kaggle.com/
  2. 新建Notebook
  3. 设置 -> Accelerator -> GPU
  4. 上传代码和数据
  5. 运行训练

---

## 二、软件环境

### Python版本
- **要求**: Python 3.8+
- **推荐**: Python 3.9 或 3.10
- **检查**: `python --version`

### 必需库

| 库名 | 版本 | 用途 |
|------|------|------|
| pandas | >=1.3.0 | 数据处理 |
| numpy | >=1.21.0 | 数值计算 |
| matplotlib | >=3.4.0 | 可视化 |
| scikit-learn | >=0.24.0 | 机器学习工具 |
| torch | >=1.9.0 | 深度学习框架 |

### 可选库

| 库名 | 用途 |
|------|------|
| tqdm | 进度条显示 |
| tensorboard | 训练可视化 |

---

## 三、安装步骤

### 方式1：使用pip（推荐）

```bash
# 1. 创建虚拟环境（推荐，避免冲突）
python -m venv lottery_ml_env

# 激活虚拟环境
# Windows:
lottery_ml_env\Scripts\activate
# Linux/Mac:
source lottery_ml_env/bin/activate

# 2. 安装基础库
pip install pandas numpy matplotlib scikit-learn

# 3. 安装PyTorch
# CPU版本（所有系统）:
pip install torch torchvision

# GPU版本（NVIDIA显卡）:
# 访问 https://pytorch.org/ 获取正确的安装命令
# 例如（CUDA 11.8）:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. 安装其他依赖
pip install tqdm

# 或一次性安装所有依赖
pip install -r requirements_ml.txt
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
pip install tqdm
```

### 验证安装

运行环境检查脚本：
```bash
python check_environment.py
```

或手动检查：
```python
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA可用: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
```

---

## 四、数据准备

### 数据格式

CSV文件，包含以下列：
```csv
期号,前区1,前区2,前区3,前区4,前区5,后区1,后区2
2023001,1,5,12,23,35,3,8
2023002,2,7,15,28,33,1,9
2023003,3,9,18,29,34,2,10
...
```

### 数据量要求

| 数据量 | 用途 | 效果 |
|--------|------|------|
| 100期 | 基础训练 | 可训练，效果一般 |
| 200期 | 推荐最少 | 效果较好 |
| 300期 | 推荐 | 效果良好 |
| 500期+ | 理想 | 效果最佳 |

### 数据质量检查

运行训练管道会自动检查：
- 数据量是否充足
- 必需列是否存在
- 数据范围是否有效（前区1-35，后区1-12）

---

## 五、训练模型

### 方式1：最简单（使用示例数据）

```bash
python quick_start_ml.py
```

自动生成示例数据并训练，适合测试环境。

### 方式2：使用你的数据文件

```bash
python ml_training_pipeline.py --data lottery_history.csv
```

### 方式3：自定义参数

```bash
python ml_training_pipeline.py \
    --data lottery_history.csv \
    --models lstm transformer \
    --epochs 50 \
    --batch-size 32 \
    --output ./my_models/
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--data` | 数据文件路径 | 无（使用示例数据） |
| `--models` | 模型类型 | `lstm` |
| `--epochs` | 训练轮数 | 30-50（根据硬件） |
| `--batch-size` | 批次大小 | 16-32（根据硬件） |
| `--output` | 输出目录 | `./ml_models/` |
| `--no-train` | 跳过训练 | False |
| `--no-evaluate` | 跳过评估 | False |
| `--no-predict` | 跳过预测 | False |

### 训练配置建议

#### CPU训练
```python
batch_size = 16
epochs = 30
hidden_size = 32
num_layers = 1
```

#### GPU训练
```python
batch_size = 32
epochs = 50
hidden_size = 64
num_layers = 2
```

### 训练过程

训练时会显示：
1. 环境检查
2. 数据质量检查
3. 训练进度（每个epoch的loss）
4. 验证结果
5. 模型保存位置

### 训练输出

训练完成后会生成：
```
ml_models/
├── lstm_model.pth          # 模型文件
├── lstm_config.json        # 配置文件
├── predictions.json        # 预测结果
└── ...
```

---

## 六、使用模型预测

### 方式1：训练时自动预测

训练完成后会自动生成预测结果：
```json
{
  "lstm": {
    "front": [5, 12, 18, 25, 32],
    "back": [3, 8]
  }
}
```

### 方式2：使用训练好的模型

```bash
python ml_prediction.py --data lottery_history.csv
```

### 方式3：在代码中使用

```python
from ml_prediction import predict_with_models
import pandas as pd

data = pd.read_csv('lottery_history.csv')
predictions = predict_with_models(data=data, model_dir='./ml_models/')

for model_type, (front, back) in predictions.items():
    print(f"{model_type}: 前区 {front}, 后区 {back}")
```

---

## 七、代码生成说明

### 已生成的代码文件

#### 1. 核心训练代码
- **`ml_training_pipeline.py`**: 完整的训练管道
  - 环境检查
  - 数据加载和验证
  - 模型训练
  - 模型评估
  - 预测生成
  - 结果保存

#### 2. 快速开始
- **`quick_start_ml.py`**: 最简单的使用方式
  - 自动生成示例数据
  - 一键训练和预测

#### 3. 预测代码
- **`ml_prediction.py`**: 使用训练好的模型预测
  - 加载模型
  - 生成预测
  - 集成多个模型

#### 4. 工具脚本
- **`check_environment.py`**: 环境检查
  - 检查Python版本
  - 检查依赖库
  - 检查GPU
  - 检查项目文件

### 代码结构

```
ml_training_pipeline.py
├── MLTrainingPipeline类
│   ├── check_data_quality()      # 数据质量检查
│   ├── train_models()             # 训练模型
│   ├── evaluate_models()          # 评估模型
│   ├── make_predictions()         # 生成预测
│   └── run_full_pipeline()        # 运行完整流程
└── main()                         # 命令行入口
```

### 如何修改代码

#### 修改训练参数
编辑 `ml_training_pipeline.py` 中的 `_get_default_config()` 方法：

```python
def _get_default_config(self) -> dict:
    return {
        'batch_size': 32,      # 修改这里
        'epochs': 50,          # 修改这里
        'learning_rate': 0.001, # 修改这里
        ...
    }
```

#### 添加新模型
1. 在 `ml_models.py` 中添加新模型类
2. 在 `trend_based_predictor.py` 中集成
3. 在 `ml_training_pipeline.py` 中添加支持

---

## 八、常见问题

### Q1: 没有GPU怎么办？

**A**: 三种选择：
1. **使用CPU训练**（较慢但可用）
2. **使用Google Colab**（免费GPU）
3. **使用Kaggle**（免费GPU，每周30小时）

### Q2: 内存不足怎么办？

**A**: 调整配置：
```bash
python ml_training_pipeline.py \
    --data your_data.csv \
    --batch-size 8  # 减小batch size
```

或修改代码中的 `hidden_size` 和 `num_layers`。

### Q3: 训练太慢怎么办？

**A**: 
1. **使用GPU**（最快）
2. **减少epochs**: `--epochs 20`
3. **减少数据量**: 先用100期测试
4. **使用云平台**: Google Colab / Kaggle

### Q4: 如何检查GPU是否可用？

**A**: 
```bash
python check_environment.py
```

或：
```python
import torch
print(torch.cuda.is_available())
```

### Q5: 模型训练失败？

**A**: 检查：
1. 数据格式是否正确
2. 数据量是否足够（至少100期）
3. 内存是否充足
4. 查看错误信息

### Q6: 预测结果不理想？

**A**: 
1. **增加数据量**: 使用更多历史数据
2. **调整参数**: 尝试不同的epochs和batch_size
3. **使用集成方法**: 训练多个模型并集成
4. **重要**: 彩票本质随机，无法准确预测

### Q7: 如何在云平台使用？

**Google Colab**:
1. 上传代码和数据到Google Drive
2. 在Colab中打开代码
3. 运行时 -> 更改运行时类型 -> GPU
4. 运行训练

**Kaggle**:
1. 创建新Notebook
2. 上传数据（Add data）
3. 设置 -> Accelerator -> GPU
4. 运行训练

---

## 九、完整工作流程

### 第一次使用

```bash
# 1. 检查环境
python check_environment.py

# 2. 安装依赖（如果未安装）
pip install -r requirements_ml.txt

# 3. 准备数据（CSV格式）
# 或使用示例数据测试

# 4. 训练模型
python ml_training_pipeline.py --data lottery_history.csv

# 5. 查看结果
cat ml_models/predictions.json

# 6. 使用模型预测
python ml_prediction.py --data lottery_history.csv
```

### 日常使用

```bash
# 1. 更新数据后重新训练
python ml_training_pipeline.py --data updated_data.csv

# 2. 使用最新模型预测
python ml_prediction.py --data updated_data.csv
```

---

## 十、总结

### 你需要准备的东西

1. ✅ **硬件**: CPU（最低）或GPU（推荐）
2. ✅ **软件**: Python 3.8+, PyTorch等库
3. ✅ **数据**: CSV格式的历史数据（至少100期）
4. ✅ **代码**: 已全部生成，直接使用

### 快速开始命令

```bash
# 1. 检查环境
python check_environment.py

# 2. 训练模型（最简单）
python quick_start_ml.py

# 3. 使用你的数据
python ml_training_pipeline.py --data your_data.csv
```

### 重要提醒

⚠️ **彩票号码本质是随机的，无法准确预测**
- 机器学习仅提供统计参考
- 不保证预测准确性
- 请理性使用，量力而行

---

## 十一、获取帮助

如果遇到问题：
1. 查看 `ML_SETUP_GUIDE.md`（详细安装指南）
2. 查看 `ML_QUICK_START.md`（快速开始）
3. 运行 `python check_environment.py` 检查环境
4. 查看错误信息和日志
