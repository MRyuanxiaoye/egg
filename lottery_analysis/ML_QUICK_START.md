# 机器学习快速开始指南

## 一、环境准备（5分钟）

### 步骤1：检查Python版本
```bash
python --version
# 需要 Python 3.8+
```

### 步骤2：安装依赖
```bash
# 安装所有依赖
pip install -r requirements_ml.txt

# 或者手动安装
pip install pandas numpy matplotlib scikit-learn torch
```

### 步骤3：验证安装
```bash
python check_environment.py
```

## 二、准备数据

### 数据格式
CSV文件，包含以下列：
```csv
期号,前区1,前区2,前区3,前区4,前区5,后区1,后区2
2023001,1,5,12,23,35,3,8
2023002,2,7,15,28,33,1,9
...
```

### 数据量要求
- **最少**: 100期
- **推荐**: 300期以上
- **理想**: 500期以上

## 三、训练模型（3种方式）

### 方式1：最简单（使用示例数据）
```bash
python quick_start_ml.py
```
自动生成示例数据并训练。

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
    --batch-size 32
```

## 四、使用训练好的模型预测

```bash
python ml_prediction.py --data lottery_history.csv
```

## 五、完整示例

### 1. 检查环境
```bash
python check_environment.py
```

### 2. 训练模型
```bash
# 使用你的数据
python ml_training_pipeline.py --data your_data.csv

# 或使用示例数据
python quick_start_ml.py
```

### 3. 查看结果
- 模型文件: `./ml_models/*_model.pth`
- 配置文件: `./ml_models/*_config.json`
- 预测结果: `./ml_models/predictions.json`

### 4. 使用模型预测
```bash
python ml_prediction.py --data your_data.csv
```

## 六、硬件要求

### CPU训练（最低配置）
- CPU: 4核心+
- 内存: 8GB+
- 训练时间: 1-3小时（300期数据）

### GPU训练（推荐）
- GPU: NVIDIA显卡（支持CUDA）
- 显存: 4GB+
- 训练时间: 10-30分钟（300期数据）

### 无GPU？使用云平台
- **Google Colab**: 免费GPU
- **Kaggle**: 免费GPU（每周30小时）

## 七、常见问题

### Q1: 没有GPU怎么办？
A: 可以使用CPU训练，只是速度较慢。或者使用Google Colab。

### Q2: 内存不足？
A: 减小batch_size（如改为8或16）

### Q3: 训练太慢？
A: 
- 使用GPU
- 减少epochs（如改为20）
- 减少数据量（先测试）

### Q4: 如何检查GPU？
A: 运行 `python check_environment.py`

## 八、代码结构

```
ml_training_pipeline.py    # 完整训练流程（主程序）
quick_start_ml.py          # 快速开始（最简单）
ml_prediction.py           # 使用训练好的模型预测
check_environment.py       # 环境检查
ML_SETUP_GUIDE.md         # 详细安装指南
```

## 九、下一步

1. ✅ 检查环境: `python check_environment.py`
2. ✅ 准备数据: CSV格式的历史数据
3. ✅ 训练模型: `python ml_training_pipeline.py --data your_data.csv`
4. ✅ 查看结果: `./ml_models/predictions.json`
5. ✅ 使用预测: `python ml_prediction.py --data your_data.csv`

## 十、命令行参数

### ml_training_pipeline.py
```bash
--data FILE          数据文件路径
--models MODEL ...   模型类型 (lstm, transformer)
--output DIR         输出目录（默认: ./ml_models/）
--epochs N           训练轮数
--batch-size N       批次大小
--no-train           跳过训练
--no-evaluate        跳过评估
--no-predict         跳过预测
```

### ml_prediction.py
```bash
--data FILE          数据文件路径
--model-dir DIR      模型目录（默认: ./ml_models/）
```

## 十一、示例输出

训练完成后，你会看到：
```
✓ LSTM 模型训练完成并已保存
✓ 预测结果已保存至: ./ml_models/predictions.json
```

预测结果示例：
```json
{
  "lstm": {
    "front": [5, 12, 18, 25, 32],
    "back": [3, 8]
  },
  "ensemble": {
    "front": [6, 13, 19, 26, 33],
    "back": [4, 9]
  }
}
```

## 十二、重要提醒

⚠️ **彩票号码本质是随机的，无法准确预测**
- 机器学习仅提供统计参考
- 不保证预测准确性
- 请理性使用，量力而行
