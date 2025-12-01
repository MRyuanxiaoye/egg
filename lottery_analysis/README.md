# 大乐透历史数据分析和预测思路

## 重要声明
**彩票号码是随机生成的，理论上无法准确预测。以下分析仅从数据统计角度提供参考，不保证预测准确性。**

## 一、数据分析维度

### 1. 基础统计分析
- **号码出现频率**：统计每个号码在历史数据中的出现次数
- **冷热号分析**：识别长期未出现的"冷号"和频繁出现的"热号"
- **遗漏值分析**：计算每个号码距离上次出现的期数

### 2. 号码组合特征分析
- **奇偶比**：前区5个号码的奇偶比例（如3:2, 4:1等）
- **大小比**：前区号码的大小比例（1-17为小，18-35为大）
- **和值分析**：前区5个号码的和值分布
- **跨度分析**：前区最大号与最小号的差值
- **连号分析**：连续号码的出现频率（如01-02, 15-16等）

### 3. 后区号码分析
- **后区号码频率**：01-12每个号码的出现频率
- **后区组合模式**：常见后区号码组合
- **后区跨度**：两个后区号码的差值

### 4. 时间序列分析
- **周期性分析**：是否存在周期性规律
- **趋势分析**：号码出现频率的时间趋势
- **季节性分析**：不同月份/季节的号码分布特征

### 5. 高级分析方法
- **关联规则挖掘**：发现号码之间的关联关系
- **聚类分析**：将号码分组，分析组合模式
- **机器学习预测**：
  - 时间序列模型（ARIMA, LSTM）
  - 分类模型（预测号码是否出现）
  - 回归模型（预测和值、跨度等）

## 二、预测策略

### 策略1：频率平衡法
- 选择出现频率接近平均值的号码
- 避免极端热号和极端冷号

### 策略2：冷号回补法
- 关注长期未出现的冷号
- 基于"均值回归"理论

### 策略3：热号延续法
- 选择近期频繁出现的号码
- 基于"趋势延续"假设

### 策略4：组合优化法
- 基于历史组合特征（奇偶比、大小比、和值等）
- 生成符合历史规律的号码组合

### 策略5：机器学习集成
- 使用多种模型预测
- 集成多个模型的预测结果

### 策略6：走势图匹配法（推荐）⭐
- **横向分析**：分析每个位置的号码走势（位置1-5，后区1-2）
- **纵向分析**：分析整个号码组合的走势（和值、跨度、奇偶比等）
- **模式匹配**：找到与当前走势最相似的历史模式
- **机器学习**：使用LSTM/Transformer学习走势规律
- **综合预测**：基于走势匹配生成预测号码

详见 `TREND_ANALYSIS_GUIDE.md` 和 `trend_based_predictor.py`

## 三、分析流程

1. **数据收集与清洗**
   - 收集完整历史数据
   - 数据格式标准化
   - 异常值检测

2. **探索性数据分析（EDA）**
   - 描述性统计
   - 可视化分析
   - 分布特征识别

3. **特征工程**
   - 提取统计特征
   - 构建时间特征
   - 创建组合特征

4. **模型训练与评估**
   - 划分训练集和测试集
   - 训练多个模型
   - 交叉验证评估

5. **预测与优化**
   - 生成候选号码
   - 组合筛选
   - 风险评估

## 四、注意事项

1. **随机性认知**：彩票本质是随机事件，历史数据不能保证未来结果
2. **样本偏差**：历史数据可能存在偏差，不代表未来规律
3. **过度拟合**：避免在历史数据上过度优化
4. **概率理解**：任何号码组合的中奖概率都是相等的
5. **理性投注**：分析仅供娱乐参考，应理性投注

## 五、技术实现

### 基础分析工具
- `lottery_analyzer.py`: 基础统计分析工具
- `prediction_models.py`: 传统预测策略
- `visualization.py`: 数据可视化工具

### 走势图分析（核心功能）⭐
- `trend_analyzer.py`: 走势图分析工具（横向+纵向）
- `ml_models.py`: 深度学习模型（LSTM、Transformer、CNN）
- `trend_based_predictor.py`: 基于走势的预测器
- `trend_prediction_example.py`: 完整使用示例

### 文档
- `TREND_ANALYSIS_GUIDE.md`: 走势分析详细指南
- `ANALYSIS_METHODOLOGY.md`: 分析方法论详解

## 六、快速开始

### 基础分析
```python
from lottery_analyzer import LotteryAnalyzer
from prediction_models import LotteryPredictor

analyzer = LotteryAnalyzer('lottery_history.csv')
predictor = LotteryPredictor(analyzer)
front, back = predictor.predict('ensemble')
```

### 走势图分析（推荐）
```python
from trend_based_predictor import TrendBasedPredictor
import pandas as pd

data = pd.read_csv('lottery_history.csv')
predictor = TrendBasedPredictor(data, use_ml=True, model_type='lstm')

# 分析走势
trends = predictor.analyze_trends()

# 生成走势图
predictor.generate_trend_report()

# 训练模型
predictor.train_ml_model(epochs=50)

# 预测
front, back = predictor.predict(method='ensemble')
```

运行完整示例：
```bash
python trend_prediction_example.py
```

## 七、机器学习训练

### 环境准备
```bash
# 1. 检查环境
python check_environment.py

# 2. 安装依赖
pip install -r requirements_ml.txt
```

### 快速开始
```bash
# 方式1：使用示例数据（最简单）
python quick_start_ml.py

# 方式2：使用你的数据
python ml_training_pipeline.py --data lottery_history.csv

# 方式3：使用训练好的模型预测
python ml_prediction.py --data lottery_history.csv
```

### 详细指南
- `ML_QUICK_START.md`: 机器学习快速开始
- `ML_SETUP_GUIDE.md`: 详细安装和配置指南
- `ml_training_pipeline.py`: 完整训练流程

### 硬件要求
- **CPU训练**: 4核心+, 8GB内存（较慢）
- **GPU训练**: NVIDIA显卡, 4GB显存（推荐，快10-20倍）
- **云平台**: Google Colab / Kaggle（免费GPU）
