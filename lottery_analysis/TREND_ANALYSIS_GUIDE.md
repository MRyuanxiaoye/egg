# 基于走势图匹配的预测方法指南

## 核心思路

### 1. 横向分析（位置走势）
分析每个位置的号码走势：
- **前区位置1-5**：每个位置的号码值如何变化
- **后区位置1-2**：每个位置的号码值如何变化
- **位置关系**：不同位置之间的关联性

### 2. 纵向分析（组合走势）
分析整个号码组合的特征走势：
- **和值走势**：前区5个号码的和值变化
- **跨度走势**：最大号与最小号的差值变化
- **奇偶比走势**：奇偶号码的比例变化
- **大小比走势**：大小号码的比例变化
- **连号走势**：连续号码的出现频率

### 3. 机器学习"看懂"走势图
使用深度学习模型学习走势模式：
- **LSTM**：学习时间序列的长期依赖关系
- **Transformer**：学习位置之间的关系和组合规律
- **CNN**：从走势图图像中提取视觉特征

### 4. 模式匹配
找到与当前走势最相似的历史模式：
- 计算当前走势与历史走势的相似度
- 选择最相似的历史模式
- 基于相似模式的下一个模式进行预测

## 使用方法

### 基础使用

```python
import pandas as pd
from trend_based_predictor import TrendBasedPredictor

# 1. 加载数据
data = pd.read_csv('lottery_history.csv')

# 2. 初始化预测器
predictor = TrendBasedPredictor(
    data=data,
    use_ml=True,  # 是否使用机器学习
    model_type='lstm'  # 模型类型：'lstm', 'transformer', 'matching'
)

# 3. 分析走势
trends = predictor.analyze_trends()

# 4. 生成走势图
image_paths = predictor.generate_trend_report()

# 5. 训练模型（可选）
predictor.train_ml_model(epochs=50)

# 6. 进行预测
front, back = predictor.predict(method='ensemble')
print(f"预测号码: 前区 {front}, 后区 {back}")
```

### 详细步骤

#### 步骤1：数据准备
```python
# CSV格式要求：
# 期号,前区1,前区2,前区3,前区4,前区5,后区1,后区2
# 2023001,1,5,12,23,35,3,8
# 2023002,2,7,15,28,33,1,9
# ...

data = pd.read_csv('lottery_history.csv')
```

#### 步骤2：走势分析
```python
predictor = TrendBasedPredictor(data)

# 横向分析：位置走势
position_trends = predictor.trend_analyzer.get_position_trends()
# 查看每个位置的走势
for pos in range(1, 6):
    values = position_trends['front_positions'][pos]
    print(f"位置{pos}走势: {values[-10:]}")

# 纵向分析：组合走势
combo_trends = predictor.trend_analyzer.get_combination_trends()
print(f"和值走势: {combo_trends['sum_trend'][-10:]}")
print(f"跨度走势: {combo_trends['span_trend'][-10:]}")
```

#### 步骤3：生成走势图
```python
# 生成所有走势图
image_paths = predictor.generate_trend_report(output_dir='./trends/')

# 生成的图像包括：
# - front_position_trends.png: 前区位置走势图
# - back_position_trends.png: 后区位置走势图
# - combination_trends.png: 组合特征走势图
# - comprehensive_trend.png: 综合走势图
```

#### 步骤4：模式匹配预测
```python
# 基于走势匹配进行预测
front, back = predictor.predict_by_trend_matching(
    lookback=20,  # 回看20期
    similarity_weight=0.7  # 相似度权重
)
```

#### 步骤5：机器学习预测
```python
# 训练模型
predictor.train_ml_model(
    epochs=50,
    batch_size=32,
    learning_rate=0.001
)

# 使用模型预测
front, back = predictor.ml_predictor.predict(lookback=30)
```

#### 步骤6：集成预测
```python
# 综合多种方法的预测结果
front, back = predictor.predict(method='ensemble')
```

## 模型说明

### LSTM模型
- **用途**：学习每个位置的走势规律（横向分析）
- **输入**：历史N期的特征（位置值+组合特征）
- **输出**：预测的7个号码（5个前区+2个后区）
- **特点**：能够捕捉时间序列的长期依赖关系

### Transformer模型
- **用途**：学习位置之间的关系和组合规律（纵向分析）
- **输入**：历史N期的特征序列
- **输出**：预测的7个号码
- **特点**：通过注意力机制学习位置之间的关联

### 模式匹配模型
- **用途**：找到与当前走势最相似的历史模式
- **方法**：
  1. 计算当前走势与历史走势的相似度
  2. 选择最相似的N个历史模式
  3. 使用这些模式的下一个模式进行预测
- **特点**：不需要训练，基于历史数据直接匹配

## 特征工程

### 位置特征
- 每个位置的号码值
- 每个位置的变化率（一阶差分）
- 每个位置的加速度（二阶差分）
- 每个位置的趋势（上升/下降/平稳）

### 组合特征
- 和值
- 跨度
- 奇偶比
- 大小比
- 连号数量
- 序列特征（差值序列）

### 时间特征
- 期数
- 周期性特征（如月份、星期等）

## 预测策略

### 策略1：模式匹配
- 找到最相似的历史走势
- 使用相似走势的下一个模式
- 优点：简单直观，不需要训练
- 缺点：可能过度依赖历史数据

### 策略2：趋势延续
- 基于当前位置的趋势预测下一期
- 使用线性外推或更复杂的回归
- 优点：捕捉趋势变化
- 缺点：随机事件没有真正的趋势

### 策略3：机器学习
- 使用深度学习模型学习走势模式
- 能够学习复杂的非线性关系
- 优点：理论上能学习更复杂的模式
- 缺点：需要大量数据和计算资源

### 策略4：集成方法
- 综合多种方法的预测结果
- 使用投票或加权平均
- 优点：更稳健，降低单一方法的风险
- 缺点：计算复杂度较高

## 注意事项

### 数据要求
- **数据量**：建议至少200期以上的历史数据
- **数据质量**：确保数据完整、准确
- **数据格式**：统一的数据格式

### 模型训练
- **训练集大小**：建议至少100期用于训练
- **验证集**：保留20%数据用于验证
- **过拟合**：注意避免在历史数据上过度优化

### 预测限制
- **随机性**：彩票本质是随机事件，无法准确预测
- **历史不代表未来**：历史走势不能保证未来规律
- **概率相等**：每个号码组合的中奖概率相等

## 文件说明

- `trend_analyzer.py`: 走势分析工具
- `ml_models.py`: 机器学习模型（LSTM、Transformer、CNN）
- `trend_based_predictor.py`: 基于走势的预测器
- `trend_prediction_example.py`: 完整使用示例
- `visualization.py`: 可视化工具

## 安装依赖

```bash
# 基础功能
pip install pandas numpy matplotlib

# 完整功能（包括机器学习）
pip install pandas numpy matplotlib torch scikit-learn
```

## 快速开始

运行示例：
```bash
python trend_prediction_example.py
```

这将：
1. 创建示例数据（或使用你的真实数据）
2. 进行走势分析
3. 生成走势图
4. 训练模型（如果使用ML）
5. 进行预测

## 总结

基于走势图匹配的预测方法通过：
1. **横向分析**：理解每个位置的走势
2. **纵向分析**：理解组合的整体走势
3. **机器学习**：学习复杂的走势模式
4. **模式匹配**：找到最相似的历史模式

来生成预测结果。虽然无法保证准确性，但提供了系统化的分析方法。
