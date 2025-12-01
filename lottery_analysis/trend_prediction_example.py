"""
基于走势图匹配的预测示例
演示如何使用走势分析和机器学习进行预测
"""

import pandas as pd
import numpy as np
from trend_based_predictor import TrendBasedPredictor
import warnings
warnings.filterwarnings('ignore')


def create_sample_data(num_periods: int = 200) -> pd.DataFrame:
    """创建示例数据"""
    data = []
    
    for period in range(1, num_periods + 1):
        # 生成有一定规律的数据（用于演示）
        # 前区：位置1-5，每个位置有趋势
        front = []
        for pos in range(1, 6):
            # 每个位置围绕一个中心值波动
            base = 5 + pos * 6 + np.sin(period / 20) * 3
            num = int(np.clip(base + np.random.normal(0, 2), 1, 35))
            front.append(num)
        front = sorted(list(set(front)))
        
        # 补充到5个号码
        while len(front) < 5:
            num = np.random.randint(1, 36)
            if num not in front:
                front.append(num)
        front = sorted(front[:5])
        
        # 后区
        back = sorted(np.random.choice(range(1, 13), 2, replace=False))
        
        row = {
            '期号': f"{period:05d}",
            '前区1': front[0],
            '前区2': front[1],
            '前区3': front[2],
            '前区4': front[3],
            '前区5': front[4],
            '后区1': back[0],
            '后区2': back[1]
        }
        data.append(row)
    
    return pd.DataFrame(data)


def main():
    """主函数：演示完整的走势分析和预测流程"""
    
    print("=" * 70)
    print("基于走势图匹配的大乐透预测系统")
    print("=" * 70)
    
    # 1. 准备数据
    print("\n[步骤1] 准备历史数据...")
    # 如果有真实数据，使用：
    # data = pd.read_csv('lottery_history.csv')
    
    # 这里使用示例数据
    data = create_sample_data(300)
    print(f"✓ 已加载 {len(data)} 期历史数据")
    
    # 2. 初始化预测器
    print("\n[步骤2] 初始化走势预测器...")
    predictor = TrendBasedPredictor(
        data=data,
        use_ml=True,  # 是否使用机器学习（需要PyTorch）
        model_type='lstm'  # 或 'transformer', 'matching'
    )
    print("✓ 预测器初始化完成")
    
    # 3. 走势分析
    print("\n[步骤3] 进行走势分析...")
    trends = predictor.analyze_trends()
    
    # 4. 生成走势图
    print("\n[步骤4] 生成走势图...")
    image_paths = predictor.generate_trend_report(output_dir='./trend_analysis/')
    
    # 5. 训练机器学习模型（可选）
    print("\n[步骤5] 训练机器学习模型...")
    try:
        predictor.train_ml_model(epochs=30, batch_size=16, learning_rate=0.001)
        print("✓ 模型训练完成")
    except Exception as e:
        print(f"模型训练失败: {e}")
        print("将使用模式匹配方法")
        predictor.use_ml = False
    
    # 6. 基于走势匹配进行预测
    print("\n[步骤6] 基于走势匹配进行预测...")
    front_matching, back_matching = predictor.predict_by_trend_matching(
        lookback=20,
        similarity_weight=0.7
    )
    print(f"\n✓ 模式匹配预测: 前区 {front_matching}, 后区 {back_matching}")
    
    # 7. 使用机器学习模型预测（如果可用）
    if predictor.use_ml and predictor.ml_predictor and predictor.ml_predictor.is_trained:
        print("\n[步骤7] 使用机器学习模型预测...")
        try:
            front_ml, back_ml = predictor.ml_predictor.predict(lookback=30)
            print(f"✓ 机器学习预测: 前区 {front_ml}, 后区 {back_ml}")
        except Exception as e:
            print(f"机器学习预测失败: {e}")
    
    # 8. 集成预测（综合多种方法）
    print("\n[步骤8] 集成预测（综合多种方法）...")
    front_ensemble, back_ensemble = predictor.predict(method='ensemble')
    print(f"✓ 集成预测结果: 前区 {front_ensemble}, 后区 {back_ensemble}")
    
    # 9. 详细分析报告
    print("\n" + "=" * 70)
    print("预测结果汇总")
    print("=" * 70)
    print(f"\n模式匹配预测: 前区 {front_matching}, 后区 {back_matching}")
    
    if predictor.use_ml and predictor.ml_predictor and predictor.ml_predictor.is_trained:
        try:
            front_ml, back_ml = predictor.ml_predictor.predict()
            print(f"机器学习预测: 前区 {front_ml}, 后区 {back_ml}")
        except:
            pass
    
    print(f"集成预测结果: 前区 {front_ensemble}, 后区 {back_ensemble}")
    
    print("\n" + "=" * 70)
    print("分析完成！")
    print("=" * 70)
    
    print("\n核心思路说明：")
    print("1. 横向分析：分析每个位置（前区1-5，后区1-2）的号码走势")
    print("2. 纵向分析：分析整个号码组合的特征走势（和值、跨度、奇偶比等）")
    print("3. 模式匹配：找到与当前走势最相似的历史模式")
    print("4. 机器学习：使用LSTM/Transformer学习走势规律")
    print("5. 综合预测：集成多种方法的预测结果")
    
    print("\n重要提醒：")
    print("- 彩票号码本质是随机的，无法准确预测")
    print("- 走势分析仅提供统计参考")
    print("- 请理性投注，量力而行")
    
    return {
        'matching': (front_matching, back_matching),
        'ensemble': (front_ensemble, back_ensemble),
        'trend_images': image_paths
    }


if __name__ == "__main__":
    result = main()
