"""
大乐透数据分析示例
演示如何使用分析工具和预测模型
"""

from lottery_analyzer import LotteryAnalyzer
from prediction_models import LotteryPredictor
import pandas as pd
import numpy as np


def create_sample_data(num_periods: int = 100) -> pd.DataFrame:
    """
    创建示例数据（用于演示）
    实际使用时应该从真实数据源加载
    """
    data = []
    
    for period in range(1, num_periods + 1):
        # 随机生成前区5个号码（1-35，不重复）
        front = sorted(np.random.choice(range(1, 36), 5, replace=False))
        # 随机生成后区2个号码（1-12，不重复）
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
    """主函数：演示完整分析流程"""
    
    print("=" * 60)
    print("大乐透历史数据分析和预测示例")
    print("=" * 60)
    
    # 1. 创建或加载数据
    print("\n[步骤1] 准备数据...")
    # 如果有真实数据文件，使用：
    # analyzer = LotteryAnalyzer('lottery_history.csv')
    
    # 这里使用示例数据
    sample_data = create_sample_data(200)
    analyzer = LotteryAnalyzer()
    analyzer.data = sample_data
    print(f"✓ 已加载 {len(sample_data)} 期数据")
    
    # 2. 基础统计分析
    print("\n[步骤2] 频率分析...")
    freq = analyzer.frequency_analysis()
    print(f"前区最热号码: {sorted(freq['front'].items(), key=lambda x: x[1], reverse=True)[:5]}")
    print(f"后区最热号码: {sorted(freq['back'].items(), key=lambda x: x[1], reverse=True)[:3]}")
    
    # 3. 冷热号分析
    print("\n[步骤3] 冷热号分析...")
    cold_hot = analyzer.cold_hot_analysis(recent_periods=30)
    print(f"前区热号（最近30期）: {cold_hot['hot_front'][:10]}")
    print(f"前区冷号（最近未出现）: {cold_hot['cold_front'][:10]}")
    
    # 4. 遗漏值分析
    print("\n[步骤4] 遗漏值分析...")
    missing = analyzer.missing_value_analysis()
    top_missing_front = sorted(
        [(num, miss) for num, miss in missing['front_missing'].items() if miss is not None],
        key=lambda x: x[1],
        reverse=True
    )[:5]
    print(f"前区最大遗漏号码: {top_missing_front}")
    
    # 5. 组合特征分析
    print("\n[步骤5] 组合特征分析...")
    odd_even = analyzer.odd_even_ratio_analysis()
    size_ratio = analyzer.size_ratio_analysis()
    sum_value = analyzer.sum_value_analysis()
    
    print(f"最常见奇偶比: {odd_even['most_common_ratio']}")
    print(f"最常见大小比: {size_ratio['most_common_ratio']}")
    print(f"和值范围: {sum_value['min']} - {sum_value['max']} (平均: {sum_value['mean']:.1f})")
    
    # 6. 使用多种策略进行预测
    print("\n[步骤6] 号码预测...")
    predictor = LotteryPredictor(analyzer)
    
    print("\n使用不同策略的预测结果：")
    strategies = ['frequency_balance', 'cold_number', 'hot_number', 'combination_optimization', 'ensemble']
    
    for strategy_name in strategies:
        try:
            front, back = predictor.predict(strategy_name)
            print(f"  {strategy_name:25s}: 前区 {front}, 后区 {back}")
        except Exception as e:
            print(f"  {strategy_name:25s}: 预测失败 - {e}")
    
    # 7. 集成预测（推荐）
    print("\n[步骤7] 集成预测（综合多种策略）...")
    front, back = predictor.predict('ensemble', consensus=True)
    print(f"✓ 集成预测结果: 前区 {front}, 后区 {back}")
    
    # 8. 生成完整分析报告
    print("\n[步骤8] 生成分析报告...")
    report = analyzer.generate_report('analysis_report.json')
    print("✓ 分析报告已保存")
    
    print("\n" + "=" * 60)
    print("分析完成！")
    print("=" * 60)
    print("\n重要提醒：")
    print("1. 彩票号码是随机生成的，无法准确预测")
    print("2. 以上分析仅供参考，不保证预测准确性")
    print("3. 请理性投注，量力而行")


if __name__ == "__main__":
    main()
