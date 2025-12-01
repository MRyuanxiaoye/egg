"""
大乐透号码预测模型
提供多种预测策略和模型
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from lottery_analyzer import LotteryAnalyzer
import random
from collections import Counter

# 尝试导入走势预测器（可选）
try:
    from trend_based_predictor import TrendBasedPredictor
    HAS_TREND_PREDICTOR = True
except ImportError:
    HAS_TREND_PREDICTOR = False


class PredictionStrategy:
    """预测策略基类"""
    
    def __init__(self, analyzer: LotteryAnalyzer):
        self.analyzer = analyzer
    
    def predict(self) -> Tuple[List[int], List[int]]:
        """
        预测下一期号码
        
        Returns:
            (前区5个号码, 后区2个号码)
        """
        raise NotImplementedError


class FrequencyBalanceStrategy(PredictionStrategy):
    """频率平衡策略：选择出现频率接近平均值的号码"""
    
    def predict(self) -> Tuple[List[int], List[int]]:
        freq_data = self.analyzer.frequency_analysis()
        
        # 前区：选择频率接近平均值的号码
        front_freq = freq_data['front_freq']
        avg_freq = np.mean(list(front_freq.values()))
        
        # 计算每个号码与平均频率的差值
        front_scores = {
            num: abs(freq - avg_freq) 
            for num, freq in front_freq.items()
            if num in range(1, 36)
        }
        
        # 选择差值最小的5个号码
        front_selected = sorted(front_scores.items(), key=lambda x: x[1])[:5]
        front_numbers = sorted([num for num, _ in front_selected])
        
        # 后区：同样方法选择2个
        back_freq = freq_data['back_freq']
        avg_back_freq = np.mean(list(back_freq.values()))
        
        back_scores = {
            num: abs(freq - avg_back_freq)
            for num, freq in back_freq.items()
            if num in range(1, 13)
        }
        
        back_selected = sorted(back_scores.items(), key=lambda x: x[1])[:2]
        back_numbers = sorted([num for num, _ in back_selected])
        
        return front_numbers, back_numbers


class ColdNumberStrategy(PredictionStrategy):
    """冷号回补策略：关注长期未出现的号码"""
    
    def predict(self, top_cold: int = 10) -> Tuple[List[int], List[int]]:
        missing_data = self.analyzer.missing_value_analysis()
        
        # 前区：选择遗漏值最大的号码
        front_missing = missing_data['front_missing']
        # 过滤掉None值，按遗漏值排序
        front_sorted = sorted(
            [(num, miss) for num, miss in front_missing.items() if miss is not None],
            key=lambda x: x[1],
            reverse=True
        )
        
        # 从最冷的号码中随机选择5个
        cold_front = [num for num, _ in front_sorted[:top_cold]]
        front_numbers = sorted(random.sample(cold_front, min(5, len(cold_front))))
        
        # 后区：同样方法
        back_missing = missing_data['back_missing']
        back_sorted = sorted(
            [(num, miss) for num, miss in back_missing.items() if miss is not None],
            key=lambda x: x[1],
            reverse=True
        )
        
        cold_back = [num for num, _ in back_sorted[:top_cold]]
        back_numbers = sorted(random.sample(cold_back, min(2, len(cold_back))))
        
        return front_numbers, back_numbers


class HotNumberStrategy(PredictionStrategy):
    """热号延续策略：选择近期频繁出现的号码"""
    
    def predict(self) -> Tuple[List[int], List[int]]:
        cold_hot = self.analyzer.cold_hot_analysis(recent_periods=30)
        
        # 前区：从热号中选择
        hot_front = cold_hot['hot_front']
        if len(hot_front) >= 5:
            front_numbers = sorted(random.sample(hot_front, 5))
        else:
            # 如果热号不足，补充随机号码
            all_front = list(range(1, 36))
            remaining = [n for n in all_front if n not in hot_front]
            front_numbers = sorted(hot_front + random.sample(remaining, 5 - len(hot_front)))
        
        # 后区
        hot_back = cold_hot['hot_back']
        if len(hot_back) >= 2:
            back_numbers = sorted(random.sample(hot_back, 2))
        else:
            all_back = list(range(1, 13))
            remaining = [n for n in all_back if n not in hot_back]
            back_numbers = sorted(hot_back + random.sample(remaining, 2 - len(hot_back)))
        
        return front_numbers, back_numbers


class CombinationOptimizationStrategy(PredictionStrategy):
    """组合优化策略：基于历史组合特征生成号码"""
    
    def predict(self) -> Tuple[List[int], List[int]]:
        # 获取历史特征
        odd_even = self.analyzer.odd_even_ratio_analysis()
        size_ratio = self.analyzer.size_ratio_analysis()
        sum_value = self.analyzer.sum_value_analysis()
        
        # 目标奇偶比（最常见）
        target_ratio = odd_even['most_common_ratio']
        odd_target, even_target = map(int, target_ratio.split(':'))
        
        # 目标大小比
        target_size = size_ratio['most_common_ratio']
        small_target, large_target = map(int, target_size.split(':'))
        
        # 目标 and 值范围（均值±标准差）
        target_sum_min = int(sum_value['mean'] - sum_value['std'])
        target_sum_max = int(sum_value['mean'] + sum_value['std'])
        
        # 生成符合特征的组合
        max_attempts = 1000
        for _ in range(max_attempts):
            # 随机生成前区号码
            front_candidates = random.sample(range(1, 36), 5)
            front_numbers = sorted(front_candidates)
            
            # 检查奇偶比
            odd_count = sum(1 for n in front_numbers if n % 2 == 1)
            if odd_count != odd_target:
                continue
            
            # 检查大小比
            small_count = sum(1 for n in front_numbers if n <= 17)
            if small_count != small_target:
                continue
            
            # 检查和值
            sum_val = sum(front_numbers)
            if target_sum_min <= sum_val <= target_sum_max:
                # 生成后区（简单随机）
                back_numbers = sorted(random.sample(range(1, 13), 2))
                return front_numbers, back_numbers
        
        # 如果无法生成符合特征的组合，返回随机组合
        front_numbers = sorted(random.sample(range(1, 36), 5))
        back_numbers = sorted(random.sample(range(1, 13), 2))
        return front_numbers, back_numbers


class EnsembleStrategy(PredictionStrategy):
    """集成策略：综合多种策略的结果"""
    
    def __init__(self, analyzer: LotteryAnalyzer):
        super().__init__(analyzer)
        self.strategies = [
            FrequencyBalanceStrategy(analyzer),
            ColdNumberStrategy(analyzer),
            HotNumberStrategy(analyzer),
            CombinationOptimizationStrategy(analyzer)
        ]
    
    def predict(self, num_predictions: int = 5) -> List[Tuple[List[int], List[int]]]:
        """
        使用多种策略生成多个预测结果
        
        Args:
            num_predictions: 生成预测结果的数量
        
        Returns:
            预测结果列表
        """
        predictions = []
        
        for _ in range(num_predictions):
            # 随机选择一个策略
            strategy = random.choice(self.strategies)
            pred = strategy.predict()
            predictions.append(pred)
        
        return predictions
    
    def predict_consensus(self) -> Tuple[List[int], List[int]]:
        """
        基于多个策略的共识预测
        选择在多个预测中出现频率高的号码
        """
        predictions = self.predict(num_predictions=20)
        
        # 统计前区号码出现频率
        front_counter = Counter()
        back_counter = Counter()
        
        for front, back in predictions:
            front_counter.update(front)
            back_counter.update(back)
        
        # 选择出现频率最高的号码
        front_numbers = sorted([num for num, _ in front_counter.most_common(5)])
        back_numbers = sorted([num for num, _ in back_counter.most_common(2)])
        
        return front_numbers, back_numbers


class TrendMatchingStrategy(PredictionStrategy):
    """走势匹配策略：基于走势图匹配"""
    
    def __init__(self, analyzer: LotteryAnalyzer):
        super().__init__(analyzer)
        if HAS_TREND_PREDICTOR and hasattr(analyzer, 'data'):
            self.trend_predictor = TrendBasedPredictor(analyzer.data, use_ml=False)
        else:
            self.trend_predictor = None
    
    def predict(self) -> Tuple[List[int], List[int]]:
        if self.trend_predictor:
            return self.trend_predictor.predict(method='matching')
        else:
            # 回退到简单策略
            from prediction_models import FrequencyBalanceStrategy
            fallback = FrequencyBalanceStrategy(self.analyzer)
            return fallback.predict()


class LotteryPredictor:
    """大乐透预测器主类"""
    
    def __init__(self, analyzer: LotteryAnalyzer):
        self.analyzer = analyzer
        self.strategies = {
            'frequency_balance': FrequencyBalanceStrategy(analyzer),
            'cold_number': ColdNumberStrategy(analyzer),
            'hot_number': HotNumberStrategy(analyzer),
            'combination_optimization': CombinationOptimizationStrategy(analyzer),
            'ensemble': EnsembleStrategy(analyzer)
        }
        
        # 添加走势匹配策略（如果可用）
        if HAS_TREND_PREDICTOR and hasattr(analyzer, 'data'):
            try:
                self.strategies['trend_matching'] = TrendMatchingStrategy(analyzer)
            except Exception as e:
                print(f"走势匹配策略初始化失败: {e}")
    
    def predict(self, strategy: str = 'ensemble', **kwargs) -> Tuple[List[int], List[int]]:
        """
        使用指定策略进行预测
        
        Args:
            strategy: 策略名称
            **kwargs: 策略特定参数
        
        Returns:
            (前区号码, 后区号码)
        """
        if strategy not in self.strategies:
            raise ValueError(f"未知策略: {strategy}")
        
        predictor = self.strategies[strategy]
        
        if strategy == 'ensemble' and 'consensus' in kwargs and kwargs['consensus']:
            return predictor.predict_consensus()
        elif strategy == 'ensemble':
            return predictor.predict(**kwargs)
        else:
            return predictor.predict(**kwargs)
    
    def multi_strategy_predict(self) -> Dict[str, Tuple[List[int], List[int]]]:
        """使用所有策略进行预测"""
        results = {}
        for name, strategy in self.strategies.items():
            if name == 'ensemble':
                results[name] = strategy.predict_consensus()
            else:
                results[name] = strategy.predict()
        return results


if __name__ == "__main__":
    print("大乐透预测模型")
    print("=" * 50)
    print("\n可用策略：")
    print("1. frequency_balance - 频率平衡策略")
    print("2. cold_number - 冷号回补策略")
    print("3. hot_number - 热号延续策略")
    print("4. combination_optimization - 组合优化策略")
    print("5. ensemble - 集成策略（推荐）")
    print("\n示例用法：")
    print("  analyzer = LotteryAnalyzer('lottery_history.csv')")
    print("  predictor = LotteryPredictor(analyzer)")
    print("  front, back = predictor.predict('ensemble')")
    print("  print(f'预测号码: 前区 {front}, 后区 {back}')")
