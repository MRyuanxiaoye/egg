"""
基于走势图匹配的预测策略
核心：通过图论和机器学习"看懂"走势图，匹配最合适的号码组合
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
from trend_analyzer import TrendAnalyzer
from ml_models import TrendMLPredictor, TrendMatcher
import warnings
warnings.filterwarnings('ignore')


class TrendBasedPredictor:
    """
    基于走势图的预测器
    核心思路：
    1. 横向分析：每个位置的号码走势（位置1、位置2...）
    2. 纵向分析：整个号码组合的走势规律
    3. 机器学习：使用深度学习模型学习走势模式
    4. 模式匹配：找到最相似的历史走势，预测下一期
    """
    
    def __init__(self, data: pd.DataFrame, use_ml: bool = True, model_type: str = 'lstm'):
        """
        初始化预测器
        
        Args:
            data: 历史数据DataFrame
            use_ml: 是否使用机器学习模型
            model_type: 模型类型 ('lstm', 'transformer', 'matching')
        """
        self.data = data
        self.trend_analyzer = TrendAnalyzer(data)
        self.use_ml = use_ml
        self.model_type = model_type
        self.ml_predictor = None
        self.matcher = TrendMatcher(self.trend_analyzer)
        
        if use_ml:
            try:
                self.ml_predictor = TrendMLPredictor(self.trend_analyzer, model_type=model_type)
            except Exception as e:
                print(f"机器学习模型初始化失败: {e}，将使用模式匹配方法")
                self.use_ml = False
    
    def analyze_trends(self) -> Dict:
        """
        综合分析走势
        
        Returns:
            包含横向和纵向走势分析的字典
        """
        print("=" * 60)
        print("走势分析")
        print("=" * 60)
        
        # 横向分析：位置走势
        print("\n[横向分析] 各位置号码走势...")
        position_trends = self.trend_analyzer.get_position_trends()
        
        print("\n前区位置特征：")
        for pos in range(1, 6):
            values = position_trends['front_positions'][pos]
            recent_values = values[-10:]
            trend = position_trends['trend_features']['position_trend'].get(f'front_{pos}', '未知')
            mean_val = position_trends['trend_features']['position_mean'].get(f'front_{pos}', 0)
            print(f"  位置{pos}: 均值={mean_val:.1f}, 趋势={trend}, 最近10期={recent_values[-5:]}")
        
        print("\n后区位置特征：")
        for pos in range(1, 3):
            values = position_trends['back_positions'][pos]
            recent_values = values[-10:]
            trend = position_trends['trend_features']['position_trend'].get(f'back_{pos}', '未知')
            mean_val = position_trends['trend_features']['position_mean'].get(f'back_{pos}', 0)
            print(f"  位置{pos}: 均值={mean_val:.1f}, 趋势={trend}, 最近10期={recent_values[-5:]}")
        
        # 纵向分析：组合走势
        print("\n[纵向分析] 号码组合走势...")
        combination_trends = self.trend_analyzer.get_combination_trends()
        
        print(f"\n和值走势: 均值={np.mean(combination_trends['sum_trend']):.1f}, "
              f"最近10期={combination_trends['sum_trend'][-10:].tolist()}")
        print(f"跨度走势: 均值={np.mean(combination_trends['span_trend']):.1f}, "
              f"最近10期={combination_trends['span_trend'][-10:].tolist()}")
        print(f"奇偶比走势: 最近10期={combination_trends['odd_even_trend'][-10:].tolist()}")
        print(f"大小比走势: 最近10期={combination_trends['size_ratio_trend'][-10:].tolist()}")
        
        return {
            'position_trends': position_trends,
            'combination_trends': combination_trends
        }
    
    def train_ml_model(self, epochs: int = 50, **kwargs):
        """训练机器学习模型"""
        if not self.use_ml or self.ml_predictor is None:
            print("机器学习模型不可用")
            return
        
        print("\n" + "=" * 60)
        print("训练机器学习模型")
        print("=" * 60)
        self.ml_predictor.train(epochs=epochs, **kwargs)
    
    def predict_by_trend_matching(self, lookback: int = 10, 
                                   similarity_weight: float = 0.7) -> Tuple[List[int], List[int]]:
        """
        基于走势匹配进行预测
        
        Args:
            lookback: 回看期数
            similarity_weight: 相似度权重（vs 趋势延续权重）
        
        Returns:
            (前区号码, 后区号码)
        """
        print("\n" + "=" * 60)
        print("基于走势匹配的预测")
        print("=" * 60)
        
        # 获取当前走势模式
        position_trends = self.trend_analyzer.get_position_trends()
        combination_trends = self.trend_analyzer.get_combination_trends()
        
        # 构建当前模式（最近N期的平均）
        recent_periods = min(lookback, len(self.data))
        current_pattern = {
            'front_positions': [
                np.mean([position_trends['front_positions'][i][-j] 
                        for j in range(1, recent_periods + 1)])
                for i in range(1, 6)
            ],
            'back_positions': [
                np.mean([position_trends['back_positions'][i][-j] 
                        for j in range(1, recent_periods + 1)])
                for i in range(1, 3)
            ],
            'sum': np.mean(combination_trends['sum_trend'][-recent_periods:]),
            'span': np.mean(combination_trends['span_trend'][-recent_periods:]),
            'odd_even': np.mean(combination_trends['odd_even_trend'][-recent_periods:]),
            'size_ratio': np.mean(combination_trends['size_ratio_trend'][-recent_periods:]),
        }
        
        print(f"\n当前走势模式（最近{recent_periods}期平均）：")
        print(f"  前区位置: {[round(n, 1) for n in current_pattern['front_positions']]}")
        print(f"  后区位置: {[round(n, 1) for n in current_pattern['back_positions']]}")
        print(f"  和值: {current_pattern['sum']:.1f}")
        print(f"  跨度: {current_pattern['span']:.1f}")
        
        # 找到最相似的历史模式
        similar_patterns = self.matcher.find_similar_patterns(current_pattern, top_k=5)
        
        print(f"\n找到 {len(similar_patterns)} 个相似的历史模式：")
        for idx, (pattern, similarity) in enumerate(similar_patterns[:3], 1):
            print(f"  模式{idx}: 相似度={similarity:.3f}, "
                  f"前区={pattern['front_positions']}, "
                  f"后区={pattern['back_positions']}")
        
        # 策略1：使用最相似模式的下一个
        if len(similar_patterns) > 0:
            best_pattern = similar_patterns[0][0]
            best_idx = best_pattern['period_idx']
            
            if best_idx < len(self.matcher.historical_patterns) - 1:
                next_pattern = self.matcher.historical_patterns[best_idx + 1]
                match_front = [int(round(n)) for n in next_pattern['front_positions']]
                match_back = [int(round(n)) for n in next_pattern['back_positions']]
            else:
                match_front = [int(round(n)) for n in best_pattern['front_positions']]
                match_back = [int(round(n)) for n in best_pattern['back_positions']]
        else:
            # 回退
            recent = self.matcher.historical_patterns[-1]
            match_front = [int(round(n)) for n in recent['front_positions']]
            match_back = [int(round(n)) for n in recent['back_positions']]
        
        # 策略2：基于趋势延续（预测下一期位置值）
        trend_front = []
        trend_back = []
        
        for pos in range(1, 6):
            values = position_trends['front_positions'][pos]
            # 使用线性回归预测下一期
            if len(values) >= 3:
                recent = values[-5:]
                # 简单线性外推
                trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0
                predicted = recent[-1] + trend
                trend_front.append(int(round(predicted)))
            else:
                trend_front.append(int(round(np.mean(values))))
        
        for pos in range(1, 3):
            values = position_trends['back_positions'][pos]
            if len(values) >= 3:
                recent = values[-5:]
                trend = (recent[-1] - recent[0]) / len(recent) if len(recent) > 1 else 0
                predicted = recent[-1] + trend
                trend_back.append(int(round(predicted)))
            else:
                trend_back.append(int(round(np.mean(values))))
        
        # 综合两种策略
        final_front = []
        final_back = []
        
        for i in range(5):
            # 加权平均
            combined = similarity_weight * match_front[i] + (1 - similarity_weight) * trend_front[i]
            final_front.append(int(round(combined)))
        
        for i in range(2):
            combined = similarity_weight * match_back[i] + (1 - similarity_weight) * trend_back[i]
            final_back.append(int(round(combined)))
        
        # 确保号码在有效范围内且不重复
        final_front = sorted(list(set([max(1, min(35, n)) for n in final_front])))
        final_back = sorted(list(set([max(1, min(12, n)) for n in final_back])))
        
        # 补充不足的号码（基于频率和趋势）
        while len(final_front) < 5:
            # 从趋势预测中补充
            if len(trend_front) > len(final_front):
                num = trend_front[len(final_front)]
            else:
                # 随机补充
                num = np.random.randint(1, 36)
            if num not in final_front and 1 <= num <= 35:
                final_front.append(num)
        final_front = sorted(final_front[:5])
        
        while len(final_back) < 2:
            if len(trend_back) > len(final_back):
                num = trend_back[len(final_back)]
            else:
                num = np.random.randint(1, 13)
            if num not in final_back and 1 <= num <= 12:
                final_back.append(num)
        final_back = sorted(final_back[:2])
        
        print(f"\n预测结果：")
        print(f"  前区: {final_front}")
        print(f"  后区: {final_back}")
        
        return final_front, final_back
    
    def predict(self, method: str = 'auto') -> Tuple[List[int], List[int]]:
        """
        综合预测方法
        
        Args:
            method: 预测方法
                - 'auto': 自动选择（优先ML，失败则用匹配）
                - 'ml': 使用机器学习模型
                - 'matching': 使用模式匹配
                - 'ensemble': 集成多种方法
        
        Returns:
            (前区号码, 后区号码)
        """
        if method == 'auto':
            if self.use_ml and self.ml_predictor and self.ml_predictor.is_trained:
                method = 'ml'
            else:
                method = 'matching'
        
        if method == 'ml':
            if self.use_ml and self.ml_predictor:
                return self.ml_predictor.predict()
            else:
                print("机器学习模型不可用，使用模式匹配")
                return self.predict_by_trend_matching()
        
        elif method == 'matching':
            return self.predict_by_trend_matching()
        
        elif method == 'ensemble':
            # 集成多种方法
            predictions = []
            
            # 方法1：模式匹配
            try:
                pred1 = self.predict_by_trend_matching()
                predictions.append(pred1)
            except Exception as e:
                print(f"模式匹配失败: {e}")
            
            # 方法2：机器学习（如果可用）
            if self.use_ml and self.ml_predictor and self.ml_predictor.is_trained:
                try:
                    pred2 = self.ml_predictor.predict()
                    predictions.append(pred2)
                except Exception as e:
                    print(f"机器学习预测失败: {e}")
            
            if len(predictions) == 0:
                # 回退到简单匹配
                return self.matcher.predict_by_matching()
            
            # 投票机制：选择出现频率最高的号码
            from collections import Counter
            
            front_counter = Counter()
            back_counter = Counter()
            
            for front, back in predictions:
                front_counter.update(front)
                back_counter.update(back)
            
            # 选择出现频率最高的号码
            final_front = sorted([num for num, _ in front_counter.most_common(5)])
            final_back = sorted([num for num, _ in back_counter.most_common(2)])
            
            # 补充不足的号码
            while len(final_front) < 5:
                num = np.random.randint(1, 36)
                if num not in final_front:
                    final_front.append(num)
            final_front = sorted(final_front[:5])
            
            while len(final_back) < 2:
                num = np.random.randint(1, 13)
                if num not in final_back:
                    final_back.append(num)
            final_back = sorted(final_back[:2])
            
            return final_front, final_back
        
        else:
            raise ValueError(f"未知的预测方法: {method}")
    
    def generate_trend_report(self, output_dir: str = './trend_analysis/'):
        """生成完整的走势分析报告"""
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        print("\n生成走势图...")
        image_paths = self.trend_analyzer.generate_trend_images(output_dir=output_dir)
        
        print("\n走势分析报告已生成：")
        for name, path in image_paths.items():
            print(f"  {name}: {path}")
        
        return image_paths
