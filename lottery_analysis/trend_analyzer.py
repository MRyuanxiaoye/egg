"""
大乐透走势图分析工具
生成横向（位置）和纵向（组合）走势图
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
from collections import defaultdict


class TrendAnalyzer:
    """走势图分析器"""
    
    def __init__(self, data: pd.DataFrame):
        """
        初始化走势分析器
        
        Args:
            data: 历史数据DataFrame，包含期号和号码列
        """
        self.data = data.sort_values('期号').reset_index(drop=True)
        self.periods = len(data)
        
    def get_position_trends(self, window_size: int = 20) -> Dict:
        """
        横向走势分析：分析每个位置的号码走势
        
        Args:
            window_size: 滑动窗口大小，用于平滑走势
        
        Returns:
            {
                'front_positions': {
                    1: [每期该位置的号码],
                    2: [每期该位置的号码],
                    ...
                },
                'back_positions': {
                    1: [每期该位置的号码],
                    2: [每期该位置的号码]
                },
                'trend_features': {
                    'position_mean': 每个位置的平均值,
                    'position_std': 每个位置的标准差,
                    'position_trend': 每个位置的趋势（上升/下降/平稳）
                }
            }
        """
        front_positions = {i: [] for i in range(1, 6)}
        back_positions = {i: [] for i in range(1, 3)}
        
        # 提取每个位置的号码
        for _, row in self.data.iterrows():
            # 前区（需要排序）
            front_nums = sorted([row[f'前区{i}'] for i in range(1, 6)])
            for pos, num in enumerate(front_nums, 1):
                front_positions[pos].append(num)
            
            # 后区（需要排序）
            back_nums = sorted([row[f'后区{i}'] for i in range(1, 3)])
            for pos, num in enumerate(back_nums, 1):
                back_positions[pos].append(num)
        
        # 计算趋势特征
        trend_features = self._calculate_trend_features(
            front_positions, back_positions, window_size
        )
        
        return {
            'front_positions': front_positions,
            'back_positions': back_positions,
            'trend_features': trend_features
        }
    
    def _calculate_trend_features(self, front_positions: Dict, 
                                   back_positions: Dict, 
                                   window_size: int) -> Dict:
        """计算趋势特征"""
        features = {
            'position_mean': {},
            'position_std': {},
            'position_trend': {},
            'position_velocity': {},  # 变化速度
            'position_acceleration': {}  # 加速度
        }
        
        # 前区位置特征
        for pos in range(1, 6):
            values = np.array(front_positions[pos])
            features['position_mean'][f'front_{pos}'] = np.mean(values)
            features['position_std'][f'front_{pos}'] = np.std(values)
            
            # 计算趋势（最近N期 vs 之前N期）
            if len(values) >= window_size * 2:
                recent_mean = np.mean(values[-window_size:])
                previous_mean = np.mean(values[-window_size*2:-window_size])
                if recent_mean > previous_mean * 1.05:
                    features['position_trend'][f'front_{pos}'] = '上升'
                elif recent_mean < previous_mean * 0.95:
                    features['position_trend'][f'front_{pos}'] = '下降'
                else:
                    features['position_trend'][f'front_{pos}'] = '平稳'
            else:
                features['position_trend'][f'front_{pos}'] = '数据不足'
            
            # 计算速度（一阶差分）
            if len(values) > 1:
                velocity = np.diff(values)
                features['position_velocity'][f'front_{pos}'] = np.mean(velocity[-window_size:])
                
                # 计算加速度（二阶差分）
                if len(velocity) > 1:
                    acceleration = np.diff(velocity)
                    features['position_acceleration'][f'front_{pos}'] = np.mean(acceleration[-window_size:])
        
        # 后区位置特征
        for pos in range(1, 3):
            values = np.array(back_positions[pos])
            features['position_mean'][f'back_{pos}'] = np.mean(values)
            features['position_std'][f'back_{pos}'] = np.std(values)
            
            if len(values) >= window_size * 2:
                recent_mean = np.mean(values[-window_size:])
                previous_mean = np.mean(values[-window_size*2:-window_size])
                if recent_mean > previous_mean * 1.05:
                    features['position_trend'][f'back_{pos}'] = '上升'
                elif recent_mean < previous_mean * 0.95:
                    features['position_trend'][f'back_{pos}'] = '下降'
                else:
                    features['position_trend'][f'back_{pos}'] = '平稳'
            else:
                features['position_trend'][f'back_{pos}'] = '数据不足'
            
            if len(values) > 1:
                velocity = np.diff(values)
                features['position_velocity'][f'back_{pos}'] = np.mean(velocity[-window_size:])
                
                if len(velocity) > 1:
                    acceleration = np.diff(velocity)
                    features['position_acceleration'][f'back_{pos}'] = np.mean(acceleration[-window_size:])
        
        return features
    
    def get_combination_trends(self, window_size: int = 20) -> Dict:
        """
        纵向走势分析：分析整个号码组合的走势
        
        Returns:
            {
                'sum_trend': 和值走势,
                'span_trend': 跨度走势,
                'odd_even_trend': 奇偶比走势,
                'size_ratio_trend': 大小比走势,
                'consecutive_trend': 连号走势,
                'combination_patterns': 组合模式
            }
        """
        trends = {
            'sum_trend': [],
            'span_trend': [],
            'odd_even_trend': [],
            'size_ratio_trend': [],
            'consecutive_trend': [],
            'front_sequence': [],  # 前区序列特征
            'back_sequence': []    # 后区序列特征
        }
        
        for _, row in self.data.iterrows():
            # 前区号码（排序后）
            front = sorted([row[f'前区{i}'] for i in range(1, 6)])
            back = sorted([row[f'后区{i}'] for i in range(1, 3)])
            
            # 和值
            trends['sum_trend'].append(sum(front))
            
            # 跨度
            trends['span_trend'].append(front[-1] - front[0])
            
            # 奇偶比
            odd_count = sum(1 for n in front if n % 2 == 1)
            trends['odd_even_trend'].append(odd_count)
            
            # 大小比
            small_count = sum(1 for n in front if n <= 17)
            trends['size_ratio_trend'].append(small_count)
            
            # 连号数量
            consecutive = sum(1 for i in range(len(front)-1) if front[i+1] - front[i] == 1)
            trends['consecutive_trend'].append(consecutive)
            
            # 序列特征（差值序列）
            front_diff = [front[i+1] - front[i] for i in range(len(front)-1)]
            back_diff = [back[1] - back[0]] if len(back) > 1 else []
            trends['front_sequence'].append(front_diff)
            trends['back_sequence'].append(back_diff)
        
        # 转换为numpy数组
        for key in ['sum_trend', 'span_trend', 'odd_even_trend', 
                   'size_ratio_trend', 'consecutive_trend']:
            trends[key] = np.array(trends[key])
        
        # 分析组合模式
        trends['combination_patterns'] = self._analyze_combination_patterns(trends)
        
        return trends
    
    def _analyze_combination_patterns(self, trends: Dict) -> Dict:
        """分析组合模式"""
        patterns = {}
        
        # 和值模式
        sum_values = trends['sum_trend']
        patterns['sum_mean'] = np.mean(sum_values)
        patterns['sum_std'] = np.std(sum_values)
        patterns['sum_range'] = (np.min(sum_values), np.max(sum_values))
        
        # 跨度模式
        span_values = trends['span_trend']
        patterns['span_mean'] = np.mean(span_values)
        patterns['span_std'] = np.std(span_values)
        
        # 奇偶比模式
        oe_values = trends['odd_even_trend']
        patterns['odd_even_dist'] = dict(zip(*np.unique(oe_values, return_counts=True)))
        
        # 大小比模式
        sr_values = trends['size_ratio_trend']
        patterns['size_ratio_dist'] = dict(zip(*np.unique(sr_values, return_counts=True)))
        
        return patterns
    
    def generate_trend_images(self, output_dir: str = './trend_images/', 
                             lookback_periods: int = 50) -> Dict[str, str]:
        """
        生成走势图图像（用于机器学习）
        
        Args:
            output_dir: 输出目录
            lookback_periods: 用于生成走势图的期数
        
        Returns:
            图像文件路径字典
        """
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        position_trends = self.get_position_trends()
        combination_trends = self.get_combination_trends()
        
        image_paths = {}
        
        # 1. 前区位置走势图（横向）
        fig, axes = plt.subplots(5, 1, figsize=(12, 10))
        recent_data = self.data.tail(lookback_periods)
        
        for pos in range(1, 6):
            ax = axes[pos-1]
            values = position_trends['front_positions'][pos][-lookback_periods:]
            periods = list(range(len(values)))
            
            ax.plot(periods, values, marker='o', markersize=3, linewidth=1.5, 
                   label=f'位置{pos}走势')
            ax.set_title(f'前区位置{pos}号码走势图（最近{lookback_periods}期）')
            ax.set_xlabel('期数（相对）')
            ax.set_ylabel('号码值')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        front_pos_path = os.path.join(output_dir, 'front_position_trends.png')
        plt.savefig(front_pos_path, dpi=150, bbox_inches='tight')
        plt.close()
        image_paths['front_positions'] = front_pos_path
        
        # 2. 后区位置走势图
        fig, axes = plt.subplots(2, 1, figsize=(12, 6))
        for pos in range(1, 3):
            ax = axes[pos-1]
            values = position_trends['back_positions'][pos][-lookback_periods:]
            periods = list(range(len(values)))
            
            ax.plot(periods, values, marker='o', markersize=3, linewidth=1.5,
                   label=f'位置{pos}走势', color='coral')
            ax.set_title(f'后区位置{pos}号码走势图（最近{lookback_periods}期）')
            ax.set_xlabel('期数（相对）')
            ax.set_ylabel('号码值')
            ax.grid(True, alpha=0.3)
            ax.legend()
        
        plt.tight_layout()
        back_pos_path = os.path.join(output_dir, 'back_position_trends.png')
        plt.savefig(back_pos_path, dpi=150, bbox_inches='tight')
        plt.close()
        image_paths['back_positions'] = back_pos_path
        
        # 3. 组合特征走势图（纵向）
        fig, axes = plt.subplots(3, 2, figsize=(14, 10))
        axes = axes.flatten()
        
        trend_names = ['sum_trend', 'span_trend', 'odd_even_trend', 
                      'size_ratio_trend', 'consecutive_trend']
        trend_labels = ['和值走势', '跨度走势', '奇偶比走势', 
                       '大小比走势', '连号走势']
        
        for idx, (name, label) in enumerate(zip(trend_names, trend_labels)):
            ax = axes[idx]
            values = combination_trends[name][-lookback_periods:]
            periods = list(range(len(values)))
            
            ax.plot(periods, values, marker='o', markersize=2, linewidth=1.5)
            ax.set_title(f'{label}（最近{lookback_periods}期）')
            ax.set_xlabel('期数（相对）')
            ax.set_ylabel('特征值')
            ax.grid(True, alpha=0.3)
        
        # 隐藏最后一个子图
        axes[5].axis('off')
        
        plt.tight_layout()
        combo_trend_path = os.path.join(output_dir, 'combination_trends.png')
        plt.savefig(combo_trend_path, dpi=150, bbox_inches='tight')
        plt.close()
        image_paths['combination_trends'] = combo_trend_path
        
        # 4. 综合走势图（所有位置叠加）
        fig, ax = plt.subplots(figsize=(14, 8))
        colors = plt.cm.tab10(np.linspace(0, 1, 5))
        
        for pos in range(1, 6):
            values = position_trends['front_positions'][pos][-lookback_periods:]
            periods = list(range(len(values)))
            ax.plot(periods, values, marker='o', markersize=2, linewidth=1.5,
                   label=f'位置{pos}', color=colors[pos-1], alpha=0.7)
        
        ax.set_title(f'前区所有位置综合走势图（最近{lookback_periods}期）')
        ax.set_xlabel('期数（相对）')
        ax.set_ylabel('号码值')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        comprehensive_path = os.path.join(output_dir, 'comprehensive_trend.png')
        plt.savefig(comprehensive_path, dpi=150, bbox_inches='tight')
        plt.close()
        image_paths['comprehensive'] = comprehensive_path
        
        print(f"走势图已生成，保存在: {output_dir}")
        return image_paths
    
    def extract_trend_features_for_ml(self, lookback: int = 30) -> np.ndarray:
        """
        提取用于机器学习的走势特征
        
        Args:
            lookback: 回看期数
        
        Returns:
            特征矩阵 shape: (periods, features)
        """
        position_trends = self.get_position_trends()
        combination_trends = self.get_combination_trends()
        
        features_list = []
        
        # 使用最近lookback期数据
        recent_periods = min(lookback, self.periods)
        
        for i in range(recent_periods):
            period_idx = self.periods - recent_periods + i
            feature_vector = []
            
            # 1. 前区位置特征（5个位置，每个位置的值）
            for pos in range(1, 6):
                values = position_trends['front_positions'][pos]
                if period_idx < len(values):
                    feature_vector.append(values[period_idx])
                else:
                    feature_vector.append(0)
            
            # 2. 后区位置特征（2个位置）
            for pos in range(1, 3):
                values = position_trends['back_positions'][pos]
                if period_idx < len(values):
                    feature_vector.append(values[period_idx])
                else:
                    feature_vector.append(0)
            
            # 3. 组合特征
            feature_vector.append(combination_trends['sum_trend'][period_idx])
            feature_vector.append(combination_trends['span_trend'][period_idx])
            feature_vector.append(combination_trends['odd_even_trend'][period_idx])
            feature_vector.append(combination_trends['size_ratio_trend'][period_idx])
            feature_vector.append(combination_trends['consecutive_trend'][period_idx])
            
            # 4. 位置变化率（一阶差分）
            if period_idx > 0:
                for pos in range(1, 6):
                    prev_val = position_trends['front_positions'][pos][period_idx-1]
                    curr_val = position_trends['front_positions'][pos][period_idx]
                    feature_vector.append(curr_val - prev_val)
            else:
                feature_vector.extend([0] * 5)
            
            # 5. 组合特征变化率
            if period_idx > 0:
                feature_vector.append(combination_trends['sum_trend'][period_idx] - 
                                    combination_trends['sum_trend'][period_idx-1])
                feature_vector.append(combination_trends['span_trend'][period_idx] - 
                                    combination_trends['span_trend'][period_idx-1])
            else:
                feature_vector.extend([0] * 2)
            
            features_list.append(feature_vector)
        
        return np.array(features_list)
    
    def get_next_period_features(self) -> np.ndarray:
        """获取下一期的特征（用于预测）"""
        # 使用最近的数据提取特征
        features = self.extract_trend_features_for_ml(lookback=1)
        return features[-1] if len(features) > 0 else None
