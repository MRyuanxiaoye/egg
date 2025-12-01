"""
大乐透历史数据分析工具
提供多种统计分析功能
"""

import pandas as pd
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import json


class LotteryAnalyzer:
    """大乐透数据分析器"""
    
    def __init__(self, data_file: str = None):
        """
        初始化分析器
        
        Args:
            data_file: 历史数据文件路径（CSV格式）
                      格式：期号,前区1,前区2,前区3,前区4,前区5,后区1,后区2
        """
        self.data = None
        if data_file:
            self.load_data(data_file)
    
    def load_data(self, file_path: str):
        """加载历史数据"""
        try:
            self.data = pd.read_csv(file_path)
            print(f"成功加载 {len(self.data)} 期历史数据")
        except Exception as e:
            print(f"数据加载失败: {e}")
            # 创建示例数据结构
            self.data = pd.DataFrame(columns=['期号', '前区1', '前区2', '前区3', '前区4', '前区5', '后区1', '后区2'])
    
    def get_front_numbers(self) -> List[List[int]]:
        """获取所有前区号码组合"""
        if self.data is None:
            return []
        front_cols = ['前区1', '前区2', '前区3', '前区4', '前区5']
        return self.data[front_cols].values.tolist()
    
    def get_back_numbers(self) -> List[List[int]]:
        """获取所有后区号码组合"""
        if self.data is None:
            return []
        back_cols = ['后区1', '后区2']
        return self.data[back_cols].values.tolist()
    
    def frequency_analysis(self) -> Dict:
        """
        频率分析：统计每个号码的出现频率
        
        Returns:
            {
                'front': {号码: 出现次数},
                'back': {号码: 出现次数},
                'front_freq': {号码: 出现频率},
                'back_freq': {号码: 出现频率}
            }
        """
        front_numbers = []
        back_numbers = []
        
        for row in self.get_front_numbers():
            front_numbers.extend(row)
        
        for row in self.get_back_numbers():
            back_numbers.extend(row)
        
        front_counter = Counter(front_numbers)
        back_counter = Counter(back_numbers)
        
        total_periods = len(self.data)
        
        front_freq = {num: count / total_periods for num, count in front_counter.items()}
        back_freq = {num: count / total_periods for num, count in back_counter.items()}
        
        return {
            'front': dict(front_counter),
            'back': dict(back_counter),
            'front_freq': front_freq,
            'back_freq': back_freq
        }
    
    def cold_hot_analysis(self, recent_periods: int = 50) -> Dict:
        """
        冷热号分析
        
        Args:
            recent_periods: 最近期数，用于判断热号
        
        Returns:
            {
                'hot_front': [热号列表],
                'cold_front': [冷号列表],
                'hot_back': [后区热号],
                'cold_back': [后区冷号]
            }
        """
        freq_data = self.frequency_analysis()
        
        # 最近期数的频率
        recent_data = self.data.tail(recent_periods)
        recent_front = []
        recent_back = []
        
        for _, row in recent_data.iterrows():
            recent_front.extend([row['前区1'], row['前区2'], row['前区3'], row['前区4'], row['前区5']])
            recent_back.extend([row['后区1'], row['后区2']])
        
        recent_front_counter = Counter(recent_front)
        recent_back_counter = Counter(recent_back)
        
        # 热号：最近出现频率高
        hot_front = [num for num, count in recent_front_counter.most_common(10)]
        hot_back = [num for num, count in recent_back_counter.most_common(5)]
        
        # 冷号：总频率低或最近未出现
        all_front = set(range(1, 36))
        all_back = set(range(1, 13))
        
        cold_front = sorted([num for num in all_front if num not in recent_front_counter])
        cold_back = sorted([num for num in all_back if num not in recent_back_counter])
        
        return {
            'hot_front': hot_front,
            'cold_front': cold_front,
            'hot_back': hot_back,
            'cold_back': cold_back
        }
    
    def missing_value_analysis(self) -> Dict:
        """
        遗漏值分析：计算每个号码距离上次出现的期数
        
        Returns:
            {
                'front_missing': {号码: 遗漏期数},
                'back_missing': {号码: 遗漏期数}
            }
        """
        front_missing = {}
        back_missing = {}
        
        # 初始化所有号码
        for num in range(1, 36):
            front_missing[num] = None
        
        for num in range(1, 13):
            back_missing[num] = None
        
        # 从最新期往前查找
        for idx, row in self.data.iterrows():
            period_num = len(self.data) - idx
            
            # 前区
            for col in ['前区1', '前区2', '前区3', '前区4', '前区5']:
                num = row[col]
                if front_missing[num] is None:
                    front_missing[num] = period_num
            
            # 后区
            for col in ['后区1', '后区2']:
                num = row[col]
                if back_missing[num] is None:
                    back_missing[num] = period_num
        
        return {
            'front_missing': front_missing,
            'back_missing': back_missing
        }
    
    def odd_even_ratio_analysis(self) -> Dict:
        """
        奇偶比分析
        
        Returns:
            {
                'ratio_distribution': {奇偶比: 出现次数},
                'most_common_ratio': 最常见的奇偶比
            }
        """
        ratios = []
        
        for row in self.get_front_numbers():
            odd_count = sum(1 for num in row if num % 2 == 1)
            even_count = 5 - odd_count
            ratio = f"{odd_count}:{even_count}"
            ratios.append(ratio)
        
        ratio_counter = Counter(ratios)
        most_common = ratio_counter.most_common(1)[0][0]
        
        return {
            'ratio_distribution': dict(ratio_counter),
            'most_common_ratio': most_common
        }
    
    def size_ratio_analysis(self) -> Dict:
        """
        大小比分析（1-17为小，18-35为大）
        
        Returns:
            {
                'ratio_distribution': {大小比: 出现次数},
                'most_common_ratio': 最常见的大小比
            }
        """
        ratios = []
        
        for row in self.get_front_numbers():
            small_count = sum(1 for num in row if num <= 17)
            large_count = 5 - small_count
            ratio = f"{small_count}:{large_count}"
            ratios.append(ratio)
        
        ratio_counter = Counter(ratios)
        most_common = ratio_counter.most_common(1)[0][0]
        
        return {
            'ratio_distribution': dict(ratio_counter),
            'most_common_ratio': most_common
        }
    
    def sum_value_analysis(self) -> Dict:
        """
        和值分析：前区5个号码的和值
        
        Returns:
            {
                'mean': 平均和值,
                'median': 中位数和值,
                'std': 标准差,
                'min': 最小和值,
                'max': 最大和值,
                'distribution': 和值分布
            }
        """
        sums = []
        
        for row in self.get_front_numbers():
            sums.append(sum(row))
        
        return {
            'mean': np.mean(sums),
            'median': np.median(sums),
            'std': np.std(sums),
            'min': min(sums),
            'max': max(sums),
            'distribution': dict(Counter(sums))
        }
    
    def span_analysis(self) -> Dict:
        """
        跨度分析：前区最大号与最小号的差值
        
        Returns:
            {
                'mean': 平均跨度,
                'median': 中位数跨度,
                'std': 标准差,
                'distribution': 跨度分布
            }
        """
        spans = []
        
        for row in self.get_front_numbers():
            spans.append(max(row) - min(row))
        
        return {
            'mean': np.mean(spans),
            'median': np.median(spans),
            'std': np.std(spans),
            'distribution': dict(Counter(spans))
        }
    
    def consecutive_analysis(self) -> Dict:
        """
        连号分析：统计连续号码的出现频率
        
        Returns:
            {
                'consecutive_count': 每期连号数量分布,
                'consecutive_pairs': 常见连号对
            }
        """
        consecutive_counts = []
        consecutive_pairs = []
        
        for row in self.get_front_numbers():
            sorted_row = sorted(row)
            count = 0
            for i in range(len(sorted_row) - 1):
                if sorted_row[i+1] - sorted_row[i] == 1:
                    count += 1
                    consecutive_pairs.append((sorted_row[i], sorted_row[i+1]))
            consecutive_counts.append(count)
        
        return {
            'consecutive_count': dict(Counter(consecutive_counts)),
            'consecutive_pairs': dict(Counter(consecutive_pairs))
        }
    
    def comprehensive_analysis(self) -> Dict:
        """综合分析：汇总所有分析结果"""
        return {
            'frequency': self.frequency_analysis(),
            'cold_hot': self.cold_hot_analysis(),
            'missing': self.missing_value_analysis(),
            'odd_even': self.odd_even_ratio_analysis(),
            'size_ratio': self.size_ratio_analysis(),
            'sum_value': self.sum_value_analysis(),
            'span': self.span_analysis(),
            'consecutive': self.consecutive_analysis()
        }
    
    def generate_report(self, output_file: str = 'analysis_report.json'):
        """生成分析报告"""
        analysis = self.comprehensive_analysis()
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)
        
        print(f"分析报告已保存至: {output_file}")
        return analysis


if __name__ == "__main__":
    # 示例用法
    analyzer = LotteryAnalyzer()
    
    # 如果有数据文件，可以这样加载：
    # analyzer.load_data('lottery_history.csv')
    
    print("大乐透数据分析工具")
    print("=" * 50)
    print("\n使用方法：")
    print("1. 准备CSV格式的历史数据文件")
    print("2. 使用 analyzer.load_data('数据文件路径') 加载数据")
    print("3. 调用各种分析方法进行分析")
    print("\n示例：")
    print("  analyzer = LotteryAnalyzer('lottery_history.csv')")
    print("  freq = analyzer.frequency_analysis()")
    print("  cold_hot = analyzer.cold_hot_analysis()")
    print("  report = analyzer.generate_report()")
