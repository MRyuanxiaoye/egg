"""
大乐透数据可视化分析
需要安装 matplotlib: pip install matplotlib
"""

try:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.use('Agg')  # 非交互式后端
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("警告: matplotlib 未安装，可视化功能不可用")

from lottery_analyzer import LotteryAnalyzer
import numpy as np


def plot_frequency_distribution(analyzer: LotteryAnalyzer, save_path: str = 'frequency_plot.png'):
    """绘制号码频率分布图"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，无法绘制图表")
        return
    
    freq_data = analyzer.frequency_analysis()
    
    # 前区频率
    front_freq = freq_data['front_freq']
    front_nums = sorted(front_freq.keys())
    front_values = [front_freq[num] for num in front_nums]
    
    # 后区频率
    back_freq = freq_data['back_freq']
    back_nums = sorted(back_freq.keys())
    back_values = [back_freq[num] for num in back_nums]
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # 前区频率图
    ax1.bar(front_nums, front_values, color='steelblue', alpha=0.7)
    ax1.axhline(y=np.mean(front_values), color='r', linestyle='--', label='平均频率')
    ax1.set_xlabel('前区号码')
    ax1.set_ylabel('出现频率')
    ax1.set_title('前区号码出现频率分布')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 后区频率图
    ax2.bar(back_nums, back_values, color='coral', alpha=0.7)
    ax2.axhline(y=np.mean(back_values), color='r', linestyle='--', label='平均频率')
    ax2.set_xlabel('后区号码')
    ax2.set_ylabel('出现频率')
    ax2.set_title('后区号码出现频率分布')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"频率分布图已保存至: {save_path}")
    plt.close()


def plot_missing_value_heatmap(analyzer: LotteryAnalyzer, save_path: str = 'missing_heatmap.png'):
    """绘制遗漏值热力图"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，无法绘制图表")
        return
    
    missing_data = analyzer.missing_value_analysis()
    front_missing = missing_data['front_missing']
    
    # 准备数据
    nums = sorted([n for n in front_missing.keys() if front_missing[n] is not None])
    missing_values = [front_missing[n] for n in nums]
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    # 创建热力图数据（将号码排列成矩阵形式）
    # 前区1-35，可以排列成5行7列
    heatmap_data = np.zeros((5, 7))
    for i, num in enumerate(nums):
        row = (num - 1) // 7
        col = (num - 1) % 7
        if row < 5:
            heatmap_data[row, col] = front_missing[num] if front_missing[num] else 0
    
    im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto')
    
    # 添加数值标注
    for i in range(5):
        for j in range(7):
            num = i * 7 + j + 1
            if num <= 35 and front_missing.get(num) is not None:
                text = ax.text(j, i, f"{num}\n{int(front_missing[num])}",
                             ha="center", va="center", color="black", fontsize=8)
    
    ax.set_title('前区号码遗漏值热力图（数字下方为遗漏期数）')
    ax.set_xlabel('列')
    ax.set_ylabel('行')
    
    # 设置x轴标签为号码
    ax.set_xticks([])
    ax.set_yticks([])
    
    plt.colorbar(im, ax=ax, label='遗漏期数')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"遗漏值热力图已保存至: {save_path}")
    plt.close()


def plot_sum_value_distribution(analyzer: LotteryAnalyzer, save_path: str = 'sum_value_plot.png'):
    """绘制和值分布图"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，无法绘制图表")
        return
    
    sum_data = analyzer.sum_value_analysis()
    
    # 获取所有和值
    sums = []
    for row in analyzer.get_front_numbers():
        sums.append(sum(row))
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 直方图
    ax.hist(sums, bins=30, color='skyblue', alpha=0.7, edgecolor='black')
    ax.axvline(x=sum_data['mean'], color='r', linestyle='--', linewidth=2, label=f"均值: {sum_data['mean']:.1f}")
    ax.axvline(x=sum_data['median'], color='g', linestyle='--', linewidth=2, label=f"中位数: {sum_data['median']:.1f}")
    
    ax.set_xlabel('和值')
    ax.set_ylabel('出现次数')
    ax.set_title('前区号码和值分布')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"和值分布图已保存至: {save_path}")
    plt.close()


def plot_ratio_distribution(analyzer: LotteryAnalyzer, save_path: str = 'ratio_plot.png'):
    """绘制奇偶比和大小比分布图"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，无法绘制图表")
        return
    
    odd_even = analyzer.odd_even_ratio_analysis()
    size_ratio = analyzer.size_ratio_analysis()
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # 奇偶比
    oe_ratios = sorted(odd_even['ratio_distribution'].keys())
    oe_counts = [odd_even['ratio_distribution'][r] for r in oe_ratios]
    ax1.bar(oe_ratios, oe_counts, color='lightblue', alpha=0.7, edgecolor='black')
    ax1.set_xlabel('奇偶比')
    ax1.set_ylabel('出现次数')
    ax1.set_title('奇偶比分布')
    ax1.grid(True, alpha=0.3, axis='y')
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
    
    # 大小比
    sr_ratios = sorted(size_ratio['ratio_distribution'].keys())
    sr_counts = [size_ratio['ratio_distribution'][r] for r in sr_ratios]
    ax2.bar(sr_ratios, sr_counts, color='lightcoral', alpha=0.7, edgecolor='black')
    ax2.set_xlabel('大小比')
    ax2.set_ylabel('出现次数')
    ax2.set_title('大小比分布')
    ax2.grid(True, alpha=0.3, axis='y')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"比例分布图已保存至: {save_path}")
    plt.close()


def generate_all_visualizations(analyzer: LotteryAnalyzer):
    """生成所有可视化图表"""
    if not HAS_MATPLOTLIB:
        print("matplotlib 未安装，跳过可视化")
        return
    
    print("\n开始生成可视化图表...")
    plot_frequency_distribution(analyzer)
    plot_missing_value_heatmap(analyzer)
    plot_sum_value_distribution(analyzer)
    plot_ratio_distribution(analyzer)
    print("\n所有可视化图表生成完成！")


if __name__ == "__main__":
    print("大乐透数据可视化工具")
    print("=" * 50)
    print("\n使用方法：")
    print("1. 安装依赖: pip install matplotlib")
    print("2. 加载数据并进行分析")
    print("3. 调用可视化函数生成图表")
    print("\n示例：")
    print("  from lottery_analyzer import LotteryAnalyzer")
    print("  from visualization import generate_all_visualizations")
    print("  analyzer = LotteryAnalyzer('lottery_history.csv')")
    print("  generate_all_visualizations(analyzer)")
