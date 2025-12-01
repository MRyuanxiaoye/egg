"""
快速开始：机器学习训练和预测
最简单的使用方式
"""

import pandas as pd
import numpy as np
from ml_training_pipeline import MLTrainingPipeline

def main():
    """快速开始示例"""
    
    print("=" * 70)
    print("大乐透机器学习 - 快速开始")
    print("=" * 70)
    
    # 方式1: 使用你的数据文件
    # data_file = 'lottery_history.csv'  # 替换为你的数据文件路径
    # pipeline = MLTrainingPipeline(data_file=data_file)
    
    # 方式2: 使用示例数据（自动生成）
    print("\n使用示例数据（如果没有提供数据文件）")
    pipeline = MLTrainingPipeline()
    
    # 运行完整流程
    print("\n开始训练和预测...")
    pipeline.run_full_pipeline(
        model_types=['lstm'],  # 可以添加 'transformer'
        train=True,            # 是否训练
        evaluate=True,         # 是否评估
        predict=True          # 是否预测
    )
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)
    print("\n下一步:")
    print("1. 查看模型文件: ./ml_models/")
    print("2. 查看预测结果: ./ml_models/predictions.json")
    print("3. 使用训练好的模型进行预测（见 ml_prediction.py）")

if __name__ == "__main__":
    main()
