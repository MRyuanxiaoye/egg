"""
使用训练好的模型进行预测
"""

import pandas as pd
import torch
import json
from pathlib import Path
from trend_based_predictor import TrendBasedPredictor

def load_trained_model(model_path: str, data: pd.DataFrame):
    """加载训练好的模型"""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    model_type = checkpoint.get('model_type', 'lstm')
    config = checkpoint.get('config', {})
    
    # 创建预测器
    predictor = TrendBasedPredictor(
        data=data,
        use_ml=True,
        model_type=model_type
    )
    
    # 加载模型
    if predictor.ml_predictor:
        predictor.ml_predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.ml_predictor.is_trained = True
        predictor.ml_predictor.scaler = checkpoint.get('scaler')
    
    return predictor

def predict_with_models(data_file: str = None, data: pd.DataFrame = None,
                        model_dir: str = './ml_models/'):
    """
    使用训练好的模型进行预测
    
    Args:
        data_file: 数据文件路径
        data: 数据DataFrame
        model_dir: 模型目录
    """
    # 加载数据
    if data is not None:
        data_df = data
    elif data_file:
        data_df = pd.read_csv(data_file)
    else:
        print("错误: 需要提供数据文件或数据DataFrame")
        return
    
    model_dir = Path(model_dir)
    
    # 查找所有模型文件
    model_files = list(model_dir.glob('*_model.pth'))
    
    if not model_files:
        print(f"在 {model_dir} 中未找到训练好的模型")
        print("请先运行 ml_training_pipeline.py 训练模型")
        return
    
    print(f"找到 {len(model_files)} 个训练好的模型")
    
    predictions = {}
    
    for model_file in model_files:
        model_type = model_file.stem.replace('_model', '')
        print(f"\n使用 {model_type.upper()} 模型预测...")
        
        try:
            predictor = load_trained_model(str(model_file), data_df)
            front, back = predictor.predict(method='ml')
            predictions[model_type] = (front, back)
            print(f"  前区: {front}")
            print(f"  后区: {back}")
        except Exception as e:
            print(f"  ✗ 预测失败: {e}")
    
    # 集成预测
    if len(predictions) > 1:
        print(f"\n集成预测（综合 {len(predictions)} 个模型）:")
        from collections import Counter
        
        front_counter = Counter()
        back_counter = Counter()
        
        for front, back in predictions.values():
            front_counter.update(front)
            back_counter.update(back)
        
        ensemble_front = sorted([num for num, _ in front_counter.most_common(5)])
        ensemble_back = sorted([num for num, _ in back_counter.most_common(2)])
        
        print(f"  前区: {ensemble_front}")
        print(f"  后区: {ensemble_back}")
        
        predictions['ensemble'] = (ensemble_front, ensemble_back)
    
    # 保存预测结果
    pred_dict = {
        model_type: {
            'front': list(front),
            'back': list(back)
        }
        for model_type, (front, back) in predictions.items()
    }
    
    output_file = model_dir / 'latest_predictions.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(pred_dict, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ 预测结果已保存至: {output_file}")
    
    return predictions

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='使用训练好的模型进行预测')
    parser.add_argument('--data', type=str, help='数据文件路径')
    parser.add_argument('--model-dir', type=str, default='./ml_models/',
                       help='模型目录')
    
    args = parser.parse_args()
    
    predict_with_models(data_file=args.data, model_dir=args.model_dir)
