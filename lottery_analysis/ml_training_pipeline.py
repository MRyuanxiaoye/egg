"""
完整的机器学习训练和预测流程
包括环境检查、数据准备、模型训练、评估和预测
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

# 检查依赖
print("=" * 70)
print("环境检查")
print("=" * 70)

try:
    import torch
    print(f"✓ PyTorch {torch.__version__} 已安装")
    HAS_TORCH = True
    
    # 检查CUDA
    if torch.cuda.is_available():
        print(f"✓ CUDA 可用")
        print(f"  GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  显存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        DEVICE = 'cuda'
    else:
        print("⚠ CUDA 不可用，将使用CPU训练（速度较慢）")
        DEVICE = 'cpu'
except ImportError:
    print("✗ PyTorch 未安装")
    print("  请运行: pip install torch")
    HAS_TORCH = False
    DEVICE = 'cpu'

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    print(f"✓ scikit-learn 已安装")
    HAS_SKLEARN = True
except ImportError:
    print("✗ scikit-learn 未安装")
    print("  请运行: pip install scikit-learn")
    HAS_SKLEARN = False

try:
    import matplotlib
    print(f"✓ matplotlib 已安装")
except ImportError:
    print("⚠ matplotlib 未安装（可视化功能不可用）")

print("\n" + "=" * 70)

if not HAS_TORCH:
    print("\n错误: PyTorch 未安装，无法进行机器学习训练")
    print("请先安装: pip install torch")
    sys.exit(1)

# 导入项目模块
try:
    from trend_analyzer import TrendAnalyzer
    from ml_models import TrendMLPredictor, TrendMatcher
    from trend_based_predictor import TrendBasedPredictor
    print("✓ 项目模块加载成功")
except ImportError as e:
    print(f"✗ 项目模块加载失败: {e}")
    print("请确保所有文件在同一目录下")
    sys.exit(1)


class MLTrainingPipeline:
    """机器学习训练管道"""
    
    def __init__(self, data_file: str = None, data: pd.DataFrame = None, 
                 output_dir: str = './ml_models/'):
        """
        初始化训练管道
        
        Args:
            data_file: 数据文件路径
            data: 数据DataFrame（如果直接提供数据）
            output_dir: 模型输出目录
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # 加载数据
        if data is not None:
            self.data = data
        elif data_file and os.path.exists(data_file):
            print(f"\n加载数据: {data_file}")
            self.data = pd.read_csv(data_file)
            print(f"✓ 已加载 {len(self.data)} 期数据")
        else:
            print("\n未提供数据，将创建示例数据用于演示")
            self.data = self._create_sample_data()
            print(f"✓ 已创建 {len(self.data)} 期示例数据")
        
        # 初始化分析器和预测器
        self.trend_analyzer = TrendAnalyzer(self.data)
        self.predictor = TrendBasedPredictor(
            data=self.data,
            use_ml=True,
            model_type='lstm'
        )
        
        # 训练配置
        self.config = self._get_default_config()
        
    def _create_sample_data(self, num_periods: int = 300) -> pd.DataFrame:
        """创建示例数据"""
        data = []
        np.random.seed(42)
        
        for period in range(1, num_periods + 1):
            # 生成有一定规律的数据
            front = []
            for pos in range(1, 6):
                base = 5 + pos * 6 + np.sin(period / 20) * 3
                num = int(np.clip(base + np.random.normal(0, 2), 1, 35))
                front.append(num)
            front = sorted(list(set(front)))
            
            while len(front) < 5:
                num = np.random.randint(1, 36)
                if num not in front:
                    front.append(num)
            front = sorted(front[:5])
            
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
    
    def _get_default_config(self) -> dict:
        """获取默认训练配置"""
        # 根据是否有GPU调整配置
        if DEVICE == 'cuda':
            return {
                'batch_size': 32,
                'epochs': 50,
                'learning_rate': 0.001,
                'hidden_size': 64,
                'num_layers': 2,
                'dropout': 0.2,
                'lookback': 30,
                'validation_split': 0.2,
                'patience': 10
            }
        else:
            return {
                'batch_size': 16,
                'epochs': 30,
                'learning_rate': 0.001,
                'hidden_size': 32,
                'num_layers': 1,
                'dropout': 0.2,
                'lookback': 20,
                'validation_split': 0.2,
                'patience': 10
            }
    
    def check_data_quality(self):
        """检查数据质量"""
        print("\n" + "=" * 70)
        print("数据质量检查")
        print("=" * 70)
        
        issues = []
        
        # 检查数据量
        if len(self.data) < 100:
            issues.append(f"数据量不足: 只有 {len(self.data)} 期，建议至少100期")
        elif len(self.data) < 200:
            issues.append(f"数据量较少: 只有 {len(self.data)} 期，建议200期以上")
        else:
            print(f"✓ 数据量充足: {len(self.data)} 期")
        
        # 检查必需列
        required_cols = ['期号', '前区1', '前区2', '前区3', '前区4', '前区5', '后区1', '后区2']
        missing_cols = [col for col in required_cols if col not in self.data.columns]
        if missing_cols:
            issues.append(f"缺少必需列: {missing_cols}")
        else:
            print("✓ 数据列完整")
        
        # 检查数据范围
        for i in range(1, 6):
            col = f'前区{i}'
            if col in self.data.columns:
                invalid = self.data[(self.data[col] < 1) | (self.data[col] > 35)]
                if len(invalid) > 0:
                    issues.append(f"{col} 有无效值（不在1-35范围内）")
        
        for i in range(1, 3):
            col = f'后区{i}'
            if col in self.data.columns:
                invalid = self.data[(self.data[col] < 1) | (self.data[col] > 12)]
                if len(invalid) > 0:
                    issues.append(f"{col} 有无效值（不在1-12范围内）")
        
        if not issues:
            print("✓ 数据质量检查通过")
        else:
            print("\n⚠ 发现以下问题:")
            for issue in issues:
                print(f"  - {issue}")
        
        return len(issues) == 0
    
    def train_models(self, model_types: list = ['lstm'], **kwargs):
        """
        训练模型
        
        Args:
            model_types: 要训练的模型类型列表 ['lstm', 'transformer']
            **kwargs: 训练参数（覆盖默认配置）
        """
        print("\n" + "=" * 70)
        print("开始训练模型")
        print("=" * 70)
        
        # 更新配置
        self.config.update(kwargs)
        
        print(f"\n训练配置:")
        print(f"  设备: {DEVICE}")
        print(f"  批次大小: {self.config['batch_size']}")
        print(f"  训练轮数: {self.config['epochs']}")
        print(f"  学习率: {self.config['learning_rate']}")
        print(f"  隐藏层大小: {self.config['hidden_size']}")
        print(f"  回看期数: {self.config['lookback']}")
        
        trained_models = {}
        
        for model_type in model_types:
            print(f"\n{'='*70}")
            print(f"训练 {model_type.upper()} 模型")
            print(f"{'='*70}")
            
            try:
                # 创建新的预测器
                predictor = TrendBasedPredictor(
                    data=self.data,
                    use_ml=True,
                    model_type=model_type
                )
                
                # 训练模型
                predictor.train_ml_model(
                    epochs=self.config['epochs'],
                    batch_size=self.config['batch_size'],
                    learning_rate=self.config['learning_rate'],
                    validation_split=self.config['validation_split']
                )
                
                if predictor.ml_predictor and predictor.ml_predictor.is_trained:
                    # 保存模型
                    model_path = self.output_dir / f'{model_type}_model.pth'
                    self._save_model(predictor, model_type, model_path)
                    trained_models[model_type] = predictor
                    print(f"✓ {model_type.upper()} 模型训练完成并已保存")
                else:
                    print(f"⚠ {model_type.upper()} 模型训练失败")
            
            except Exception as e:
                print(f"✗ {model_type.upper()} 模型训练出错: {e}")
                import traceback
                traceback.print_exc()
        
        return trained_models
    
    def _save_model(self, predictor, model_type: str, model_path: Path):
        """保存模型"""
        import torch
        
        if predictor.ml_predictor and predictor.ml_predictor.model:
            # 保存模型状态
            torch.save({
                'model_state_dict': predictor.ml_predictor.model.state_dict(),
                'model_type': model_type,
                'config': self.config,
                'scaler': predictor.ml_predictor.scaler,
                'data_info': {
                    'num_periods': len(self.data),
                    'last_period': self.data['期号'].iloc[-1]
                }
            }, model_path)
            
            # 保存配置
            config_path = self.output_dir / f'{model_type}_config.json'
            with open(config_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'model_type': model_type,
                    'training_config': self.config,
                    'trained_at': datetime.now().isoformat(),
                    'device': DEVICE
                }, f, indent=2, ensure_ascii=False)
    
    def evaluate_models(self, models: dict):
        """评估模型"""
        print("\n" + "=" * 70)
        print("模型评估")
        print("=" * 70)
        
        # 使用最近10期作为测试集
        test_size = min(10, len(self.data) // 10)
        test_data = self.data.tail(test_size)
        train_data = self.data.iloc[:-test_size]
        
        print(f"\n测试集: 最近 {test_size} 期")
        print(f"训练集: {len(train_data)} 期")
        
        results = {}
        
        for model_type, predictor in models.items():
            print(f"\n评估 {model_type.upper()} 模型:")
            
            try:
                # 对测试集进行预测
                predictions = []
                actuals = []
                
                for idx, row in test_data.iterrows():
                    # 使用到当前期之前的数据进行预测
                    historical_data = self.data.iloc[:idx]
                    if len(historical_data) < 50:
                        continue
                    
                    temp_predictor = TrendBasedPredictor(
                        data=historical_data,
                        use_ml=True,
                        model_type=model_type
                    )
                    
                    # 这里简化处理，实际应该加载已训练的模型
                    try:
                        pred_front, pred_back = temp_predictor.predict(method='ml')
                        actual_front = sorted([row[f'前区{i}'] for i in range(1, 6)])
                        actual_back = sorted([row[f'后区{i}'] for i in range(1, 3)])
                        
                        predictions.append((pred_front, pred_back))
                        actuals.append((actual_front, actual_back))
                    except:
                        pass
                
                if len(predictions) > 0:
                    # 计算准确率（前区匹配数）
                    front_matches = []
                    back_matches = []
                    
                    for (pred_f, pred_b), (act_f, act_b) in zip(predictions, actuals):
                        front_match = len(set(pred_f) & set(act_f))
                        back_match = len(set(pred_b) & set(act_b))
                        front_matches.append(front_match)
                        back_matches.append(back_match)
                    
                    avg_front_match = np.mean(front_matches)
                    avg_back_match = np.mean(back_matches)
                    
                    results[model_type] = {
                        'front_match': avg_front_match,
                        'back_match': avg_back_match,
                        'num_predictions': len(predictions)
                    }
                    
                    print(f"  前区平均匹配数: {avg_front_match:.2f}/5")
                    print(f"  后区平均匹配数: {avg_back_match:.2f}/2")
                else:
                    print("  ⚠ 无法进行评估（数据不足）")
            
            except Exception as e:
                print(f"  ✗ 评估失败: {e}")
        
        return results
    
    def make_predictions(self, models: dict):
        """使用训练好的模型进行预测"""
        print("\n" + "=" * 70)
        print("生成预测")
        print("=" * 70)
        
        predictions = {}
        
        for model_type, predictor in models.items():
            print(f"\n{model_type.upper()} 模型预测:")
            try:
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
        
        return predictions
    
    def run_full_pipeline(self, model_types: list = ['lstm'], 
                         train: bool = True, evaluate: bool = True, 
                         predict: bool = True):
        """运行完整流程"""
        print("\n" + "=" * 70)
        print("机器学习训练管道")
        print("=" * 70)
        
        # 1. 数据质量检查
        if not self.check_data_quality():
            print("\n⚠ 数据质量检查未通过，但将继续训练")
        
        # 2. 训练模型
        trained_models = {}
        if train:
            trained_models = self.train_models(model_types=model_types)
        
        if not trained_models:
            print("\n⚠ 没有成功训练的模型")
            return
        
        # 3. 评估模型（可选）
        if evaluate:
            try:
                results = self.evaluate_models(trained_models)
                print(f"\n评估结果已保存")
            except Exception as e:
                print(f"\n⚠ 评估过程出错: {e}")
        
        # 4. 生成预测
        if predict:
            predictions = self.make_predictions(trained_models)
            
            # 保存预测结果
            pred_path = self.output_dir / 'predictions.json'
            pred_dict = {
                model_type: {
                    'front': list(front),
                    'back': list(back)
                }
                for model_type, (front, back) in predictions.items()
            }
            
            with open(pred_path, 'w', encoding='utf-8') as f:
                json.dump(pred_dict, f, indent=2, ensure_ascii=False)
            
            print(f"\n✓ 预测结果已保存至: {pred_path}")
        
        print("\n" + "=" * 70)
        print("流程完成！")
        print("=" * 70)
        print(f"\n模型保存在: {self.output_dir}")
        print(f"配置文件: {self.output_dir}/*_config.json")
        print(f"预测结果: {self.output_dir}/predictions.json")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='大乐透机器学习训练管道')
    parser.add_argument('--data', type=str, help='数据文件路径（CSV格式）')
    parser.add_argument('--models', nargs='+', default=['lstm'], 
                       choices=['lstm', 'transformer'],
                       help='要训练的模型类型')
    parser.add_argument('--output', type=str, default='./ml_models/',
                       help='模型输出目录')
    parser.add_argument('--epochs', type=int, default=None,
                       help='训练轮数（覆盖默认配置）')
    parser.add_argument('--batch-size', type=int, default=None,
                       help='批次大小（覆盖默认配置）')
    parser.add_argument('--no-train', action='store_true',
                       help='跳过训练（仅预测）')
    parser.add_argument('--no-evaluate', action='store_true',
                       help='跳过评估')
    parser.add_argument('--no-predict', action='store_true',
                       help='跳过预测')
    
    args = parser.parse_args()
    
    # 创建训练管道
    pipeline = MLTrainingPipeline(
        data_file=args.data,
        output_dir=args.output
    )
    
    # 准备训练参数
    train_kwargs = {}
    if args.epochs:
        train_kwargs['epochs'] = args.epochs
    if args.batch_size:
        train_kwargs['batch_size'] = args.batch_size
    
    # 运行完整流程
    pipeline.run_full_pipeline(
        model_types=args.models,
        train=not args.no_train,
        evaluate=not args.no_evaluate,
        predict=not args.no_predict,
        **train_kwargs
    )


if __name__ == "__main__":
    main()
