"""
基于深度学习的走势图识别和预测模型
使用CNN、LSTM、Transformer等模型来"看懂"走势图
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("警告: PyTorch 未安装，深度学习功能不可用")

try:
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.model_selection import train_test_split
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    print("警告: scikit-learn 未安装")


class TrendDataset(Dataset):
    """走势数据数据集"""
    
    def __init__(self, features: np.ndarray, targets: np.ndarray = None):
        """
        Args:
            features: 特征矩阵 (samples, time_steps, features) 或 (samples, features)
            targets: 目标值 (samples, output_dim) 或 None（用于预测）
        """
        self.features = torch.FloatTensor(features)
        self.has_targets = targets is not None
        if self.has_targets:
            self.targets = torch.FloatTensor(targets)
        else:
            self.targets = None
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        if self.has_targets:
            return self.features[idx], self.targets[idx]
        else:
            return self.features[idx]


class PositionLSTM(nn.Module):
    """
    LSTM模型：学习每个位置的走势规律
    横向分析：理解每个位置号码的变化趋势
    """
    
    def __init__(self, input_size: int = 19, hidden_size: int = 64, 
                 num_layers: int = 2, output_size: int = 7, dropout: float = 0.2):
        """
        Args:
            input_size: 输入特征维度（位置值+组合特征）
            hidden_size: LSTM隐藏层大小
            num_layers: LSTM层数
            output_size: 输出维度（5个前区+2个后区）
            dropout: Dropout比率
        """
        super(PositionLSTM, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=True
        )
        
        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        lstm_out, _ = self.lstm(x)
        # 取最后一个时间步的输出
        last_output = lstm_out[:, -1, :]
        
        # 全连接层
        out = self.fc1(last_output)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out


class TrendCNN(nn.Module):
    """
    CNN模型：从走势图图像中提取特征
    用于识别走势图的视觉模式
    """
    
    def __init__(self, num_classes: int = 7):
        """
        Args:
            num_classes: 输出类别数（5个前区+2个后区）
        """
        super(TrendCNN, self).__init__()
        
        # 卷积层
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2, 2)
        
        # 全连接层
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        self.dropout1 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        # x shape: (batch, 1, height, width)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.pool3(x)
        
        # 展平
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.dropout1(x)
        x = self.relu4(x)
        x = self.fc2(x)
        
        return x


class TrendTransformer(nn.Module):
    """
    Transformer模型：学习位置之间的关系和组合规律
    纵向分析：理解号码组合的整体走势
    """
    
    def __init__(self, input_size: int = 19, d_model: int = 128, 
                 nhead: int = 8, num_layers: int = 4, 
                 output_size: int = 7, dropout: float = 0.1):
        """
        Args:
            input_size: 输入特征维度
            d_model: Transformer模型维度
            nhead: 注意力头数
            num_layers: Transformer层数
            output_size: 输出维度
            dropout: Dropout比率
        """
        super(TrendTransformer, self).__init__()
        
        # 输入投影
        self.input_projection = nn.Linear(input_size, d_model)
        
        # 位置编码
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        # Transformer编码器
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        # 输出层
        self.fc = nn.Linear(d_model, output_size)
        
    def forward(self, x):
        # x shape: (batch, seq_len, features)
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)
        # 取最后一个时间步或平均池化
        x = x.mean(dim=1)  # (batch, d_model)
        x = self.fc(x)
        return x


class PositionalEncoding(nn.Module):
    """位置编码"""
    
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 100):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TrendMatcher:
    """
    走势匹配器：基于历史走势模式匹配最相似的组合
    """
    
    def __init__(self, trend_analyzer, similarity_threshold: float = 0.8):
        """
        Args:
            trend_analyzer: TrendAnalyzer实例
            similarity_threshold: 相似度阈值
        """
        self.trend_analyzer = trend_analyzer
        self.similarity_threshold = similarity_threshold
        self.historical_patterns = []
        self._build_pattern_library()
    
    def _build_pattern_library(self):
        """构建历史模式库"""
        position_trends = self.trend_analyzer.get_position_trends()
        combination_trends = self.trend_analyzer.get_combination_trends()
        
        # 提取每个历史期的模式特征
        for i in range(len(self.trend_analyzer.data)):
            pattern = {
                'period_idx': i,
                'front_positions': [position_trends['front_positions'][j][i] 
                                   for j in range(1, 6)],
                'back_positions': [position_trends['back_positions'][j][i] 
                                  for j in range(1, 3)],
                'sum': combination_trends['sum_trend'][i],
                'span': combination_trends['span_trend'][i],
                'odd_even': combination_trends['odd_even_trend'][i],
                'size_ratio': combination_trends['size_ratio_trend'][i],
            }
            self.historical_patterns.append(pattern)
    
    def calculate_similarity(self, pattern1: Dict, pattern2: Dict) -> float:
        """计算两个模式的相似度"""
        # 位置相似度（加权）
        front_sim = 0
        for i in range(5):
            diff = abs(pattern1['front_positions'][i] - pattern2['front_positions'][i])
            # 归一化到0-1
            front_sim += 1 - (diff / 35.0)
        front_sim /= 5
        
        back_sim = 0
        for i in range(2):
            diff = abs(pattern1['back_positions'][i] - pattern2['back_positions'][i])
            back_sim += 1 - (diff / 12.0)
        back_sim /= 2
        
        # 组合特征相似度
        sum_diff = abs(pattern1['sum'] - pattern2['sum']) / 200.0  # 归一化
        span_diff = abs(pattern1['span'] - pattern2['span']) / 35.0
        oe_diff = abs(pattern1['odd_even'] - pattern2['odd_even']) / 5.0
        sr_diff = abs(pattern1['size_ratio'] - pattern2['size_ratio']) / 5.0
        
        combo_sim = 1 - (sum_diff + span_diff + oe_diff + sr_diff) / 4
        
        # 综合相似度（位置权重0.6，组合权重0.4）
        total_sim = 0.6 * (front_sim * 0.7 + back_sim * 0.3) + 0.4 * combo_sim
        
        return total_sim
    
    def find_similar_patterns(self, current_pattern: Dict, top_k: int = 10) -> List[Tuple[Dict, float]]:
        """找到最相似的历史模式"""
        similarities = []
        
        for hist_pattern in self.historical_patterns:
            sim = self.calculate_similarity(current_pattern, hist_pattern)
            similarities.append((hist_pattern, sim))
        
        # 按相似度排序
        similarities.sort(key=lambda x: x[1], reverse=True)
        
        return similarities[:top_k]
    
    def predict_by_matching(self, lookback: int = 10) -> Tuple[List[int], List[int]]:
        """
        基于模式匹配进行预测
        
        Args:
            lookback: 使用最近N期的平均模式作为当前模式
        
        Returns:
            (前区号码, 后区号码)
        """
        # 获取最近N期的平均模式
        recent_patterns = self.historical_patterns[-lookback:]
        
        avg_pattern = {
            'front_positions': [
                np.mean([p['front_positions'][i] for p in recent_patterns])
                for i in range(5)
            ],
            'back_positions': [
                np.mean([p['back_positions'][i] for p in recent_patterns])
                for i in range(2)
            ],
            'sum': np.mean([p['sum'] for p in recent_patterns]),
            'span': np.mean([p['span'] for p in recent_patterns]),
            'odd_even': np.mean([p['odd_even'] for p in recent_patterns]),
            'size_ratio': np.mean([p['size_ratio'] for p in recent_patterns]),
        }
        
        # 找到最相似的历史模式
        similar_patterns = self.find_similar_patterns(avg_pattern, top_k=5)
        
        # 使用最相似模式的下一个模式作为预测
        if len(similar_patterns) > 0:
            best_pattern = similar_patterns[0][0]
            best_idx = best_pattern['period_idx']
            
            # 如果最佳模式不是最后一期，使用它的下一期
            if best_idx < len(self.historical_patterns) - 1:
                next_pattern = self.historical_patterns[best_idx + 1]
                front = [int(round(n)) for n in next_pattern['front_positions']]
                back = [int(round(n)) for n in next_pattern['back_positions']]
            else:
                # 使用最佳模式本身
                front = [int(round(n)) for n in best_pattern['front_positions']]
                back = [int(round(n)) for n in best_pattern['back_positions']]
        else:
            # 回退：使用最近期的模式
            recent = self.historical_patterns[-1]
            front = [int(round(n)) for n in recent['front_positions']]
            back = [int(round(n)) for n in recent['back_positions']]
        
        # 确保号码在有效范围内且不重复
        front = sorted(list(set([max(1, min(35, n)) for n in front])))
        back = sorted(list(set([max(1, min(12, n)) for n in back])))
        
        # 如果数量不足，补充随机号码
        while len(front) < 5:
            num = np.random.randint(1, 36)
            if num not in front:
                front.append(num)
        front = sorted(front[:5])
        
        while len(back) < 2:
            num = np.random.randint(1, 13)
            if num not in back:
                back.append(num)
        back = sorted(back[:2])
        
        return front, back


class TrendMLPredictor:
    """
    基于机器学习的走势预测器
    整合多种模型进行预测
    """
    
    def __init__(self, trend_analyzer, model_type: str = 'lstm'):
        """
        Args:
            trend_analyzer: TrendAnalyzer实例
            model_type: 模型类型 ('lstm', 'transformer', 'cnn', 'ensemble')
        """
        self.trend_analyzer = trend_analyzer
        self.model_type = model_type
        self.model = None
        self.scaler = StandardScaler() if HAS_SKLEARN else None
        self.is_trained = False
    
    def prepare_training_data(self, lookback: int = 30, forecast: int = 1):
        """
        准备训练数据
        
        Args:
            lookback: 回看期数
            forecast: 预测期数
        
        Returns:
            (X, y): 特征和目标
        """
        # 提取特征
        all_features = []
        all_targets = []
        
        data_len = len(self.trend_analyzer.data)
        
        for i in range(lookback, data_len - forecast + 1):
            # 获取lookback期的特征
            period_features = []
            for j in range(i - lookback, i):
                features = self.trend_analyzer.extract_trend_features_for_ml(lookback=1)
                if len(features) > 0:
                    period_features.append(features[0])
            
            if len(period_features) == lookback:
                # 获取目标（下一期的号码）
                target_row = self.trend_analyzer.data.iloc[i]
                front_target = sorted([target_row[f'前区{j}'] for j in range(1, 6)])
                back_target = sorted([target_row[f'后区{j}'] for j in range(1, 3)])
                target = front_target + back_target
                
                all_features.append(period_features)
                all_targets.append(target)
        
        X = np.array(all_features)
        y = np.array(all_targets)
        
        return X, y
    
    def train(self, epochs: int = 50, batch_size: int = 32, 
              learning_rate: float = 0.001, validation_split: float = 0.2):
        """训练模型"""
        if not HAS_TORCH:
            print("PyTorch 未安装，无法训练模型")
            return
        
        print("准备训练数据...")
        X, y = self.prepare_training_data()
        
        if len(X) == 0:
            print("数据不足，无法训练")
            return
        
        print(f"训练数据形状: X={X.shape}, y={y.shape}")
        
        # 数据标准化
        if self.scaler:
            X_reshaped = X.reshape(-1, X.shape[-1])
            X_scaled = self.scaler.fit_transform(X_reshaped)
            X = X_scaled.reshape(X.shape)
        
        # 划分训练集和验证集
        if HAS_SKLEARN:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=validation_split, random_state=42
            )
        else:
            split_idx = int(len(X) * (1 - validation_split))
            X_train, X_val = X[:split_idx], X[split_idx:]
            y_train, y_val = y[:split_idx], y[split_idx:]
        
        # 创建模型
        input_size = X.shape[2]
        output_size = y.shape[1]
        
        if self.model_type == 'lstm':
            self.model = PositionLSTM(input_size=input_size, output_size=output_size)
        elif self.model_type == 'transformer':
            self.model = TrendTransformer(input_size=input_size, output_size=output_size)
        else:
            self.model = PositionLSTM(input_size=input_size, output_size=output_size)
        
        # 训练设置
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # 数据加载器
        train_dataset = TrendDataset(X_train, y_train)
        val_dataset = TrendDataset(X_val, y_val)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
        # 训练循环
        best_val_loss = float('inf')
        patience = 10
        patience_counter = 0
        
        print(f"\n开始训练 {self.model_type.upper()} 模型...")
        for epoch in range(epochs):
            # 训练阶段
            self.model.train()
            train_loss = 0
            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            # 验证阶段
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            # 早停
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"早停于 Epoch {epoch+1}")
                    break
        
        self.is_trained = True
        print("训练完成！")
    
    def predict(self, lookback: int = 30) -> Tuple[List[int], List[int]]:
        """使用训练好的模型进行预测"""
        if not self.is_trained:
            print("模型未训练，使用模式匹配方法")
            matcher = TrendMatcher(self.trend_analyzer)
            return matcher.predict_by_matching()
        
        if not HAS_TORCH or self.model is None:
            matcher = TrendMatcher(self.trend_analyzer)
            return matcher.predict_by_matching()
        
        # 获取最近lookback期的特征
        features = self.trend_analyzer.extract_trend_features_for_ml(lookback=lookback)
        
        if len(features) < lookback:
            # 数据不足，使用模式匹配
            matcher = TrendMatcher(self.trend_analyzer)
            return matcher.predict_by_matching()
        
        # 标准化
        if self.scaler:
            features_scaled = self.scaler.transform(features)
        else:
            features_scaled = features
        
        # 转换为模型输入格式
        X = features_scaled.reshape(1, lookback, -1)
        X_tensor = torch.FloatTensor(X)
        
        # 预测
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
            predictions = outputs.numpy()[0]
        
        # 转换为号码
        front_pred = [int(round(n)) for n in predictions[:5]]
        back_pred = [int(round(n)) for n in predictions[5:]]
        
        # 确保号码在有效范围内且不重复
        front = sorted(list(set([max(1, min(35, n)) for n in front_pred])))
        back = sorted(list(set([max(1, min(12, n)) for n in back_pred])))
        
        # 补充不足的号码
        while len(front) < 5:
            num = np.random.randint(1, 36)
            if num not in front:
                front.append(num)
        front = sorted(front[:5])
        
        while len(back) < 2:
            num = np.random.randint(1, 13)
            if num not in back:
                back.append(num)
        back = sorted(back[:2])
        
        return front, back
