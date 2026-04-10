import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import os

# ==========================================
# 1. 全局配置与超参数 (Global Config & Hyperparams)
# ==========================================
torch.manual_seed(42)
np.random.seed(42)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MAX_RUL = 125       # 早期退化平滑阈值 (Piecewise Linear RUL)
SEQ_LEN = 30        # 滑动窗口时间步长
BATCH_SIZE = 64     
LR = 0.001          # 初始学习率
EPOCHS = 100        # 最大训练轮数
PATIENCE = 15       # 早停容忍度：验证集 Loss 连续 15 轮不降则触发早停

# ==========================================
# 2. 数据加载与特征工程 (Data Pipeline)
# ==========================================
def load_and_preprocess(train_path, test_path, truth_path, val_ratio=0.1):
    """加载 C-MAPSS 数据，执行归一化、平滑与训练/验证集划分"""
    index_cols = ['unit_number', 'time_cycles']
    # 筛选高相关性传感器特征 (基于领域先验知识)
    selected_sensors = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 
                        'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 
                        'sensor_17', 'sensor_20', 'sensor_21']
    all_cols = index_cols + ['s1', 's2', 's3'] + [f'sensor_{i}' for i in range(1, 22)]

    # 读取原始数据
    train_df = pd.read_csv(train_path, sep='\s+', header=None, names=all_cols)
    test_df = pd.read_csv(test_path, sep='\s+', header=None, names=all_cols)
    y_truth = pd.read_csv(truth_path, sep='\s+', header=None, names=['RUL'])

    # 构建分段线性 RUL 标签 (Piecewise Linear Degradation Model)
    max_cycle = train_df.groupby('unit_number')['time_cycles'].max().reset_index(name='max')
    train_df = train_df.merge(max_cycle, on='unit_number')
    train_df['RUL'] = (train_df['max'] - train_df['time_cycles']).clip(upper=MAX_RUL)

    # 划分训练集与验证集 (按引擎 ID 划分，严格防止时间序列数据泄露)
    unique_units = train_df['unit_number'].unique()
    val_units = np.random.choice(unique_units, size=int(len(unique_units) * val_ratio), replace=False)
    val_df = train_df[train_df['unit_number'].isin(val_units)].copy()
    train_df = train_df[~train_df['unit_number'].isin(val_units)].copy()

    # Min-Max 归一化 (仅使用训练集拟合，防止未来数据泄露)
    scaler = MinMaxScaler()
    train_df[selected_sensors] = scaler.fit_transform(train_df[selected_sensors])
    val_df[selected_sensors] = scaler.transform(val_df[selected_sensors])
    test_df[selected_sensors] = scaler.transform(test_df[selected_sensors])

    # 滑动平均滤波，消除传感器高频噪声
    def apply_smoothing(df):
        for col in selected_sensors:
            df[col] = df.groupby('unit_number')[col].transform(lambda x: x.rolling(10, 1).mean())
        return df

    return apply_smoothing(train_df), apply_smoothing(val_df), apply_smoothing(test_df), y_truth, selected_sensors

def create_sequences(df, feature_cols, is_test=False):
    """基于时间窗口切分序列数据"""
    data, labels = [], []
    for uid in df['unit_number'].unique():
        unit_data = df[df['unit_number'] == uid]
        feat = unit_data[feature_cols].values
        if not is_test:
            target = unit_data['RUL'].values
            for i in range(len(unit_data) - SEQ_LEN + 1):
                data.append(feat[i:i+SEQ_LEN])
                labels.append(target[i+SEQ_LEN-1])
        else:
            # 测试集仅取每个引擎最后一条序列进行寿命预测
            last_feat = feat[-SEQ_LEN:]
            if len(last_feat) < SEQ_LEN:
                last_feat = np.pad(last_feat, ((SEQ_LEN-len(last_feat), 0), (0, 0)), 'edge')
            data.append(last_feat)
    return torch.tensor(np.array(data), dtype=torch.float32), torch.tensor(np.array(labels), dtype=torch.float32)

# ==========================================
# 3. 模型架构 (Model Architecture)
# ==========================================
class SelfAttention(nn.Module):
    """时序自注意力层，聚焦临近失效的关键退化特征"""
    def __init__(self, hidden_dim):
        super().__init__()
        self.w = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, x):
        weights = torch.softmax(torch.tanh(self.w(x)), dim=1)
        return torch.sum(weights * x, dim=1)

class MSCNN_BiLSTM_Att(nn.Module):
    """多尺度 CNN + BiLSTM + 注意力机制 融合网络"""
    def __init__(self, input_dim):
        super().__init__()
        # 多尺度空间特征提取
        self.cnn3 = nn.Conv1d(input_dim, 32, kernel_size=3, padding=1)
        self.cnn5 = nn.Conv1d(input_dim, 32, kernel_size=5, padding=2)
        # 双向时序特征提取 (强化长程依赖)
        self.lstm = nn.LSTM(64, 64, num_layers=2, batch_first=True, bidirectional=True, dropout=0.3)
        self.att = SelfAttention(128)
        # 回归预测头
        self.fc = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3), nn.Linear(32, 1))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        """Kaiming 初始化与正交初始化，提升深层网络收敛稳定性"""
        if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if m.bias is not None: nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LSTM):
            for name, param in m.named_parameters():
                if 'weight_ih' in name: nn.init.xavier_uniform_(param.data)
                elif 'weight_hh' in name: nn.init.orthogonal_(param.data)
                elif 'bias' in name: nn.init.constant_(param.data, 0)

    def forward(self, x):
        x = x.permute(0, 2, 1) # 转换维度适应 Conv1d: [batch, features, seq]
        x = torch.cat([torch.relu(self.cnn3(x)), torch.relu(self.cnn5(x))], dim=1).permute(0, 2, 1)
        out, _ = self.lstm(x)
        return self.fc(self.att(out))

# ==========================================
# 4. 早停监控器 (Early Stopping Monitor)
# ==========================================
class EarlyStopping:
    """监控验证集 Loss，避免过拟合，自动保存最优权重"""
    def __init__(self, patience=10, path='best_model.pth'):
        self.patience = patience
        self.path = path
        self.counter = 0
        self.best_loss = float('inf')
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            torch.save(model.state_dict(), self.path) # 保存最优参数
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

# ==========================================
# 5. 执行流程 (Main Pipeline)
# ==========================================
if __name__ == "__main__":
    print(f">>> 计算平台初始化完成: {DEVICE}")
    
    # 加载与切分数据
    train_df, val_df, test_df, y_truth, feature_cols = load_and_preprocess(
        "./archive/train_FD001.txt", "./archive/test_FD001.txt", "./archive/RUL_FD001.txt"
    )
    
    x_train, y_train = create_sequences(train_df, feature_cols)
    x_val, y_val = create_sequences(val_df, feature_cols)
    
    train_loader = DataLoader(list(zip(x_train, y_train)), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(list(zip(x_val, y_val)), batch_size=BATCH_SIZE, shuffle=False)

    # 初始化模型与优化组件
    model = MSCNN_BiLSTM_Att(len(feature_cols)).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.2)
    criterion = nn.MSELoss()
    early_stopper = EarlyStopping(patience=PATIENCE, path='rul_best_model.pth')

    # 训练循环
    print(">>> 启动模型训练...")
    for epoch in range(EPOCHS):
        # --- 训练阶段 ---
        model.train()
        train_loss = 0
        for seqs, labs in train_loader:
            seqs, labs = seqs.to(DEVICE), labs.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(seqs), labs.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0) # 梯度裁剪
            optimizer.step()
            train_loss += loss.item()
        
        # --- 验证阶段 ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for seqs, labs in val_loader:
                seqs, labs = seqs.to(DEVICE), labs.to(DEVICE)
                val_loss += criterion(model(seqs), labs.unsqueeze(1)).item()
                
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        scheduler.step()
        
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:03d} | Train Loss: {train_loss:.2f} | Val Loss: {val_loss:.2f}")

        # 早停检测
        early_stopper(val_loss, model)
        if early_stopper.early_stop:
            print(f">>> 触发早停机制 (Epoch {epoch+1})！最优验证集 Loss: {early_stopper.best_loss:.2f}")
            break

    # ==========================================
    # 6. 模型评估与可视化 (Evaluation)
    # ==========================================
    print(">>> 正在加载最优模型权重进行测试集评估...")
    model.load_state_dict(torch.load('rul_best_model.pth', weights_only=True)) # 加载最优权重
    model.eval()
    
    x_test, _ = create_sequences(test_df, feature_cols, is_test=True)
    with torch.no_grad():
        preds = model(x_test.to(DEVICE)).cpu().numpy().flatten().clip(0, MAX_RUL)

    rmse = np.sqrt(mean_squared_error(y_truth['RUL'], preds))
    print(f"\n🔥 最终测试集 RMSE: {rmse:.2f}")

    # 绘制真实值与预测值对比图
    plt.figure(figsize=(12, 5))
    plt.plot(y_truth['RUL'].values, label='Ground Truth', color='black', linewidth=1.5)
    plt.plot(preds, label='Model Prediction', color='red', linestyle='--', alpha=0.8)
    plt.title(f"Aero-Engine RUL Prediction on C-MAPSS FD001 (RMSE: {rmse:.2f})")
    plt.xlabel("Engine Unit Sequence")
    plt.ylabel("Remaining Useful Life (Cycles)")
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.show()