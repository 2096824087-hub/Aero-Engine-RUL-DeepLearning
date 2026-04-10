import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ==========================================
# Environment Setup & Configuration
# ==========================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_PATH = "./archive/train_FD001.txt"
TEST_PATH = "./archive/test_FD001.txt"
TRUTH_PATH = "./archive/RUL_FD001.txt"

MAX_RUL = 125
SEQ_LEN = 40
BATCH_SIZE = 64
EPOCHS = 40
LR = 0.001
WEIGHT_DECAY = 1e-4

INDEX_COLS = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
SELECTED_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 
                    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 
                    'sensor_17', 'sensor_20', 'sensor_21']
ALL_COLS = INDEX_COLS + [f'sensor_{i}' for i in range(1, 22)]

# ==========================================
# Data Processing Pipeline 
# (Reused logic, encapsulated for clarity)
# ==========================================
def preprocess_data(file_path, is_train=True, scaler=None):
    df = pd.read_csv(file_path, sep='\s+', header=None, names=ALL_COLS)
    
    if is_train:
        max_cycle = df.groupby('unit_number')['time_cycles'].max().reset_index(name='max_cycle')
        df = df.merge(max_cycle, on='unit_number', how='left')
        df['RUL'] = (df['max_cycle'] - df['time_cycles']).clip(upper=MAX_RUL)
        
        scaler = MinMaxScaler()
        df[SELECTED_SENSORS] = scaler.fit_transform(df[SELECTED_SENSORS])
    else:
        df[SELECTED_SENSORS] = scaler.transform(df[SELECTED_SENSORS])
        
    for col in SELECTED_SENSORS:
        df[col + '_smooth'] = df.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
    return df, scaler, [f + '_smooth' for f in SELECTED_SENSORS]

train_df, scaler, feature_cols = preprocess_data(TRAIN_PATH, is_train=True)

class CMAPSSDataset(Dataset):
    def __init__(self, df, seq_len, feature_cols):
        self.data, self.labels = [], []
        for unit_id in df['unit_number'].unique():
            unit_df = df[df['unit_number'] == unit_id]
            features, target = unit_df[feature_cols].values, unit_df['RUL'].values
            for i in range(len(unit_df) - seq_len + 1):
                self.data.append(features[i : i + seq_len])
                self.labels.append(target[i + seq_len - 1])
                
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

train_loader = DataLoader(CMAPSSDataset(train_df, SEQ_LEN, feature_cols), batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# Model Architecture: CNN-LSTM
# ==========================================
class MultiScaleCNN(nn.Module):
    def __init__(self, num_features):
        super().__init__()
        self.conv1 = nn.Conv1d(num_features, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(num_features, 16, kernel_size=5, padding=2)
        self.conv3 = nn.Conv1d(num_features, 16, kernel_size=7, padding=3)
        self.bn1, self.bn2, self.bn3 = nn.BatchNorm1d(16), nn.BatchNorm1d(16), nn.BatchNorm1d(16)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.permute(0, 2, 1) # Shape: [Batch, Features, Seq_len]
        out1 = self.relu(self.bn1(self.conv1(x)))
        out2 = self.relu(self.bn2(self.conv2(x)))
        out3 = self.relu(self.bn3(self.conv3(x)))
        out = torch.cat((out1, out2, out3), dim=1) # Shape: [Batch, 48, Seq_len]
        return out.permute(0, 2, 1)

class CNN_LSTM_Combine(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.cnn = MultiScaleCNN(input_dim)
        self.lstm = nn.LSTM(input_size=48, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = self.cnn(x)
        out, _ = self.lstm(x)
        return self.fc(self.dropout(out[:, -1, :]))

model = CNN_LSTM_Combine(input_dim=len(feature_cols), hidden_dim=32).to(device)

# ==========================================
# Training Process
# ==========================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.5)

print(f">>> Starting CNN-LSTM Training on {device}...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(seqs), labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    scheduler.step()
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | MSE Loss: {epoch_loss/len(train_loader):.4f}")

# ==========================================
# Evaluation & Visualization
# ==========================================
test_df, _, _ = preprocess_data(TEST_PATH, is_train=False, scaler=scaler)
y_truth = pd.read_csv(TRUTH_PATH, sep='\s+', header=None, names=['RUL'])

test_sequences = []
for unit_id in test_df['unit_number'].unique():
    unit_data = test_df[test_df['unit_number'] == unit_id]
    features = unit_data[feature_cols].values[-SEQ_LEN:]
    if len(features) < SEQ_LEN:
        padding = np.repeat(features[0:1, :], SEQ_LEN - len(features), axis=0)
        features = np.vstack((padding, features))
    test_sequences.append(features)

X_test_tensor = torch.tensor(np.array(test_sequences), dtype=torch.float32).to(device)

model.eval()
with torch.no_grad():
    predictions = model(X_test_tensor).cpu().numpy().flatten()
    predictions = np.clip(predictions, 0, MAX_RUL)

true_rul = y_truth['RUL'].values.flatten()
rmse = np.sqrt(mean_squared_error(true_rul, predictions))
print(f"Test Set RMSE: {rmse:.2f}")

sort_idx = np.argsort(true_rul)
plt.figure(figsize=(10, 5))
plt.plot(true_rul[sort_idx], label='Ground Truth', color='royalblue', linewidth=2)
plt.plot(predictions[sort_idx], label='CNN-LSTM Prediction', color='firebrick', linestyle='--', alpha=0.8)
plt.title(f'Feature Fusion Evaluation (RMSE: {rmse:.2f})')
plt.xlabel('Engine Index (Sorted by Truth)')
plt.ylabel('Remaining Useful Life (Cycles)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()