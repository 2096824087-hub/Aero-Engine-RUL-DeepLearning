import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

# ==========================================
# Configuration & Hyperparameters
# ==========================================
TRAIN_PATH = "./archive/train_FD001.txt"
TEST_PATH = "./archive/test_FD001.txt"
TRUTH_PATH = "./archive/RUL_FD001.txt"

MAX_RUL = 125          # Piecewise Linear degradation threshold
SEQ_LEN = 30           # Sliding window size
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.003
HIDDEN_DIM = 64
NUM_LAYERS = 2

# Feature Selection
INDEX_COLS = ['unit_number', 'time_cycles', 'setting_1', 'setting_2', 'setting_3']
SENSOR_COLS = [f'sensor_{i}' for i in range(1, 22)]
ALL_COLS = INDEX_COLS + SENSOR_COLS

# Selected sensors based on variance analysis
SELECTED_SENSORS = ['sensor_2', 'sensor_3', 'sensor_4', 'sensor_7', 'sensor_8', 
                    'sensor_11', 'sensor_12', 'sensor_13', 'sensor_14', 'sensor_15', 
                    'sensor_17', 'sensor_20', 'sensor_21']

# ==========================================
# Data Preprocessing Pipeline
# ==========================================
def preprocess_data(file_path, is_train=True, scaler=None):
    """ Load, clean, scale, and smooth C-MAPSS data """
    df = pd.read_csv(file_path, sep='\s+', header=None, names=ALL_COLS)
    
    # RUL Calculation (Only for training data in this step)
    if is_train:
        max_cycle = df.groupby('unit_number')['time_cycles'].max().reset_index()
        max_cycle.columns = ['unit_number', 'max_cycle']
        df = df.merge(max_cycle, on='unit_number', how='left')
        df['RUL'] = df['max_cycle'] - df['time_cycles']
        df['RUL'] = df['RUL'].clip(upper=MAX_RUL)
    
    # Feature Scaling
    if is_train:
        scaler = MinMaxScaler()
        df[SELECTED_SENSORS] = scaler.fit_transform(df[SELECTED_SENSORS])
    else:
        df[SELECTED_SENSORS] = scaler.transform(df[SELECTED_SENSORS])
        
    # Moving Average Smoothing
    for col in SELECTED_SENSORS:
        df[col + '_smooth'] = df.groupby('unit_number')[col].transform(
            lambda x: x.rolling(window=10, min_periods=1).mean()
        )
        
    smooth_features = [f + '_smooth' for f in SELECTED_SENSORS]
    return df, scaler, smooth_features

print(">>> Preprocessing Data...")
train_df, scaler, feature_cols = preprocess_data(TRAIN_PATH, is_train=True)

# ==========================================
# Dataset Construction
# ==========================================
class CMAPSSDataset(Dataset):
    """ Time-series dataset generation via sliding window """
    def __init__(self, df, seq_len, feature_cols):
        self.data, self.labels = [], []
        
        for unit_id in df['unit_number'].unique():
            unit_df = df[df['unit_number'] == unit_id]
            features = unit_df[feature_cols].values
            target = unit_df['RUL'].values
            
            for i in range(len(unit_df) - seq_len + 1):
                self.data.append(features[i : i + seq_len])
                self.labels.append(target[i + seq_len - 1])
                
        self.data = torch.tensor(np.array(self.data), dtype=torch.float32)
        self.labels = torch.tensor(np.array(self.labels), dtype=torch.float32)

    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx], self.labels[idx]

train_dataset = CMAPSSDataset(train_df, SEQ_LEN, feature_cols)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

# ==========================================
# Model Architecture
# ==========================================
class RULPredictorLSTM(nn.Module):
    """ Standard Multi-layer LSTM for RUL Prediction """
    def __init__(self, input_dim, hidden_dim, num_layers):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = RULPredictorLSTM(input_dim=len(feature_cols), hidden_dim=HIDDEN_DIM, num_layers=NUM_LAYERS).to(device)

# ==========================================
# Training Loop
# ==========================================
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)

print(f">>> Starting Training on {device}...")
for epoch in range(EPOCHS):
    model.train()
    epoch_loss = 0
    for seqs, labels in train_loader:
        seqs, labels = seqs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(seqs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    if (epoch + 1) % 5 == 0:
        print(f"Epoch [{epoch+1}/{EPOCHS}] | MSE Loss: {epoch_loss/len(train_loader):.4f}")

# ==========================================
# Evaluation & Visualization
# ==========================================
print(">>> Evaluating on Test Set...")
test_df, _, _ = preprocess_data(TEST_PATH, is_train=False, scaler=scaler)
y_truth = pd.read_csv(TRUTH_PATH, sep='\s+', header=None, names=['RUL'])

test_sequences = []
for unit_id in test_df['unit_number'].unique():
    unit_data = test_df[test_df['unit_number'] == unit_id]
    features = unit_data[feature_cols].values[-SEQ_LEN:]
    
    # Edge Padding for short sequences
    if len(features) < SEQ_LEN:
        pad_size = SEQ_LEN - len(features)
        padding = np.repeat(features[0:1, :], pad_size, axis=0)
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

# Plotting Results
sort_idx = np.argsort(true_rul)
plt.figure(figsize=(10, 5))
plt.plot(true_rul[sort_idx], label='Ground Truth', color='royalblue', linewidth=2)
plt.plot(predictions[sort_idx], label='LSTM Prediction', color='darkorange', linestyle='--', alpha=0.8)
plt.title(f'RUL Prediction Evaluation (RMSE: {rmse:.2f})')
plt.xlabel('Engine Index (Sorted by Truth)')
plt.ylabel('Remaining Useful Life (Cycles)')
plt.legend()
plt.grid(alpha=0.3)
plt.show()