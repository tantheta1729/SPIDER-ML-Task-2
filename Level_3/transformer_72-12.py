import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ==========================================
# 1. SETUP & REPRODUCIBILITY
# ==========================================
SEED = 1305
random.seed(SEED) 
torch.manual_seed(SEED)
np.random.seed(SEED)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('On device:', device)

# ==========================================
# 2. DATA LOADING & CLEANING
# ==========================================
LOCAL_CSV_PATH = "jena_climate_2009_2016.csv" 

print(f"Loading Jena Climate dataset from local file: {LOCAL_CSV_PATH}...")

try:
    data = pd.read_csv(LOCAL_CSV_PATH)
except FileNotFoundError:
    print(f"\nERROR: Could not find the file at '{LOCAL_CSV_PATH}'.")
    exit()

print("Data loaded successfully! Preprocessing and cleaning...")

data['Date Time'] = pd.to_datetime(data['Date Time'], format='%d.%m.%Y %H:%M:%S')
data = data.set_index('Date Time').sort_index()

# Fix negative wind velocities
data.loc[data['wv (m/s)'] < -1, 'wv (m/s)'] = 0.0
data.loc[data['max. wv (m/s)'] < -1, 'max. wv (m/s)'] = 0.0

# Convert to vector wind components
wv = data.pop("wv (m/s)")
max_wv = data.pop("max. wv (m/s)")
wd_rad = data.pop("wd (deg)") * np.pi / 180.0

data['wx'] = wv * np.cos(wd_rad)
data['wy'] = wv * np.sin(wd_rad)
data['max wx'] = max_wv * np.cos(wd_rad)
data['max wy'] = max_wv * np.sin(wd_rad)

# Resample and interpolate
data = data.resample('1h').mean()
# Backward fill is used ONLY as a fallback for the very first few rows in the training set.
data = data.interpolate(limit_direction='forward')
data = data.bfill() 

# ==========================================
# 3. FEATURE ENGINEERING (TIME)
# ==========================================
seconds = data.index.map(pd.Timestamp.timestamp)
seconds_in_day = 24 * 60 * 60
seconds_in_year = 365 * seconds_in_day

data['sin_of_day'] = np.sin(seconds * 2 * np.pi / seconds_in_day)
data['cos_of_day'] = np.cos(seconds * 2 * np.pi / seconds_in_day)
data['sin_of_year'] = np.sin(seconds * 2 * np.pi / seconds_in_year)
data['cos_of_year'] = np.cos(seconds * 2 * np.pi / seconds_in_year)

target_name = 'T (degC)'
columns = list(data.columns)

# ==========================================
# 4. DATA SPLITTING & NORMALIZATION
# ==========================================
inp = 72
cast = 12

n = len(data)
train_end = int(n * 0.7)
val_end = int(n * 0.85)

# Calculate statistics strictly on training data
train_data_raw = data.iloc[:train_end]
mean = train_data_raw[columns].mean()
std = train_data_raw[columns].std()

target_index = columns.index(target_name)
target_mean = mean.iloc[target_index]
target_std = std.iloc[target_index]

# Normalize entire dataset using training stats
data_norm = (data[columns] - mean) / std

# Overlap boundaries by (72) so val and test sets don't lose their first forecasting windows
train_data = data_norm.iloc[:train_end]
val_data = data_norm.iloc[train_end - inp : val_end]
test_data = data_norm.iloc[val_end - inp :]

# ==========================================
# 5. SEQUENCE GENERATION & DATALOADERS
# ==========================================
print("Generating sequences (Converting rows into 72-hour chunks)...")

def sequence(inp_data):
    np_data = inp_data.values.astype(np.float32)
    max_sequences = len(np_data) - inp - cast + 1
    X = np.zeros((max_sequences, inp, np_data.shape[1]), dtype=np.float32)
    y = np.zeros((max_sequences, cast), dtype=np.float32)

    for i in range(max_sequences):
        X[i] = np_data[i : i + inp]
        y[i] = np_data[i + inp : i + inp + cast, target_index]
    return X, y

X_train, y_train = sequence(train_data)
X_val, y_val = sequence(val_data)
X_test, y_test = sequence(test_data)

class WeatherSequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X)
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.y[index]

batch_size = 256
train_loader = DataLoader(WeatherSequenceDataset(X_train, y_train), batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(WeatherSequenceDataset(X_val, y_val), batch_size, shuffle=False)
test_loader = DataLoader(WeatherSequenceDataset(X_test, y_test), batch_size, shuffle=False)

# ==========================================
# 6. TRANSFORMER ARCHITECTURE
# ==========================================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return x

class JenaTransformer(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int, 
                 nhead: int, num_layers: int, dropout: float = 0.2):
        super().__init__()
        self.d_model = d_model
        
        self.project = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model * 4, 
            dropout=dropout, 
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, output_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = self.project(x)
        x_pos = self.pos_enc(x_proj)
        x_enc = self.encoder(x_pos)
        x_pooled = x_enc.mean(dim=1) 
        out = self.head(x_pooled) 
        return out

# ==========================================
# 7. TRAINING SETUP
# ==========================================
input_features = X_train.shape[2]
forecast_horizon = cast

model = JenaTransformer(
    input_dim=input_features, 
    output_dim=forecast_horizon, 
    d_model=64,       
    nhead=4,          
    num_layers=3,
    dropout=0.2
).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
loss_fn = nn.HuberLoss(delta=1.75)

def train_epoch(model, optimizer, loss_fn):
    running_t = 0.0
    running_v = 0.0

    model.train()
    for xt, yt in train_loader:
        xt, yt = xt.to(device), yt.to(device)
        optimizer.zero_grad()
        y_ = model(xt)
        loss = loss_fn(y_, yt)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        running_t += loss.item()

    model.eval()
    with torch.no_grad():
        for xv, yv in val_loader:
            xv, yv = xv.to(device), yv.to(device)
            y_ = model(xv)
            loss = loss_fn(y_, yv)
            running_v += loss.item()

    return running_t / len(train_loader), running_v / len(val_loader)

# ==========================================
# 8. TRAINING LOOP
# ==========================================
train_losses, val_losses = [], []
least_loss = float('inf')

print("Starting training loop...")
for i in range(1, 31):
    train_loss, val_loss = train_epoch(model, optimizer, loss_fn)
    train_losses.append(train_loss)
    val_losses.append(val_loss)
    print(f'Epoch {i:02d} | Train: {train_loss:5f} | Val: {val_loss:5f}')
    scheduler.step(val_loss)

    if val_loss < least_loss:
        torch.save(model.state_dict(), 'jena_transformer.pt')
        least_loss = val_loss

# Load best model
model.load_state_dict(torch.load('jena_transformer.pt', weights_only=True))
print(f'Best Validation Loss: {least_loss}')

# ==========================================
# 9. EVALUATION & METRICS
# ==========================================
def evaluate(model, loader):
    predictions, actuals = [], []
    model.eval()
    with torch.no_grad():
        for xt, yt in loader:
            xt, yt = xt.to(device), yt.to(device)
            y_ = model(xt)
            predictions.append(y_.cpu().numpy())
            actuals.append(yt.cpu().numpy())
    return np.concatenate(predictions), np.concatenate(actuals)

predictions, actuals = evaluate(model, test_loader)
predictions_ = predictions * target_std + target_mean
actuals_ = actuals * target_std + target_mean

print('\nEvaluation of Values in Celsius:')
predictions_tensor_ = torch.from_numpy(predictions_)
actuals_tensor_ = torch.from_numpy(actuals_)
print(f'MSE:\t{F.mse_loss(predictions_tensor_, actuals_tensor_)}')
print(f'MAE:\t{F.l1_loss(predictions_tensor_, actuals_tensor_)}')
print(f'Huber:\t{F.huber_loss(predictions_tensor_, actuals_tensor_)}')

# ==========================================
# 10. PLOTTING
# ==========================================
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label='Training Loss (Huber)', color='#0b57d0')
plt.plot(val_losses, label='Validation Loss (Huber)', color='#d85900')
plt.title('Transformer Training vs Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax in axes:
    idx = random.randint(0, len(predictions) - 1)
    ax.plot(predictions_[idx], label='Transformer Forecast', marker='o', color='#0b57d0')
    ax.plot(actuals_[idx], label='Actual Temperature', marker='o', color='#146c2e')
    ax.set_title(f'Sample Forecast Window (Index: {idx})')
    ax.set_xlabel('Future Hours')
    ax.set_ylabel('Temperature (°C)')
    
    min_val = min(min(predictions_[idx]), min(actuals_[idx]))
    max_val = max(max(predictions_[idx]), max(actuals_[idx]))
    padding = max((max_val - min_val) * 0.4, 3.0) 
    ax.set_ylim(min_val - padding, max_val + padding)
    
    ax.grid(alpha=0.4)
    ax.legend()
plt.tight_layout()
plt.show()