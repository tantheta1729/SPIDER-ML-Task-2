import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import random
import time

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
    dataset_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"
    print(f"Local file not found. Downloading from {dataset_url}...")
    data = pd.read_csv(dataset_url, compression='zip')

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

train_data_raw = data.iloc[:train_end]
mean = train_data_raw[columns].mean()
std = train_data_raw[columns].std()

target_index = columns.index(target_name)
target_mean = mean.iloc[target_index]
target_std = std.iloc[target_index]

data_norm = (data[columns] - mean) / std

train_data = data_norm.iloc[:train_end]
val_data = data_norm.iloc[train_end - inp : val_end]
test_data = data_norm.iloc[val_end - inp :]

# ==========================================
# 5. SEQUENCE GENERATION & DATALOADERS
# ==========================================
print("Generating sequences (72-hour input, 12-hour forecast)...")

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
# 6. MODEL ARCHITECTURES (TRANSFORMER & LSTM)
# ==========================================
# Primary Model: Transformer[cite: 2]
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
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, 
            dropout=dropout, batch_first=True
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

# Comparison Model: Custom LSTM[cite: 2]
class CustomLSTMCell(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.w_x = nn.Parameter(torch.empty(4 * hidden_dim, input_dim))
        self.w_h = nn.Parameter(torch.empty(4 * hidden_dim, hidden_dim))
        self.b_x = nn.Parameter(torch.empty(4 * hidden_dim))
        self.b_h = nn.Parameter(torch.empty(4 * hidden_dim))
        self.initialize_weights()

    def initialize_weights(self):
        limit = 1.0 / math.sqrt(self.hidden_dim)
        for param in self.parameters():
            nn.init.uniform_(param, -limit, limit)

    def forward(self, x, previous_states):
        h, c = previous_states
        gates = (x @ self.w_x.T + self.b_x) + (h @ self.w_h.T + self.b_h)
        i_gate, f_gate, c_gate, o_gate = gates.chunk(4, dim=1)
        i = torch.sigmoid(i_gate)
        f = torch.sigmoid(f_gate)
        c_candidate = torch.tanh(c_gate)
        o = torch.sigmoid(o_gate)
        c_next = f * c + i * c_candidate
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

class JenaCustomLSTM(nn.Module):
    def __init__(self, feature_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layer = CustomLSTMCell(feature_dim, self.hidden_dim)
        self.output_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.hidden_dim, output_dim)
        )

    def forward(self, x):
        batch, seq_len, _ = x.size()
        active_device = x.device
        h = torch.zeros(batch, self.hidden_dim, device=active_device)
        c = torch.zeros(batch, self.hidden_dim, device=active_device)

        for t in range(seq_len):
            x_t = x[:, t, :]
            h, c = self.lstm_layer(x_t, (h, c))

        return self.output_head(h)

# ==========================================
# 7. TRAINING UTILITY
# ==========================================
def train_model(model, name, epochs=20):
    print(f"\n--- Training {name} ---")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=5e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)
    loss_fn = nn.HuberLoss(delta=1.75)
    
    train_losses, val_losses = [], []
    least_loss = float('inf')
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        running_t = 0.0
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
        running_v = 0.0
        with torch.no_grad():
            for xv, yv in val_loader:
                xv, yv = xv.to(device), yv.to(device)
                y_ = model(xv)
                loss = loss_fn(y_, yv)
                running_v += loss.item()

        t_loss = running_t / len(train_loader)
        v_loss = running_v / len(val_loader)
        train_losses.append(t_loss)
        val_losses.append(v_loss)
        print(f'Epoch {epoch:02d} | Train: {t_loss:.5f} | Val: {v_loss:.5f}')
        scheduler.step(v_loss)

        if v_loss < least_loss:
            torch.save(model.state_dict(), f'jena_{name.lower()}.pt')
            least_loss = v_loss

    total_time = time.time() - start_time
    model.load_state_dict(torch.load(f'jena_{name.lower()}.pt', weights_only=True))
    print(f'Finished {name} Training in {total_time:.2f}s | Best Val Loss: {least_loss:.5f}')
    return train_losses, val_losses, total_time

# ==========================================
# 8. EXECUTION & TRAINING
# ==========================================
input_features = X_train.shape[2]

transformer_model = JenaTransformer(
    input_dim=input_features, output_dim=cast, 
    d_model=64, nhead=4, num_layers=3, dropout=0.2
).to(device)

lstm_model = JenaCustomLSTM(
    feature_dim=input_features, hidden_dim=128, output_dim=cast
).to(device)

tf_train_losses, tf_val_losses, tf_time = train_model(transformer_model, "Transformer", epochs=20)
lstm_train_losses, lstm_val_losses, lstm_time = train_model(lstm_model, "CustomLSTM", epochs=20)

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

tf_preds, actuals = evaluate(transformer_model, test_loader)
lstm_preds, _ = evaluate(lstm_model, test_loader)

tf_preds_c = tf_preds * target_std + target_mean
lstm_preds_c = lstm_preds * target_std + target_mean
actuals_c = actuals * target_std + target_mean

tf_tensor = torch.from_numpy(tf_preds_c)
lstm_tensor = torch.from_numpy(lstm_preds_c)
act_tensor = torch.from_numpy(actuals_c)

print('\n--- Comparative Evaluation Metrics (in Celsius) ---')
print('Transformer (Primary):')
print(f'  MSE:   {F.mse_loss(tf_tensor, act_tensor):.4f}')
print(f'  MAE:   {F.l1_loss(tf_tensor, act_tensor):.4f}')
print(f'  Runtime: {tf_time:.2f}s')

print('\nCustom LSTM (Comparison):')
print(f'  MSE:   {F.mse_loss(lstm_tensor, act_tensor):.4f}')
print(f'  MAE:   {F.l1_loss(lstm_tensor, act_tensor):.4f}')
print(f'  Runtime: {lstm_time:.2f}s')

# ==========================================
# 10. PLOTTING
# ==========================================
plt.figure(figsize=(10, 5))
plt.plot(tf_train_losses, label='Transformer Train Loss', color='#0b57d0', linestyle='--')
plt.plot(tf_val_losses, label='Transformer Val Loss', color='#0b57d0')
plt.plot(lstm_train_losses, label='Custom LSTM Train Loss', color='#d85900', linestyle='--')
plt.plot(lstm_val_losses, label='Custom LSTM Val Loss', color='#d85900')
plt.title('Training Convergence: Transformer vs Custom LSTM')
plt.xlabel('Epochs')
plt.ylabel('Huber Loss')
plt.grid(alpha=0.3)
plt.legend()
plt.show()

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
for ax in axes:
    idx = random.randint(0, len(tf_preds_c) - 1)
    ax.plot(actuals_c[idx], label='Actual Temp', marker='o', color='#146c2e')
    ax.plot(tf_preds_c[idx], label='Transformer Forecast', marker='x', color='#0b57d0')
    ax.plot(lstm_preds_c[idx], label='LSTM Forecast', marker='^', color='#d85900', alpha=0.7)
    ax.set_title(f'Sample Forecast Window (Index: {idx})')
    ax.set_xlabel('Future Hours')
    ax.set_ylabel('Temperature (°C)')
    ax.grid(alpha=0.4)
    ax.legend()
plt.tight_layout()
plt.show()