import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from random import randint

# ==========================================
# PHASE 1: INITIALIZATION & DATA INGESTION
# ==========================================
dataset_url = "https://storage.googleapis.com/tensorflow/tf-keras-datasets/jena_climate_2009_2016.csv.zip"

random_seed = 1305
torch.manual_seed(random_seed)
np.random.seed(random_seed)

compute_device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Active device:', compute_device)

print("Downloading and loading climate data...")
climate_data = pd.read_csv(dataset_url, compression='zip')
climate_data['Date Time'] = pd.to_datetime(climate_data['Date Time'], format='%d.%m.%Y %H:%M:%S')
climate_data.set_index('Date Time', inplace=True)
climate_data.sort_index(inplace=True)

# ==========================================
# PHASE 2: DATA CLEANING & FEATURE ENGINEERING
# ==========================================
# Fixing physical sensor errors
climate_data.loc[climate_data['wv (m/s)'] < -1, 'wv (m/s)'] = 0.0
climate_data.loc[climate_data['max. wv (m/s)'] < -1, 'max. wv (m/s)'] = 0.0

wind_vel = climate_data.pop("wv (m/s)")
max_wind_vel = climate_data.pop("max. wv (m/s)")
wind_deg = climate_data.pop("wd (deg)") * np.pi / 180.0

# Converting wind to vector components
climate_data['wx'] = wind_vel * np.cos(wind_deg)
climate_data['wy'] = wind_vel * np.sin(wind_deg)
climate_data['mwx'] = max_wind_vel * np.cos(wind_deg)
climate_data['mwy'] = max_wind_vel * np.sin(wind_deg)

# Resampling to hourly frequency and handle missing values
climate_data = climate_data.resample('1h').mean()
climate_data = climate_data.interpolate(limit_direction='both')

# Generating cyclical time features
seconds = climate_data.index.map(pd.Timestamp.timestamp)
seconds_in_day = 24 * 60 * 60
seconds_in_year = 365.2425 * seconds_in_day

climate_data['day_sin'] = np.sin(seconds * 2 * np.pi / seconds_in_day)
climate_data['day_cos'] = np.cos(seconds * 2 * np.pi / seconds_in_day)
climate_data['year_sin'] = np.sin(seconds * 2 * np.pi / seconds_in_year)
climate_data['year_cos'] = np.cos(seconds * 2 * np.pi / seconds_in_year)

target_col = 'T (degC)'
all_columns = list(climate_data.columns)
target_idx = all_columns.index(target_col)

# ==========================================
# PHASE 3: DATASET SPLITTING & SCALING
# ==========================================
print("Splitting and normalizing sequences...")
total_rows = len(climate_data)
train_split = int(total_rows * 0.7)
val_split = int(total_rows * 0.85)

train_df = climate_data.iloc[:train_split]
val_df = climate_data.iloc[train_split - 72 : val_split]
test_df = climate_data.iloc[val_split - 72 :]

# Calculate training statistics for scaling
train_mean = train_df[all_columns].mean()
train_std = train_df[all_columns].std() 

# Apply scaling
train_df = (train_df[all_columns] - train_mean) / train_std
val_df = (val_df[all_columns] - train_mean) / train_std
test_df = (test_df[all_columns] - train_mean) / train_std

temp_mean = train_mean.iloc[target_idx]
temp_std = train_std.iloc[target_idx]

def create_sliding_windows(dataframe, t_idx, history_len=72, forecast_len=12):
    """Generates sequential input/output pairs for the model."""
    raw_values = dataframe.values.astype(np.float32)
    valid_windows = len(raw_values) - history_len - forecast_len + 1 
    
    X_windows = np.zeros((valid_windows, history_len, raw_values.shape[1]), dtype=np.float32) 
    y_windows = np.zeros((valid_windows, forecast_len), dtype=np.float32)

    for i in range(valid_windows):
        X_windows[i] = raw_values[i : i + history_len]
        y_windows[i] = raw_values[i + history_len : i + history_len + forecast_len, t_idx]

    return X_windows, y_windows

X_train, y_train = create_sliding_windows(train_df, target_idx)
X_test, y_test = create_sliding_windows(test_df, target_idx)
X_val, y_val = create_sliding_windows(val_df, target_idx)

class TimeSeriesDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X) 
        self.y = torch.from_numpy(y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

batch_size = 128 
train_loader = DataLoader(TimeSeriesDataset(X_train, y_train), batch_size=batch_size, shuffle=True, drop_last=True)
val_loader = DataLoader(TimeSeriesDataset(X_val, y_val), batch_size=batch_size, shuffle=False)
test_loader = DataLoader(TimeSeriesDataset(X_test, y_test), batch_size=batch_size, shuffle=False)

# ==========================================
# PHASE 4: MODEL ARCHITECTURE
# ==========================================
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

class ForecastingModel(nn.Module):
    def __init__(self, feature_dim, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lstm_layer = CustomLSTMCell(feature_dim, self.hidden_dim)

        self.output_head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.hidden_dim, 12)
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
# PHASE 5: TRAINING PIPELINE
# ==========================================
if __name__ == "__main__":
    print("Initializing model training...")
    num_features = len(all_columns)
    model = ForecastingModel(num_features, 128).to(compute_device) 
    
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=2e-5) 
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2) 
    loss_criterion = nn.HuberLoss()

    train_loss_history = []
    val_loss_history = []

    def train_single_epoch(current_model, opt, criterion):
        total_train_loss = 0.0
        total_val_loss = 0.0

        current_model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(compute_device), batch_y.to(compute_device)

            opt.zero_grad()
            predictions = current_model(batch_x)
            loss = criterion(predictions, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(current_model.parameters(), 5.0)
            opt.step()

            total_train_loss += loss.item()

        current_model.eval()
        with torch.no_grad():
            for val_x, val_y in val_loader:
                val_x, val_y = val_x.to(compute_device), val_y.to(compute_device)
                val_preds = current_model(val_x)
                loss = criterion(val_preds, val_y)
                total_val_loss += loss.item()

        avg_train = total_train_loss / len(train_loader)
        avg_val = total_val_loss / len(val_loader)
        return avg_train, avg_val

    total_epochs = 18
    best_val_loss = float('inf')
    best_weights = None

    for epoch in range(1, total_epochs + 1):
        loss_t, loss_v = train_single_epoch(model, optimizer, loss_criterion)
        train_loss_history.append(loss_t)
        val_loss_history.append(loss_v)
        
        print(f'Epoch {epoch:02d} | Train Loss: {loss_t:.5f} | Val Loss: {loss_v:.5f}')
        
        lr_scheduler.step(loss_v)

        if loss_v < best_val_loss:
            best_weights = model.state_dict().copy()
            best_val_loss = loss_v
            print("   -> Checkpoint saved (New best validation loss)")

    model.load_state_dict(best_weights)
    print(f"\nTraining Complete. Best Validation Loss: {best_val_loss:.5f}")

    # ==========================================
    # PHASE 6: EVALUATION & VISUALIZATION
    # ==========================================
    plt.figure(figsize=(10, 4))
    plt.plot(train_loss_history, label='Training Loss')
    plt.plot(val_loss_history, label='Validation Loss')
    plt.title("Model Convergence History")
    plt.xlabel("Epoch")
    plt.ylabel("Huber Loss")
    plt.legend()
    plt.grid(True, linestyle=":", alpha=0.7)
    plt.show()

    def evaluate_test_set(eval_model, data_loader):
        preds_list, actuals_list = [], []
        eval_model.eval()
        with torch.no_grad():
            for x, y in data_loader:
                x, y = x.to(compute_device), y.to(compute_device)
                output = eval_model(x)
                preds_list.append(output.cpu().numpy())
                actuals_list.append(y.cpu().numpy())

        return np.concatenate(preds_list), np.concatenate(actuals_list)

    raw_preds, raw_actuals = evaluate_test_set(model, test_loader)

    # Convert normalized values back to Celsius
    final_preds = raw_preds * temp_std + temp_mean
    final_actuals = raw_actuals * temp_std + temp_mean

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("12-Hour Temperature Forecast vs Actuals", fontsize=15)

    random_samples = [randint(0, len(final_preds)//3), 
                      randint(len(final_preds)//3, 2*len(final_preds)//3), 
                      randint(2*len(final_preds)//3, len(final_preds)-1)]

    for ax, idx in zip(axes, random_samples):
        ax.plot(final_actuals[idx], label='Actual Temp (°C)', color='green', marker='o')
        ax.plot(final_preds[idx], label='Forecast (°C)', color='red', linestyle='--', marker='x')
        ax.set_xlabel("Hours into future")
        ax.set_ylabel("Temperature (°C)")
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.legend()

    plt.tight_layout()
    plt.show()

    # Compute final metrics on unscaled data
    tensor_preds = torch.from_numpy(final_preds)
    tensor_actuals = torch.from_numpy(final_actuals)

    final_mse = F.mse_loss(tensor_preds, tensor_actuals)
    final_mae = F.l1_loss(tensor_preds, tensor_actuals)
    final_huber = F.huber_loss(tensor_preds, tensor_actuals)

    print('\nFinal Test Set Metrics (in Celsius):')
    print('-' * 40)
    print(f'Mean Squared Error (MSE):  {final_mse:.5f}')
    print(f'Mean Absolute Error (MAE): {final_mae:.5f}')
    print(f'Huber Loss:                {final_huber:.5f}')