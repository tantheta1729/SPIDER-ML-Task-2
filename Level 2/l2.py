import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt

# Set random seed for absolute reproducibility across execution runs
torch.manual_seed(42)
np.random.seed(42)

# =====================================================================
# PHASE 1: DATA LOADING & HOURLY DOWNSAMPLING (image_a26ba8.png)
# =====================================================================

def load_and_preprocess_climate_data(csv_path):
    if os.path.exists(csv_path):
        print(f"Success: Loading dataset from local file -> {csv_path}")
        df = pd.read_csv(csv_path)
    else:
        print(f"⚠️ {csv_path} not found in this folder!")
        print("Generating a temporary mock dataset so the pipeline runs immediately...")
        date_range = pd.date_range(start="2026-01-01", periods=12000, freq="10min")
        df = pd.DataFrame({
            "Date Time": date_range,
            "p (mbar)": np.random.normal(1013, 10, size=12000),
            "T (degC)": np.random.normal(15, 5, size=12000) + np.sin(np.linspace(0, 50, 12000)) * 5,
            "rho (g/m**3)": np.random.normal(1200, 20, size=12000)
        })

    numeric_df = df.select_dtypes(include=[np.number])
    feature_names = list(numeric_df.columns)
    raw_array = numeric_df.values
    
    num_hours = len(raw_array) // 6
    trimmed_array = raw_array[:num_hours * 6]
    reshaped = trimmed_array.reshape(num_hours, 6, -1)
    hourly_data = np.mean(reshaped, axis=1)
    
    temp_col_idx = feature_names.index("T (degC)") if "T (degC)" in feature_names else 1
    return hourly_data, temp_col_idx

# =====================================================================
# PHASE 2: SLIDING WINDOW DATASET MANAGEMENT (image_a25487.png)
# =====================================================================

class SlidingWindowWeatherDataset(Dataset):
    def __init__(self, hourly_data, temp_idx, input_hours=72, forecast_hours=12):
        self.input_hours = input_hours
        self.forecast_hours = forecast_hours
        self.temp_idx = temp_idx
        
        self.scaler = StandardScaler()
        self.scaled_features = self.scaler.fit_transform(hourly_data)
        
        self.target_scaler = StandardScaler()
        self.target_scaler.fit(hourly_data[:, [temp_idx]])
        
    def __len__(self):
        return len(self.scaled_features) - self.input_hours - self.forecast_hours + 1
        
    def __getitem__(self, idx):
        x = self.scaled_features[idx : idx + self.input_hours]
        y = self.scaled_features[idx + self.input_hours : idx + self.input_hours + self.forecast_hours, self.temp_idx]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

# =====================================================================
# PHASE 3: HANDWRITTEN CELL WITH ORTHOGONAL INITIALIZATION
# =====================================================================

class CustomLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(CustomLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        combined_size = input_size + hidden_size
        
        self.forget_gate = nn.Linear(combined_size, hidden_size)
        self.input_gate = nn.Linear(combined_size, hidden_size)
        self.candidate_gate = nn.Linear(combined_size, hidden_size)
        self.output_gate = nn.Linear(combined_size, hidden_size)
        
        self._init_orthogonal_weights()
        
    def _init_orthogonal_weights(self):
        for gate in [self.forget_gate, self.input_gate, self.candidate_gate, self.output_gate]:
            nn.init.xavier_uniform_(gate.weight[:, :self.input_size])
            nn.init.orthogonal_(gate.weight[:, self.input_size:])
            nn.init.constant_(gate.bias, 0.0)
        nn.init.constant_(self.forget_gate.bias, 1.0)
        
    def forward(self, x_t, h_prev, c_prev):
        combined = torch.cat((x_t, h_prev), dim=1)
        f_t = torch.sigmoid(self.forget_gate(combined))       
        i_t = torch.sigmoid(self.input_gate(combined))        
        c_tilde = torch.tanh(self.candidate_gate(combined))   
        c_next = (f_t * c_prev) + (i_t * c_tilde)             
        o_t = torch.sigmoid(self.output_gate(combined))       
        h_next = o_t * torch.tanh(c_next)                     
        return h_next, c_next

# =====================================================================
# PHASE 4: STACKED ARCHITECTURE WITH AMPLITUDE GAIN CONTROLLER
# =====================================================================

class StackedLSTMWeatherForecaster(nn.Module):
    def __init__(self, input_size, temp_idx, hidden_size=128, forecast_output_dim=12):
        super(StackedLSTMWeatherForecaster, self).__init__()
        self.hidden_size = hidden_size
        self.temp_idx = temp_idx 
        
        self.layer1 = CustomLSTMCell(input_size, hidden_size)
        self.layer2 = CustomLSTMCell(hidden_size, hidden_size)
        
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.regression_head = nn.Linear(hidden_size, forecast_output_dim)
        
    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        baseline_temp = x[:, -1, self.temp_idx].unsqueeze(1)
        
        h1 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c1 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        h2 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        c2 = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        for t in range(seq_len):
            x_t = x[:, t, :]
            h1, c1 = self.layer1(x_t, h1, c1)       
            h2, c2 = self.layer2(h1, h2, c2)       
            
        normalized_features = self.layer_norm(h2)
        
        # 🚀 THE AMPLITUDE CONTROLLER: Calculate the raw deviations from the baseline
        raw_deviation = self.regression_head(self.dropout(normalized_features))
        
        # Multiply the deviation by 0.4 to compress the wave's overshoot and match the stable winter scale
        out_forecast = (0.4 * raw_deviation) + baseline_temp
        return out_forecast

# =====================================================================
# PHASE 5: EVALUATION ENGINE & METRIC PARSING (image_a254c1.png)
# =====================================================================

def evaluate_model_performance(model, data_loader, temp_idx, device, loss_fn):
    model.eval()
    all_predictions, all_targets, all_last_hour_temps = [], [], []
    total_val_loss = 0.0
    
    with torch.no_grad():
        for x_batch, y_batch in data_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            preds = model(x_batch)
            
            loss = loss_fn(preds, y_batch)
            total_val_loss += loss.item() * x_batch.size(0)
            
            all_predictions.append(preds.cpu())
            all_targets.append(y_batch.cpu())
            all_last_hour_temps.append(x_batch[:, -1, temp_idx].cpu())
            
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_last_hour_temps = torch.cat(all_last_hour_temps, dim=0)
    
    mean_val_loss = total_val_loss / len(data_loader.dataset)
    pred_mean_trends = (all_predictions.mean(dim=1) > all_last_hour_temps).long().numpy()
    target_mean_trends = (all_targets.mean(dim=1) > all_last_hour_temps).long().numpy()
    
    print("\n" + "="*60)
    print("SUBMISSION QUALITY METRICS REPORT (image_a254c1.png)")
    print("="*60)
    print(f"Validation Loss (MSE): {mean_val_loss:.5f}")
    print("\nConfusion Matrix:")
    print(confusion_matrix(target_mean_trends, pred_mean_trends))
    print("\nClassification Report Summary:")
    print(classification_report(target_mean_trends, pred_mean_trends, target_names=["Temp Decrease/Steady", "Temp Increase"], zero_division=0))
    
    return mean_val_loss, all_predictions, all_targets

# =====================================================================
# PHASE 6: EXECUTION PIPELINE
# =====================================================================

if __name__ == "__main__":
    CSV_FILE_PATH = "jena_climate_2009_2016.csv"
    INPUT_WINDOW = 72
    FORECAST_WINDOW = 12
    BATCH_SIZE = 128
    HIDDEN_DIM = 128        
    LEARNING_RATE = 0.001  
    EPOCHS = 20            
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing processing pipeline. Target device: {device}")
    
    hourly_data, temp_column_idx = load_and_preprocess_climate_data(CSV_FILE_PATH)
    dataset = SlidingWindowWeatherDataset(hourly_data, temp_idx=temp_column_idx, input_hours=INPUT_WINDOW, forecast_hours=FORECAST_WINDOW)
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    num_features = hourly_data.shape[1]
    model = StackedLSTMWeatherForecaster(input_size=num_features, temp_idx=temp_column_idx, hidden_size=HIDDEN_DIM, forecast_output_dim=FORECAST_WINDOW).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    loss_criterion = nn.MSELoss()
    
    train_hist, val_hist = [], []
    
    print("\nTraining custom network parameters across sliding windows...")
    for epoch in range(EPOCHS):
        model.train()
        running_train_loss = 0.0
        
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(x_batch)
            loss = loss_criterion(outputs, y_batch)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            running_train_loss += loss.item() * x_batch.size(0)
            
        epoch_train_loss = running_train_loss / len(train_loader.dataset)
        train_hist.append(epoch_train_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Complete -> Train MSE Loss: {epoch_train_loss:.5f}")
        epoch_val_loss, _, _ = evaluate_model_performance(model, val_loader, temp_column_idx, device, loss_criterion)
        val_hist.append(epoch_val_loss)
        
    print("\nGenerating final stabilized submission charts...")
    
    # Chart A: Convergence Curve History window
    plt.figure(figsize=(10, 4))
    plt.plot(train_hist, label="Training Loss", color="dodgerblue", lw=2)
    plt.plot(val_hist, label="Validation Loss", color="orange", lw=2)
    plt.title("Optimized Model Performance: MSE Loss History")
    plt.xlabel("Training Epoch Count")
    plt.ylabel("Loss Scale")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.show()

    # Chart B: Real vs Calibrated Output window
    model.eval()
    with torch.no_grad():
        sample_x, sample_y = next(iter(val_loader))
        sample_preds = model(sample_x.to(device)).cpu().numpy()
        sample_y = sample_y.numpy()
        
    plt.figure(figsize=(10, 4))
    unscaled_truth = dataset.target_scaler.inverse_transform(sample_y[0].reshape(-1, 1)).flatten()
    unscaled_prediction = dataset.target_scaler.inverse_transform(sample_preds[0].reshape(-1, 1)).flatten()
    
    plt.plot(unscaled_truth, label="True Ground Temperature Path", color="darkgreen", marker='o', lw=2)
    plt.plot(unscaled_prediction, label="Model Handwritten Forecast Prediction", color="crimson", linestyle="--", marker='x', lw=2)
    plt.title("Perfected Evaluation Window: 12-Hour Weather Trend Trajectory Analysis")
    plt.xlabel("Future Projection Timestep (Hours)")
    plt.ylabel("Temperature Profile (°C)")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.show()