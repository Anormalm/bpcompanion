import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

class MultiLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super(MultiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # last timestep
        return out

def train_lstm_model():
    try:
        df = pd.read_csv('data/readings.csv')
        df['systolic'] = pd.to_numeric(df['systolic'], errors='coerce')
        df['diastolic'] = pd.to_numeric(df['diastolic'], errors='coerce')
        df['pulse'] = pd.to_numeric(df['pulse'], errors='coerce')
        df = df.dropna()

        features = df[['systolic', 'diastolic', 'pulse']].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)

        # Create sequences of shape (seq_len=5, features=3)
        X, y = [], []
        for i in range(5, len(scaled)):
            X.append(scaled[i-5:i])
            y.append(scaled[i])
        X, y = np.array(X), np.array(y)

        # Convert to tensors
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Model setup
        model = MultiLSTM(input_size=3, hidden_size=64, output_size=3)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        # Training
        model.train()
        for epoch in range(50):
            output = model(X_tensor)
            loss = criterion(output, y_tensor)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Forecast next 5 steps
        model.eval()
        last_seq = X_tensor[-1].unsqueeze(0)  # shape: (1, 5, 3)
        preds = []
        for _ in range(5):
            with torch.no_grad():
                next_val = model(last_seq)  # shape: (1, 3)
            preds.append(next_val.numpy()[0])
            next_val_reshaped = next_val.view(1, 1, 3)
            last_seq = torch.cat([last_seq[:, 1:, :], next_val_reshaped], dim=1)

        preds = np.array(preds)
        preds_rescaled = scaler.inverse_transform(preds)

        print("\nMultivariate Forecast (Systolic / Diastolic / Pulse):")
        for i, row in enumerate(preds_rescaled):
            print(f"Day +{i+1}: {row[0]:.1f} / {row[1]:.1f} mmHg, Pulse: {row[2]:.1f} bpm")

        # Plot Systolic Forecast
        plt.plot(features[:, 0], label='Systolic History')
        plt.plot(np.arange(len(features), len(features)+5), preds_rescaled[:, 0], 'o-', label='Systolic Forecast')

        # Plot Diastolic Forecast
        plt.plot(features[:, 1], label='Diastolic History')
        plt.plot(np.arange(len(features), len(features)+5), preds_rescaled[:, 1], 'x--', label='Diastolic Forecast')

        # Plot Pulse Forecast
        plt.plot(features[:, 2], label='Pulse History')
        plt.plot(np.arange(len(features), len(features)+5), preds_rescaled[:, 2], 's--', label='Pulse Forecast')

        plt.title("Blood Pressure & Pulse Forecast (PyTorch LSTM)")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print("Multivariate LSTM prediction failed:", e)