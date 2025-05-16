import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from fpdf import FPDF
import torch
import torch.nn as nn

# LSTM model class
class MultiLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=64, output_size=3):
        super(MultiLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def generate_report():
    try:
        df = pd.read_csv('data/readings.csv')
        df['systolic'] = pd.to_numeric(df['systolic'], errors='coerce')
        df['diastolic'] = pd.to_numeric(df['diastolic'], errors='coerce')
        df['pulse'] = pd.to_numeric(df['pulse'], errors='coerce')
        df = df.dropna()

        # ========== 1. Averages ==========
        avg_sys = df['systolic'].mean()
        avg_dia = df['diastolic'].mean()
        avg_pulse = df['pulse'].mean()

        # ========== 2. Recent readings ==========
        recent = df.tail(5)

        # ========== 3. Anomaly Detection ==========
        rule_anomalies = df[(df['systolic'] > 160) | (df['diastolic'] > 100) | (df['pulse'] > 120)]

        # ========== 4. LSTM Forecast ==========
        features = df[['systolic', 'diastolic', 'pulse']].values
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(features)

        X, y = [], []
        for i in range(5, len(scaled)):
            X.append(scaled[i-5:i])
            y.append(scaled[i])
        X_tensor = torch.tensor(np.array(X), dtype=torch.float32)
        model = MultiLSTM()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = nn.MSELoss()

        model.train()
        for epoch in range(50):
            optimizer.zero_grad()
            out = model(X_tensor)
            loss = criterion(out, torch.tensor(np.array(y), dtype=torch.float32))
            loss.backward()
            optimizer.step()

        model.eval()
        preds = []
        last_seq = X_tensor[-1].unsqueeze(0)
        for _ in range(5):
            with torch.no_grad():
                next_val = model(last_seq)
            preds.append(next_val.numpy()[0])
            last_seq = torch.cat([last_seq[:, 1:, :], next_val.view(1, 1, 3)], dim=1)
        forecast = scaler.inverse_transform(np.array(preds))

        # ========== 5. Generate PDF ==========
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font('Arial', 'B', 14)
        pdf.cell(0, 10, "Grandpa's Blood Pressure Report", ln=True, align='C')

        pdf.set_font('Arial', '', 11)
        pdf.ln(8)
        pdf.cell(0, 10, f"Average Systolic: {avg_sys:.1f} mmHg", ln=True)
        pdf.cell(0, 10, f"Average Diastolic: {avg_dia:.1f} mmHg", ln=True)
        pdf.cell(0, 10, f"Average Pulse: {avg_pulse:.1f} bpm", ln=True)

        pdf.ln(8)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Recent Readings:", ln=True)
        pdf.set_font('Arial', '', 10)
        for _, row in recent.iterrows():
            pdf.cell(0, 8, f"{row['date']} {row['time']} - {row['systolic']}/{row['diastolic']} mmHg, Pulse: {row['pulse']} | {row['notes']}", ln=True)

        pdf.ln(8)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Rule-based Anomalies (>160/100 mmHg or Pulse > 120):", ln=True)
        pdf.set_font('Arial', '', 10)
        if rule_anomalies.empty:
            pdf.cell(0, 8, "No anomalies detected.", ln=True)
        else:
            for _, row in rule_anomalies.iterrows():
                pdf.cell(0, 8, f"{row['date']} {row['systolic']}/{row['diastolic']}, Pulse: {row['pulse']}", ln=True)

        pdf.ln(8)
        pdf.set_font('Arial', 'B', 12)
        pdf.cell(0, 10, "Forecast (Next 5 Days):", ln=True)
        pdf.set_font('Arial', '', 10)
        for i, f in enumerate(forecast):
            pdf.cell(0, 8, f"Day +{i+1}: {f[0]:.1f} / {f[1]:.1f} mmHg, Pulse: {f[2]:.1f} bpm", ln=True)

        pdf.output("bp_full_report.pdf")
        print("Full report saved as bp_full_report.pdf")

    except Exception as e:
        print("Failed to generate full report:", e)