import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest


def detect_anomalies():
    try:
        df = pd.read_csv('data/readings.csv')
        df['systolic'] = pd.to_numeric(df['systolic'], errors='coerce')
        df['diastolic'] = pd.to_numeric(df['diastolic'], errors='coerce')
        df['pulse'] = pd.to_numeric(df['pulse'], errors='coerce')  
        df = df.dropna()

        rules = df[(df['systolic'] > 140) | (df['diastolic'] > 90) | (df['pulse'] > 120)]

        features = df[['systolic', 'diastolic', 'pulse']]
        model = IsolationForest(contamination=0.1, random_state=42)
        df['anomaly'] = model.fit_predict(features)

        ml_outliers = df[df['anomaly'] == -1]
        print(f"\nRule-based anomalies detected: {len(rules)}")
        print(rules[['date', 'systolic', 'diastolic', 'pulse']].to_string(index=False))
        
        print(f"\nMachine learning anomalies detected: {len(ml_outliers)}")
        print(ml_outliers[['date', 'systolic', 'diastolic', 'pulse']].to_string(index=False))

        
    except Exception as e:
        print(f"Error detecting anomalies:", e)