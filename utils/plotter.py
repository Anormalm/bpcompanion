import pandas as pd
import matplotlib.pyplot as plt

def plot_readings():
    try:
        df = pd.read_csv('data/readings.csv')
        df['datetime'] = pd.to_datetime(df['date']+ ' ' + df['time'])
        df = df.sort_values('datetime')

        df['systolic'] = pd.to_numeric(df['systolic'], errors='coerce')
        df['diastolic'] = pd.to_numeric(df['diastolic'], errors='coerce')


        plt.figure(figsize=(10, 5))
        plt.plot(df['datetime'], df['systolic'], label='Systolic', marker='o')
        plt.plot(df['datetime'], df['diastolic'], label='Diastolic', marker='x')
        plt.title('Blood Pressure Trend')
        plt.xlabel('Date')
        plt.ylabel('Blood Pressure (mmHg)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"Error plotting trend:", e)
