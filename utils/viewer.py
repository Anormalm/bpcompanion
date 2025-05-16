import pandas as pd

def view_summary():
    try:
        df = pd.read_csv('data/readings.csv')
        print("\nRecent BP Readings:")
        print(df.tail(5).to_string(index=False))

        avg_sys = df['systolic'].astype(float).mean()
        avg_dia = df['diastolic'].astype(float).mean()
        print(f"\nAverage BP: {avg_sys:.1f}/{avg_dia:.1f} mmHg")
        
    except Exception as e:
        print(f"Error reading data: {e}")
