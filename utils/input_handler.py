import csv
from datetime import datetime

DATA_FILE = 'data/readings.csv'

def get_input():
    print("Enter today's BP reading:")
    date = input("Date (YYYY-MM-DD)[default:today]: ") or datetime.today().strftime('%Y-%m-%d')
    time = input("Time (HH:MM)[default:now]: ") or datetime.now().strftime('%H:%M')
    systolic = input("Systolic: ")
    diastolic = input("Diastolic: ")
    pulse = input("Pulse: ")
    notes = input("Notes(e.g., before/after meds, activity): ")
    return [date, time, systolic, diastolic, pulse, notes]

def save_entry(entry):
    with open(DATA_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(entry)
    print("Entry saved successfully!")

    