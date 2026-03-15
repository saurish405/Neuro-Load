import pandas as pd
import numpy as np
import os

# Ensure the data folder exists
if not os.path.exists('data'):
    os.makedirs('data')

def generate_traffic(days=3):
    print("Generating synthetic traffic data...")
    time_index = pd.date_range("2026-01-01", periods=1440*days, freq="min")
    
    # Sine wave for day/night + random noise + random spikes
    base = 50 + 40 * np.sin(np.arange(len(time_index)) * (2 * np.pi / 1440))
    noise = np.random.normal(0, 5, len(time_index))
    spikes = np.where(np.random.rand(len(time_index)) > 0.99, np.random.randint(20, 80), 0)
    
    traffic = np.maximum(5, base + noise + spikes)
    df = pd.DataFrame({'timestamp': time_index, 'requests': traffic.astype(int)})
    df.to_csv('data/traffic_logs.csv', index=False)
    print("Success: data/traffic_logs.csv created.")


if __name__ == "__main__":
    generate_traffic()