import pandas as pd
import numpy as np
import os

def generate_stress():
    if not os.path.exists('data'): os.makedirs('data')
    
    # 120 minutes of simulation
    time_index = pd.date_range("2026-01-01", periods=120, freq="min")
    
    # Logic: 40 mins normal -> 30 mins MASSIVE SPIKE -> 50 mins recovery
    normal = [np.random.randint(40, 60) for _ in range(40)]
    spike = [np.random.randint(400, 550) for _ in range(30)]
    recovery = [np.random.randint(60, 80) for _ in range(50)]
    
    traffic = normal + spike + recovery
    
    df = pd.DataFrame({'timestamp': time_index, 'requests': traffic})
    df.to_csv('data/stress_traffic.csv', index=False)
    print("Success: data/stress_traffic.csv created for Flash Crowd testing.")

if __name__ == "__main__":
    generate_stress()