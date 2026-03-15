import pandas as pd
import numpy as np
from baselines import get_traditional_results

# PASTE YOUR ACTUAL AI RESULTS HERE
ai_tasks = [16845, 1532, 981] 
speeds = [0.5, 0.8, 1.2] # Seconds per task for S1, S2, S3

def calculate_summary():
    # 1. Calculate AI Time
    ai_time = sum(ai_tasks[i] * speeds[i] for i in range(3))
    
    # 2. Get Traditional Times from baselines.py
    (rr_stats, rr_time), (wrr_stats, wrr_time), (lc_stats, lc_time) = get_traditional_results('data/stress_traffic.csv')
    
    # 3. Compile Data
    methods = [
        ("Round Robin", rr_time),
        ("Weighted RR", wrr_time),
        ("Least Connections", lc_time),
        ("Hybrid AI (Ours)", ai_time)
    ]
    
    print("\n" + "="*50)
    print(f"{'METHOD':<25} | {'TOTAL PROC. TIME (SEC)':<20}")
    print("-" * 50)
    
    for name, time in methods:
        print(f"{name:<25} | {time:<20.2f}")
    
    print("="*50)
    
    # Efficiency Gain vs Round Robin
    gain = ((rr_time - ai_time) / rr_time) * 100
    print(f"Efficiency Improvement: {gain:.2f}% faster than Round Robin")

if __name__ == "__main__":
    calculate_summary()