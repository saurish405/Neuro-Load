import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ── 1. AI RESULTS (Use your current numbers from main2.py) ──────────────────
AI_TASKS = [23173, 42598, 157582]   
SPEEDS   = [0.5, 1.0, 1.5]          
# ─────────────────────────────────────────────────────────────────────────────

DATA_PATH = 'data/google_data.csv'

def run_summary():
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
        from baselines import get_traditional_results
    except ImportError:
        print("Error: Ensure src/baselines.py exists!")
        return

    # Run traditional baselines
    (rr_stats, rr_time), (wrr_stats, wrr_time), (lc_stats, lc_time) = get_traditional_results(DATA_PATH)

    # Normalization (Matches the work volume)
    total_ai = sum(AI_TASKS)
    total_rr = sum(rr_stats)
    if total_rr > 0:
        scale = total_ai / total_rr
        rr_stats = [s * scale for s in rr_stats]; rr_time *= scale
        wrr_stats = [s * scale for s in wrr_stats]; wrr_time *= scale
        lc_stats = [s * scale for s in lc_stats]; lc_time *= scale

    # Correct Time Calculation
    ai_time = sum(AI_TASKS[i] / SPEEDS[i] for i in range(3))

    # ── 2. THE FORMATTED OUTPUT (MATCHING THE SCREENSHOT) ───────────────────
    print("\n" + "=" * 65)
    print(f"{'METHOD':<22} | {'S1 Tasks':>9} | {'S2 Tasks':>9} | {'S3 Tasks':>9}")
    print("-" * 65)
    
    methods_table = [
        ("Round Robin", rr_stats),
        ("Weighted RR", wrr_stats),
        ("Least Connections", lc_stats),
        ("Hybrid AI (Ours)", AI_TASKS)
    ]
    
    for name, stats in methods_table:
        print(f"{name:<22} | {int(stats[0]):>9,} | {int(stats[1]):>9,} | {int(stats[2]):>9,}")

    print("\n" + "=" * 65)
    print(f"{'METHOD':<22} | {'PROC. TIME (s)':>15} | {'vs Round Robin':>14}")
    print("-" * 65)
    
    efficiency_table = [
        ("Round Robin", rr_time),
        ("Weighted RR", wrr_time),
        ("Least Connections", lc_time),
        ("Hybrid AI (Ours)", ai_time),
    ]
    
    best_time = min(t for _, t in efficiency_table)

    for name, t in efficiency_table:
        gain = ((rr_time - t) / rr_time) * 100
        marker = "  <-- BEST" if t == best_time else ""
        print(f"{name:<22} | {t:>15,.2f} | {gain:>+13.1f}%{marker}")
    print("=" * 65)

    # ── 3. PLOTTING (Separate Windows) ──────────────────────────────────────
    plt.figure(num="Workload Comparison", figsize=(10, 6))
    labels = ['S1', 'S2', 'S3']
    x = np.arange(len(labels))
    width = 0.2
    plt.bar(x - 1.5*width, rr_stats, width, label='Round Robin', color='#bdc3c7')
    plt.bar(x - 0.5*width, wrr_stats, width, label='Weighted RR', color='#95a5a6')
    plt.bar(x + 0.5*width, lc_stats, width, label='Least Conn', color='#34495e')
    plt.bar(x + 1.5*width, AI_TASKS, width, label='Hybrid AI', color='#2ecc71', edgecolor='black')
    plt.xticks(x, labels); plt.legend(); plt.title("Task Distribution")

    plt.figure(num="Efficiency Gain", figsize=(8, 6))
    method_names = [m for m, _ in efficiency_table]
    times = [t for _, t in efficiency_table]
    plt.bar(method_names, times, color=['#bdc3c7', '#95a5a6', '#34495e', '#2ecc71'])
    plt.title("Total Processing Time (Lower is Better)")
    plt.show()

if __name__ == "__main__":
    run_summary()
