import pandas as pd
import numpy as np

def get_traditional_results(csv_path, total_minutes=120):
    df = pd.read_csv(csv_path).head(total_minutes)
    traffic = df['requests'].values
    
    # Processing speeds (Seconds per task)
    speeds = [0.5, 0.8, 1.2] 

    # 1. Round Robin
    rr_stats = [0, 0, 0]
    for i, req in enumerate(traffic):
        for r in range(req):
            rr_stats[(i + r) % 3] += 1
    rr_time = sum(rr_stats[i] * speeds[i] for i in range(3))

    # 2. Weighted Round Robin
    wrr_stats = [0, 0, 0]
    weights = [0, 0, 0, 1, 1, 2]
    idx = 0
    for req in traffic:
        for _ in range(req):
            wrr_stats[weights[idx % len(weights)]] += 1
            idx += 1
    wrr_time = sum(wrr_stats[i] * speeds[i] for i in range(3))

    # 3. Least Connections
    lc_stats = [0, 0, 0]
    queues = [0, 0, 0]
    for req in traffic:
        for _ in range(req):
            best_s = np.argmin(queues)
            lc_stats[best_s] += 1
            queues[best_s] += 1
        for s in range(3):
            cleared = int(1 / speeds[s] * 10)
            queues[s] = max(0, queues[s] - cleared)
    lc_time = sum(lc_stats[i] * speeds[i] for i in range(3))
            
    return (rr_stats, rr_time), (wrr_stats, wrr_time), (lc_stats, lc_time)