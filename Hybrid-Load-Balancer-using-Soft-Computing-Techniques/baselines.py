import pandas as pd
import numpy as np
import os
import ast

def _load_traffic(source, total_minutes=None):
    """
    Accepts either a file path or a DataFrame. 
    Handles Google Cluster Trace 'average_usage' extraction.
    """
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    elif isinstance(source, str):
        if not os.path.exists(source):
            raise FileNotFoundError(f"CSV not found: {source}")
        # We take .head(1000) to match the main2.py simulation scale
        df = pd.read_csv(source).head(1000)
    else:
        raise TypeError("source must be a file path (str) or a pandas DataFrame.")

    df.columns = df.columns.str.strip()

    # --- GOOGLE CLUSTER DATA PARSING ---
    if 'average_usage' in df.columns:
        print("Baselines: Extracting CPU from Google average_usage column...")
        def extract_cpu(val):
            try:
                # Converts "{'cpus': 0.x, ...}" string to float
                d = ast.literal_eval(str(val))
                return float(d.get('cpus', 0))
            except:
                return 0.0
        raw = df['average_usage'].apply(extract_cpu).values
    
    # --- FALLBACK FOR TRADITIONAL DATASETS ---
    elif 'request_count' in df.columns:
        raw = df['request_count'].values
    elif 'requests' in df.columns:
        raw = df['requests'].values
    else:
        available = list(df.columns)
        raise ValueError(f"No valid traffic column found. Available: {available}")

    # Limit minutes if requested
    if total_minutes:
        raw = raw[:total_minutes]

    # Normalize/Scale logic to match main2.py (SCALE = 1500)
    if raw.max() <= 1.0:
        scale = 1500
        traffic = np.maximum(1, (raw * scale).astype(int))
    else:
        traffic = raw.astype(int)

    return traffic


def get_traditional_results(source, total_minutes=None):
    """
    Simulates Round-Robin, Weighted Round-Robin, and Least-Connections.
    Returns: (rr_stats, rr_time), (wrr_stats, wrr_time), (lc_stats, lc_time)
    """
    traffic = _load_traffic(source, total_minutes)
    
    # Server configuration — must match main2.py and summary_stats.py
    # Speeds: S1=0.5, S2=1.0, S3=1.5
    speeds = [0.5, 1.0, 1.5]
    proc_times = [1.0 / s for s in speeds]  # [2.0s, 1.0s, 0.667s] per task

    # 1. Round Robin (Simple rotation)
    rr_stats = [0, 0, 0]
    global_idx = 0
    for req_count in traffic:
        for _ in range(req_count):
            rr_stats[global_idx % 3] += 1
            global_idx += 1
    rr_time = sum(rr_stats[i] * proc_times[i] for i in range(3))

    # 2. Weighted Round Robin (Priority to faster servers)
    # Ratio 1:2:3 for S1:S2:S3
    w_pattern = [0, 1, 1, 2, 2, 2] 
    wrr_stats = [0, 0, 0]
    w_idx = 0
    for req_count in traffic:
        for _ in range(req_count):
            wrr_stats[w_pattern[w_idx % len(w_pattern)]] += 1
            w_idx += 1
    wrr_time = sum(wrr_stats[i] * proc_times[i] for i in range(3))

    # 3. Least Connections (Send to server with smallest queue)
    lc_stats = [0, 0, 0]
    queues = [0, 0, 0]
    for req_count in traffic:
        for _ in range(req_count):
            best_server = int(np.argmin(queues))
            lc_stats[best_server] += 1
            queues[best_server] += 1
        
        # Drain queues at the end of every minute based on server speed
        for s in range(3):
            drain_rate = int(speeds[s] * 60) # Tasks processed per minute
            queues[s] = max(0, queues[s] - drain_rate)
            
    lc_time = sum(lc_stats[i] * proc_times[i] for i in range(3))

    return (rr_stats, rr_time), (wrr_stats, wrr_time), (lc_stats, lc_time)
