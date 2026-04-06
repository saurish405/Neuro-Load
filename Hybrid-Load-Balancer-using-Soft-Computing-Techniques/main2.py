import simpy
import numpy as np
import pandas as pd
import os
import ast
from src.ann_predictor import LoadPredictor
from src.fuzzy_logic import FuzzyController
from src.sim_engine import Server
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def extract_cpu_value(val):
    """Parses the {'cpus': 0.x} string from Google Cluster Data."""
    try:
        d = ast.literal_eval(str(val))
        return float(d.get('cpus', 0))
    except:
        return 0.0

def main():
    print("\n--- NEO-VITA FINAL RUN: GOOGLE CLUSTER ---")

    # ── STEP 1: DATA PREPARATION ─────────────────────────────────────────────
    file_path = 'data/google_data.csv'
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    print("Step 1: Processing Google Cluster logs and handling outliers...")
    df = pd.read_csv(file_path).head(1000)
    df.columns = df.columns.str.strip()

    raw_values = df['average_usage'].apply(extract_cpu_value).values
    limit = np.percentile(raw_values, 95)
    clipped = np.clip(raw_values, 0, limit)
    
    scaler = MinMaxScaler()
    traffic_df = pd.DataFrame()
    traffic_df['request_count'] = scaler.fit_transform(clipped.reshape(-1, 1)).flatten()

    # ── STEP 2: ANN TRAINING / LOADING ───────────────────────────────────────
    print("\nStep 2: Training / Loading ANN...")
    predictor = LoadPredictor()
    model_path = 'models/predictor_v2.keras'

    if not os.path.exists(model_path):
        predictor.train_from_df(traffic_df)
    else:
        print("Loading pre-trained model...")
        predictor.load_model(model_path)

    # ── STEP 3: SIMULATION ───────────────────────────────────────────────────
    env   = simpy.Environment()
    fuzzy = FuzzyController()

    servers = [
        Server(env, id=1, speed=0.5, capacity=30),
        Server(env, id=2, speed=1.0, capacity=60),
        Server(env, id=3, speed=1.5, capacity=90),
    ]

    SCALE = 1200 
    simulation_minutes = len(traffic_df)
    minute_logs = []

    def run_simulation(env):
        print(f"\nStep 3: Running simulation for {simulation_minutes} minutes...")

        for i in range(simulation_minutes):
            current_vol = max(1, int(traffic_df.iloc[i]['request_count'] * SCALE))

            if i >= 60:
                history = traffic_df.iloc[i - 60:i]['request_count'].tolist()
            else:
                history = [traffic_df.iloc[i]['request_count']] * 60

            trend_score = predictor.compute_trend_score(history)

            for _ in range(current_vol):
                scores = []
                for s in servers:
                    current_load = (s.resource.count / s.resource.capacity) * 100.0
                    score = fuzzy.compute_priority(current_load, trend_score, s.speed)
                    scores.append(score)

                best_s = servers[np.argmax(scores)]
                env.process(best_s.handle_task(f"Google_{i}_{_}"))

            yield env.timeout(1)

            if i % 100 == 0:
                print(f"  Minute {i:4} | Vol: {current_vol:4} | Trend: {trend_score:5.1f}%")

            minute_logs.append({
                'minute': i,
                'volume': current_vol,
                'trend_score': trend_score,
            })

    env.process(run_simulation(env))
    env.run()

    # ── STEP 4: PLOTTING (SEPARATE WINDOWS) ─────────────────────────────────
    print("\n--- Final Results ---")
    total_tasks = sum(s.tasks_processed for s in servers)
    ids = [f"S{s.id}\n(speed={s.speed})" for s in servers]
    tasks = [int(s.tasks_processed) for s in servers]
    logs_df = pd.DataFrame(minute_logs)

    # Window 1: Workload Distribution
    plt.figure(num="Workload Distribution", figsize=(8, 6))
    bars = plt.bar(ids, tasks, color=['#ff7675', '#74b9ff', '#55efc4'])
    plt.title('Workload Distribution — Neuro-Fuzzy Balancer', fontweight='bold')
    plt.ylabel('Total Tasks Processed')
    for i, v in enumerate(tasks):
        plt.text(i, v + (max(tasks) * 0.02), f'{v:,}', ha='center', fontweight='bold')
    plt.tight_layout()

    # Window 2: Traffic vs ANN Trend
    plt.figure(num="Traffic & Trend Analysis", figsize=(10, 6))
    ax = plt.gca()
    ax2b = ax.twinx()
    
    line1 = ax.plot(logs_df['minute'], logs_df['volume'], color='blue', alpha=0.5, label='Traffic Volume')
    ax.set_ylabel('Request Volume', color='blue')
    ax.set_xlabel('Simulation Minute')
    
    line2 = ax2b.plot(logs_df['minute'], logs_df['trend_score'], color='red', alpha=0.7, label='ANN Trend')
    ax2b.set_ylabel('Trend Score (%)', color='red')
    ax2b.set_ylim(0, 110)
    
    plt.title('Traffic Volume & ANN Trend Score Over Time', fontweight='bold')
    
    lns = line1 + line2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc='upper left')
    plt.tight_layout()

    # ── STEP 5: UPDATE SUMMARY STATS ─────────────────────────────────────────
    summary_file = 'data/summary_stats.csv'
    if not os.path.exists('data'): os.makedirs('data')

    run_results = {
        'Timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
        'Total_Tasks': total_tasks,
        'S1_Tasks': tasks[0],
        'S2_Tasks': tasks[1],
        'S3_Tasks': tasks[2],
        'S1_Percent': round((tasks[0] / total_tasks * 100), 2),
        'S2_Percent': round((tasks[1] / total_tasks * 100), 2),
        'S3_Percent': round((tasks[2] / total_tasks * 100), 2)
    }

    stats_df = pd.DataFrame([run_results])
    if not os.path.exists(summary_file):
        stats_df.to_csv(summary_file, index=False)
    else:
        stats_df.to_csv(summary_file, mode='a', header=False, index=False)

    # ── STEP 6: OUTPUT FOR SUMMARYSTATS.PY ──────────────────────────────────
    print("\n" + "="*40)
    print("COPY THESE VALUES INTO summarystats.py:")
    print(f"AI_TASKS = {tasks}")
    print(f"SPEEDS   = [0.5, 1.0, 1.5]")
    print("="*40)

    # This opens both windows at the same time
    plt.show()

if __name__ == "__main__":
    main()
