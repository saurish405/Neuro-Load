import simpy
import pandas as pd
import numpy as np
import os
from src.sim_engine import Server
from src.ann_predictor import LoadPredictor
from src.fuzzy_logic import FuzzyController
from src.utils import plot_server_performance

def main():
    env = simpy.Environment()
    predictor = LoadPredictor()
    data_path = 'D:\\LoadBalancer\\Hybrid-Load-Balancer-using-Soft-Computing-Techniques\\data\\traffic_logs.csv'
    model_path = 'models/predictor.keras'
    if not os.path.exists(model_path):
        print("Training Neural Network on Google Borg patterns...")
        predictor.train_from_csv(data_path)
    else:
        print("Using existing Google-trained model.")
    fuzzy = FuzzyController()
    servers = [Server(env, 1, 0.5), Server(env, 2, 0.8), Server(env, 3, 1.2)]
    traffic_df = pd.read_csv(data_path)
    def run_sim(env):
        for i in range(min(120, len(traffic_df))): 
            requests = int(traffic_df.iloc[i]['Requests_per_Minute'])
            history = traffic_df.iloc[max(0, i-60):i]['Requests_per_Minute'].tolist()
            if len(history) < 60: 
                history = [traffic_df['Requests_per_Minute'].mean()] * 60 
            trend = predictor.predict_next(history)
            max_traffic = traffic_df['Requests_per_Minute'].max()
            trend_score = min(100, (trend / max_traffic) * 100)
            for _ in range(requests):
                scores = []
                for s in servers:
                    load = (s.resource.count / s.resource.capacity) * 100
                    scores.append(fuzzy.compute_priority(load, trend_score))
                best_s = servers[np.argmax(scores)]
                env.process(best_s.handle_task(f"Google_Task_{i}_{_}"))
            yield env.timeout(1)
            if i % 10 == 0: 
                print(f" Minute {i} processed - Handling {requests} Google requests...")
    print("Starting Intelligent Load Balancer Simulation (Google Borg Data)...")
    env.process(run_sim(env))
    env.run()
    print("\n--- Final Stats ---")
    for s in servers:
        print(f"Server {s.id} (Speed {s.speed}s): {s.tasks_processed} tasks handled.")
    plot_server_performance(servers)

if __name__ == "__main__":
    main()