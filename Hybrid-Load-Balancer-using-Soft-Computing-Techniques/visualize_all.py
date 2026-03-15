import matplotlib.pyplot as plt
import numpy as np
from baselines import get_traditional_results

# PASTE YOUR AI RESULTS HERE FROM YOUR TERMINAL RUN
ai_results = [16845, 1532, 981] 

# Get traditional results
# 1. Get the results (Note how we unpack the tuples now)
(rr_data, rr_time), (wrr_data, wrr_time), (lc_data, lc_time) = get_traditional_results('data/stress_traffic.csv')

# 2. Update the plotting section to use the _data variables
def plot_master_comparison():
    labels = ['Server 1 (Fast)', 'Server 2 (Med)', 'Server 3 (Slow)']
    x = np.arange(len(labels))
    width = 0.2

    plt.figure(figsize=(12, 7))
    # Use rr_data, wrr_data, and lc_data here!
    plt.bar(x - 1.5*width, rr_data, width, label='Round Robin', color='#bdc3c7')
    plt.bar(x - 0.5*width, wrr_data, width, label='Weighted RR', color='#95a5a6')
    plt.bar(x + 0.5*width, lc_data, width, label='Least Conn', color='#34495e')
    plt.bar(x + 1.5*width, ai_results, width, label='Hybrid AI (Ours)', color='#2ecc71')

    plt.ylabel('Tasks Processed')
    plt.title('Stress Test Comparison: AI vs Traditional Balancers')
    plt.xticks(x, labels)
    plt.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    plt.savefig('plots/stress_comparison.png')
    plt.show()

if __name__ == "__main__":
    plot_master_comparison()