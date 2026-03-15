import matplotlib.pyplot as plt
import numpy as np
import os

def plot_server_performance(server_data):
    # 1. Ensure the folder exists
    if not os.path.exists('plots'):
        os.makedirs('plots')

    ids = [f"Server {s.id}\n(Speed {s.speed})" for s in server_data]
    tasks = [s.tasks_processed for s in server_data]

    # Use a clear style
    plt.style.use('ggplot') 
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw the bars
    bars = ax.bar(ids, tasks, color=['#4CAF50', '#2196F3', '#FF9800'])
    
    ax.set_xlabel('Server Configuration')
    ax.set_ylabel('Total Tasks Processed')
    ax.set_title('Intelligent Load Balancer: Google Borg Workload Distribution')
    
    # Add labels on top of bars
    for bar in bars:
        yval = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, yval + 500, int(yval), 
                ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    
    # 2. Save BEFORE showing
    plt.savefig('plots/performance_results.png')
    print("DONE: Plot saved to plots/performance_results.png")
    
    # 3. THE FIX: Force the window to stay active and draw
    print(" Opening graph window... Please wait a second for it to load.")
    plt.show(block=True)