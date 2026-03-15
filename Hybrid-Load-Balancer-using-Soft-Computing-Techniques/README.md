# 🚀 Intelligent Hybrid Load Balancer: ANN-Fuzzy-GA Approach

This project implements a proactive, intelligent load balancing system designed for heterogeneous server environments. 
By hybridizing **Artificial Neural Networks (ANN)**, **Fuzzy Logic**, and **Genetic Algorithms (GA)**, the system anticipates traffic spikes and optimizes workload distribution to minimize latency and prevent server bottlenecks.

---

## 📈 Executive Summary of Findings
The primary achievement of this project is the transition from **reactive** to **proactive** load balancing. 

* **Efficiency Gain:** Our Hybrid AI model achieved a **32.9% reduction in processing time** during peak stress compared to standard Round Robin algorithms.
* **Intelligent Prioritization:** The system successfully identified hardware heterogeneity, funneling the vast majority of tasks to high-performance nodes while maintaining "safety valves" on slower nodes to prevent system failure.

---

## 🧠 Methodology & Soft Computing Techniques

The system utilizes a three-layer "Soft Computing" brain to manage traffic:

1.  **Artificial Neural Network (ANN) - The Forecaster:**
    Analyzes historical time-series traffic logs to identify patterns. It predicts incoming request volumes, allowing the system to prepare for surges *before* they hit the servers.
2.  **Fuzzy Logic - The Decision Maker:**
    Uses linguistic variables (e.g., "Low Load," "Critical Spike") to handle the uncertainty of network traffic. It evaluates the ANN prediction alongside real-time server health to calculate an optimal "Priority Score" for routing.
3.  **Genetic Algorithm (GA) - The Optimizer:**
    Acts as the evolutionary layer. It continuously tunes the Fuzzy membership functions and ANN weights to ensure the system remains optimized as traffic patterns evolve over time.



---

## 🖥️ Server Configuration
The simulation environment consists of three heterogeneous servers with varying processing capacities. "Speed" represents the time (in seconds) required to handle a single task.

| Server | Role | Processing Speed (Sec/Task) |
| :--- | :--- | :--- |
| **Server 1** | Primary (High Performance) | 0.5s |
| **Server 2** | Secondary (Mid-Range) | 0.8s |
| **Server 3** | Backup (Low Performance) | 1.2s |

---

## 📊 Phase 1: Normal Operating Conditions
**Dataset:** `traffic_logs.csv` (Avg. 50 requests/min)

In standard conditions, the AI maximizes the efficiency of Server 1.

### Workload Distribution (Tasks Handled)
* **Server 1:** 4,706 tasks
* **Server 2:** 1,565 tasks
* **Server 3:** 927 tasks

### Performance & Efficiency Comparison
| Method | Total Processing Time (Sec) | Efficiency Gain vs. RR |
| :--- | :--- | :--- |
| Round Robin | ~6,000.00 | Baseline |
| Weighted RR | ~5,100.00 | 15.0% |
| Least Connections | ~5,800.00 | 3.3% |
| **Hybrid AI (Ours)** | **4,717.40** | **21.4% Improvement** |



---

## ⚡ Phase 2: Stressed Conditions (Flash Crowd)
**Dataset:** `stress_traffic.csv` (Sudden surge to 500 requests/min)

During a 500% traffic spike, the ANN's predictive layer allows the system to brace for the load.

### Workload Distribution (Tasks Handled)
* **Server 1:** 16,845 tasks
* **Server 2:** 1,532 tasks
* **Server 3:** 981 tasks

### Performance & Efficiency Comparison
| Method | Total Processing Time (Sec) | Efficiency Gain vs. RR |
| :--- | :--- | :--- |
| Round Robin | 16,133.40 | Baseline |
| Least Connections | 15,641.80 | 3.0% |
| Weighted RR | 13,872.80 | 14.0% |
| **Hybrid AI (Ours)** | **10,825.30** | **32.9% Improvement** |



> **Note:** All generated visual reports, including workload distribution bars and comparison charts, can be found in the `/plots` directory of this repository.

---

## 🛠️ Detailed Installation & Usage Guide

Follow these steps to run the simulation on your local machine:

### 1. Prerequisites
Ensure you have Python 3.10+ installed. Clone the repository and install dependencies:
```bash
pip install -r requirements.txt
2. Prepare the Datasets
You must generate the traffic data before running the simulation:

Run python generate_data.py for normal traffic.

Run python generate_stress.py for the flash crowd scenario.

3. Running the Intelligent Simulation
To test different conditions, you must manually point the simulation to the correct CSV file:

Open main.py.

Locate the line: pd.read_csv('data/traffic_logs.csv').

Change it to 'data/stress_traffic.csv' to run the Stress Test.

Execute: python main.py.

Important: After the simulation ends, copy the "Final Stats" (tasks handled by each server) from your terminal.

4. Visualizing Comparative Results
To generate the comparison graphs (like master_comparison.png):

Open visualize_all.py.

Update the ai_results list with the data you copied from your terminal in the previous step:
ai_results = [Server1_Tasks, Server2_Tasks, Server3_Tasks]

Update the csv_path inside visualize_all.py to match the dataset you used (normal or stress).

Execute: python visualize_all.py.

5. Checking Stats
Run python summary_stats.py to see the mathematical breakdown of time saved and efficiency percentages.
