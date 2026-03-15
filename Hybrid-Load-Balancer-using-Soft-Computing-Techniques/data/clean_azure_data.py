import pandas as pd
import numpy as np
RAW_AZURE_FILE = 'azure_raw.csv' 
OUTPUT_FILE = 'data/real_azure_traffic.csv'

print("Loading a chunk of the massive Azure dataset...")

try:
    df = pd.read_csv(RAW_AZURE_FILE, usecols=[0, 4], names=['Timestamp', 'AvgCPU'], nrows=1000000, header=None)
    
    print("✅ Raw data loaded! Cleaning and converting to traffic spikes...")

   
    grouped_traffic = df.groupby('Timestamp')['AvgCPU'].sum().reset_index()

   
    max_cpu = grouped_traffic['AvgCPU'].max()
    grouped_traffic['Requests_per_Minute'] = (grouped_traffic['AvgCPU'] / max_cpu) * 500
    grouped_traffic['Requests_per_Minute'] = grouped_traffic['Requests_per_Minute'].astype(int)

    # 5. Drop the old CPU column and keep it clean
    final_df = grouped_traffic[['Timestamp', 'Requests_per_Minute']]

    # 6. Save the new, lightweight file!
    final_df.to_csv(OUTPUT_FILE, index=False)
    
    print(f"MASSIVE W! Successfully compressed Azure data and saved to {OUTPUT_FILE}")
    print(f"Total time steps generated: {len(final_df)}")

except FileNotFoundError:
    print(f" ERROR: Could not find '{RAW_AZURE_FILE}'. Make sure you extracted it into this folder!")