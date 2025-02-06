import numpy as np
import pandas as pd
import os

'''
TODO:
Find a away to handle timestamps
'''

file_name = "20160930_203718"
file_path = f"datasets/GSATM/data/{file_name}.csv"
df = pd.read_csv(file_path)

## Test to see different compression rations
window_size = 10 

## fix this but keep timestamps seperate
if 'timestamp' in df.columns:
    timestamps = df['timestamp'].to_numpy()
else:
    timestamps = np.arange(len(df)) 

df_numeric = df.select_dtypes(include=[np.number])
time_series = df_numeric.to_numpy()

compressed_data = []
compressed_timestamps = []

for i in range(0, len(time_series), window_size):
    segment = time_series[i:i + window_size]
    compressed_data.append(np.round(np.mean(segment, axis=0), 4))
    compressed_timestamps.append(timestamps[i]) 

columns = [f"{col}" for col in df_numeric.columns]
compressed_df = pd.DataFrame(compressed_data, columns=columns)

##compressed_df.insert(0, "timestamp", compressed_timestamps)

results_folder = "compressed_datasets"
os.makedirs(results_folder, exist_ok=True)

compressed_file_path = os.path.join(results_folder, f"compressed_PMC_{file_name}_{window_size}.csv")
compressed_df.to_csv(compressed_file_path, index=False)

compression_ratio = len(time_series) / len(compressed_data)

print("Original Shape:", time_series.shape)
print("Compressed Shape:", compressed_df.shape)
print(f"Compression Ratio: {compression_ratio}")
