import numpy as np
import pandas as pd
import os

file_name = "1"
file_path = f"datasets/GSATM/data/{file_name}.csv"
df = pd.read_csv(file_path)

df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors='coerce')

df_numeric = df.select_dtypes(include=[np.number])

time_series = df_numeric.to_numpy()

# Change this to change compression ratio
window_size = 3

compressed_data = [
    np.round(np.mean(time_series[i:i + window_size], axis=0), 4)
    for i in range(0, len(time_series), window_size)
]

compressed_df = pd.DataFrame(compressed_data, columns=df_numeric.columns)

results_folder = "compressed_datasets"
os.makedirs(results_folder, exist_ok=True)

compressed_file_path = os.path.join(results_folder, f"PMC_{file_name}_{window_size}.csv")
compressed_df.to_csv(compressed_file_path, index=False)

compression_ratio = len(time_series) / len(compressed_data)

print("Original Shape:", time_series.shape)
print("Compressed Shape:", compressed_df.shape)
print(f"Compression Ratio: {compression_ratio:.2f}")
