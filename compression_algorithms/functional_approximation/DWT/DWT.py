import numpy as np
import pywt
import pandas as pd
import os

file_name = "1"
file_path = f"datasets/GSATM/data/{file_name}.csv"
df = pd.read_csv(file_path)

df_numeric = df.iloc[:, 1:]

time_series = df_numeric.to_numpy()

wavelet = 'db4'
level = 1

threshold = 0.001  

coeffs = [
    pywt.wavedec(time_series[:, i], wavelet, level=level) 
    for i in range(time_series.shape[1])
]

for i in range(len(coeffs)):
    for j in range(1, len(coeffs[i])):
        coeffs[i][j] = np.where(np.abs(coeffs[i][j]) < threshold, 0, coeffs[i][j])

compressed_data = [c[0] for c in coeffs]

compressed_array = np.round(np.stack(compressed_data, axis=1), 4)

results_folder = "compressed_datasets"
os.makedirs(results_folder, exist_ok=True)

compressed_file_path = os.path.join(results_folder, f"DWT_{file_name}_{level}.csv")
compressed_df = pd.DataFrame(compressed_array, columns=df_numeric.columns)

compressed_df.to_csv(compressed_file_path, index=False)

compression_ratio = len(time_series) / len(compressed_df)
print("Original Shape:", time_series.shape)
print("Compressed Shape:", compressed_array.shape)
print(f"Compression Ratio: {compression_ratio}")
