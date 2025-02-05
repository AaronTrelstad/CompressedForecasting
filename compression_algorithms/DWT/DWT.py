import numpy as np
import pywt
import pandas as pd
import os

file_name = "20160930_203718"
file_path = f"datasets/GSATM/data/{file_name}.csv"
df = pd.read_csv(file_path)

df_numeric = df.select_dtypes(include=[np.number])

time_series = df_numeric.to_numpy()

wavelet = 'db4'
level = 3

coeffs = [pywt.wavedec(time_series[:, i], wavelet, level=level) for i in range(time_series.shape[1])]

compressed_data = [c[0] for c in coeffs]

compressed_array = np.stack(compressed_data, axis=1)

results_folder = "compressed_datasets"
os.makedirs(results_folder, exist_ok=True) 

compressed_file_path = os.path.join(results_folder, f"compressed_DWT_{file_name}.csv")
compressed_df = pd.DataFrame(compressed_array, columns=df_numeric.columns)
compressed_df.to_csv(compressed_file_path, index=False)

print("Original Shape:", time_series.shape)
print("Compressed Shape:", compressed_array.shape)
