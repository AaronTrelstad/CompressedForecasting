import numpy as np
import pywt
import pandas as pd
import os

'''
TODO:
Find a away to handle timestamps
'''

file_name = "20160930_203718"
file_path = f"datasets/GSATM/data/{file_name}.csv"
df = pd.read_csv(file_path)

'''Currently using Daubechies wavelet (DB4) '''
wavelet = 'db4'

'''Decresing levels will lower compresssion ratio'''
level = 2

df_numeric = df.select_dtypes(include=[np.number])
time_series = df_numeric.to_numpy()

coeffs = [pywt.wavedec(time_series[:, i], wavelet, level=level) for i in range(time_series.shape[1])]

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
