import numpy as np
import pandas as pd
from scipy.stats import entropy

def calculate_average_variance(df):
    feature_variances = df.var()
    average_variance = feature_variances.mean()
    
    return average_variance

def calculate_average_entropy(df, num_bins=10):
    entropy_values = []
    
    for column in df.columns:
        data_binned = np.digitize(df[column], bins=np.linspace(df[column].min(), df[column].max(), num_bins))
        
        probabilities = np.bincount(data_binned) / len(df[column])
        
        feature_entropy = entropy(probabilities)
        entropy_values.append(feature_entropy)
    
    average_entropy = np.mean(entropy_values)
    
    return average_entropy

file_name = "1"
file_path = f"datasets/GSATM/data/{file_name}.csv"

df = pd.read_csv(file_path)
df = df.drop(columns=['Time (s)'])

avg_variance = calculate_average_variance(df)
print(f"Average Variance: {avg_variance}")

avg_entropy = calculate_average_entropy(df)
print(f"Average Entropy: {avg_entropy}")

compressed_file_name = "PMC_1_3"
compressed_file_path = f"compressed_datasets/{compressed_file_name}.csv"

df = pd.read_csv(compressed_file_path)
df = df.drop(columns=['Time (s)'])

avg_variance = calculate_average_variance(df)
print(f"Average Variance: {avg_variance}")

avg_entropy = calculate_average_entropy(df)
print(f"Average Entropy: {avg_entropy}")
