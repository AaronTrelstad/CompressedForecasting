import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, SimpleRNN, RepeatVector, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import os

file_name = "1"
file_path = f"datasets/GSATM/data/{file_name}.csv"
df = pd.read_csv(file_path)

df_numeric = df.iloc[:, 1:]
time_series = df_numeric.to_numpy()

scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(df) 

time_steps = 10 
n_features = df.shape[1]
n_samples = len(scaled_data) // time_steps 

scaled_data = scaled_data[:n_samples * time_steps]
reshaped_data = scaled_data.reshape(n_samples, time_steps, n_features)

def rnn_autoencoder(input_shape, encoding_dim=10):
    inputs = Input(shape=input_shape)

    encoded = SimpleRNN(encoding_dim, activation='relu', return_sequences=False)(inputs)
    
    decoded = RepeatVector(input_shape[0])(encoded)
    decoded = SimpleRNN(n_features, activation='sigmoid', return_sequences=True)(decoded)

    autoencoder = Model(inputs, decoded)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

    return autoencoder

encoding_dim = 10 
autoencoder = rnn_autoencoder(input_shape=(time_steps, n_features), encoding_dim=encoding_dim)

autoencoder.fit(reshaped_data, reshaped_data, epochs=10, batch_size=64, verbose=1)

encoder = Model(autoencoder.input, autoencoder.layers[1].output)
compressed_data = encoder.predict(reshaped_data)

compressed_samples = compressed_data.shape[0]
compressed_data_reshaped = np.tile(compressed_data[:compressed_samples], (1, n_features // encoding_dim + 1))[:, :n_features]

results_folder = "compressed_datasets"
os.makedirs(results_folder, exist_ok=True)

compressed_file_path = os.path.join(results_folder, f"RNN_{file_name}.csv")

compressed_df = pd.DataFrame(compressed_data_reshaped, columns=df.columns)
compressed_df.to_csv(compressed_file_path, index=False)

print("Compression Ratio: ", len(time_series) / len(compressed_df))
