from darts import TimeSeries
from darts.models import TransformerModel
from darts.metrics import mape, mse
import numpy as np
import pandas as pd
import torch
import time
from sklearn.preprocessing import MinMaxScaler

def prepare_data(file_path, target):
    df = pd.read_csv(file_path)
    df = df.astype("float32")
    df = df.interpolate(method='linear')

    scaler = MinMaxScaler(feature_range=(0, 1))
    df[target] = scaler.fit_transform(df[[target]])

    data = TimeSeries.from_dataframe(df, value_cols=target)

    return data, scaler

def train_and_evaluate(data, forecast_length, scaler):
    validation_cutoff = data.time_index[int(len(data) * 0.8)]
    data_train, data_val = data.split_after(validation_cutoff)

    # Add more settings for CUDA or GPU, MPS is for MacOS
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    model = TransformerModel(
        input_chunk_length=48,
        output_chunk_length=1,
        dropout=0.2,
        num_encoder_layers=3,
        num_decoder_layers=3,
        batch_size=64,
        n_epochs=2,
        optimizer_kwargs={"lr": 1e-3},
        pl_trainer_kwargs={"accelerator": device}
    )

    start_train = time.time()
    model.fit(data_train)
    training_time = time.time() - start_train

    start_pred = time.time()
    pred = model.predict(n=forecast_length)
    inference_time = time.time() - start_pred

    mape_value = mape(data_val, pred)
    error = mse(data_val, pred)
    rmse_value = np.sqrt(error)

    return mape_value, rmse_value, training_time, inference_time

file_name = "1"
file_path = f"datasets/GSATM/data/{file_name}.csv"

compressed_file_name = "PMC_1_3"
compressed_file_path = f"compressed_datasets/{compressed_file_name}.csv"

target = "CO (ppm)"
forecast_length = 1

data, scaler = prepare_data(file_path, target)

print("Evaluating Uncompressed Data:")
mape_uncompressed, rmse_uncompressed, train_time_uncompressed, infer_time_uncompressed = train_and_evaluate(data, forecast_length, scaler)

compressed_data, scaler = prepare_data(compressed_file_path, target)

print("\nEvaluating Compressed Data:")
mape_compressed, rmse_compressed, train_time_compressed, infer_time_compressed = train_and_evaluate(compressed_data, forecast_length, scaler)

print("\nResults:")
print(f"Uncompressed - MAPE: {mape_uncompressed:.4f}%, RMSE: {rmse_uncompressed:.4f}, Training Time: {train_time_uncompressed:.4f}s, Inference Time: {infer_time_uncompressed:.4f}s")
print(f"Compressed   - MAPE: {mape_compressed:.4f}%, RMSE: {rmse_compressed:.4f}, Training Time: {train_time_compressed:.4f}s, Inference Time: {infer_time_compressed:.4f}s")
