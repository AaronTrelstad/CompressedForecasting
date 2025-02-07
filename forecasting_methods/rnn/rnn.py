from darts import TimeSeries
from darts.models import RNNModel
from darts.metrics import mse
import numpy as np
import pandas as pd
import torch
import time

def calculate_mape(actual, predicted, forecast_length):
    actual_subset = actual[:forecast_length]
    return np.mean(np.abs((actual_subset - predicted) / actual_subset)) * 100

def prepare_data(file_path, target):
    df = pd.read_csv(file_path)

    ## Dataset has irregular timestamps so for now im dropping the column
    df = df.drop(columns=['Time (s)'])

    df = df.astype("float32")

    df = df.interpolate(method='linear')

    data = TimeSeries.from_dataframe(df, value_cols=target)

    return data

def train_and_evaluate(data, forecast_length):
    validation_cutoff = data.time_index[int(len(data) * 0.8)]
    data_train, data_val = data.split_after(validation_cutoff)

    torch.set_default_dtype(torch.float32)

    '''
    Need to look into configuring the parametes
    '''
    model = RNNModel(
        model="LSTM",
        input_chunk_length=24,
        output_chunk_length=1,
        hidden_dim=25,
        n_rnn_layers=2,
        batch_size=32,
        n_epochs=2,
        optimizer_kwargs={"lr": 1e-3},
        random_state=42,
    )

    start_train = time.time()
    model.fit(data_train)
    training_time = time.time() - start_train
    print(f"Training Time: {training_time:.4f} seconds")

    start_pred = time.time()
    pred = model.predict(n=forecast_length)
    inference_time = time.time() - start_pred
    print(f"Inference Time: {inference_time:.4f} seconds")

    actual_values = data_val.values()
    predicted_values = pred.values()
    mape = calculate_mape(actual_values, predicted_values, forecast_length)
    print(f"Mean Absolute Percentage Error: {mape:.4f}%")

    error = mse(data_val, pred)
    rmse_value = np.sqrt(error) 
    print(f"Root Mean Squared Error: {rmse_value:.4f}")

file_name = "20160930_203718"
file_path = f"datasets/GSATM/data/{file_name}.csv"

##file_name = "test"
##file_path = f"compressed_datasets/{file_name}.csv"

target = "Flow rate (mL/min)"
forecast_length = 4

data = prepare_data(file_path, target)
train_and_evaluate(data, forecast_length)
