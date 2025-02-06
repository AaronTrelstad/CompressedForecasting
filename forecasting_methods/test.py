from darts import TimeSeries
from darts.models import RNNModel, LinearRegressionModel
import pandas as pd

file_name = "20160930_203718"
file_path = f"datasets/GSATM/data/{file_name}.csv"
df = pd.read_csv(file_path)

data = TimeSeries.from_dataframe(df)

print(data)
