import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data_prep.Nifty50Dataset import Nifty50Dataset
from src.forecasting.GaussianTransformer import CustomTransformer
from src.forecasting.StockSeriesForcaster import StockSeriesForecaster

model = CustomTransformer(d_model=4, nhead=2)
dataset = Nifty50Dataset("/Users/maniksomayaji/Documents/omscs/capstone_project/data/NIFTY 50_minute_data.csv", window_size=300, prediction_horizon=1)
loader = DataLoader(dataset, batch_size=2, shuffle=False)

criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

forecaster = StockSeriesForecaster(model, optimizer, criterion)
forecaster.train(loader)

num_batches = 3
for i, (x, y) in enumerate(loader):
    if i >= num_batches:
        break
    output = forecaster.autoregressive_predict(x, max_len=20)
    print("Generated:", output)

