import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import pandas_ta as ta

class Nifty50Dataset(Dataset):
    def __init__(self, csv_path, window_size=60, prediction_horizon=1, transform=None):
        self.data = pd.read_csv(csv_path, parse_dates=['date'])
        self.data = self.data[500000:600000]

        features = ['close']
        self.data = self.data[features]

        #self.data['rsi'] = ta.rsi(self.data['close'], length=14)
        #self.data['ema_20'] = ta.ema(self.data['close'], length=20)
        #self.data['ema_50'] = ta.ema(self.data['close'], length=50)
        #self.data['macd'] = ta.macd(self.data['close'])['MACD_12_26_9']
        #self.data['macd_signal'] = ta.macd(self.data['close'])['MACDs_12_26_9']
        #self.data['macd_hist'] = ta.macd(self.data['close'])['MACDh_12_26_9']

        self.data.bfill(inplace=True)

        self.feature_columns = self.data.columns.tolist()

        #self.data = (self.data - self.data.mean()) / (self.data.std() + 1e-6)
        self.data = self.data.values

        self.window_size = window_size
        self.prediction_horizon = prediction_horizon
        self.transform = transform

    def __len__(self):
        return len(self.data) - self.window_size - self.prediction_horizon

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.window_size]

        next_close = self.data[idx + self.window_size + self.prediction_horizon - 1]#[
            #self.feature_columns.index('close')]

        return torch.tensor(x, dtype=torch.float32), torch.tensor(next_close, dtype=torch.float32)


#dataset = Nifty50Dataset("/Users/maniksomayaji/Documents/omscs/capstone_project/data/NIFTY 50_minute_data.csv", window_size=3, prediction_horizon=1)

#loader = DataLoader(dataset, batch_size=2, shuffle=False)

#for batch_x, batch_y in loader:
#    print("Batch x:", batch_x)
#    print("Batch y:", batch_y)
#    print("---")