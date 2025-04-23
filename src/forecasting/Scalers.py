import torch
import numpy as np
import pandas_ta as ta
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler\

def enrich_tensor(close_tensor):
    B, window, _ = close_tensor.shape
    enriched = []

    for i in range(B):
        close = close_tensor[i, :, 0].cpu().numpy()
        df = pd.DataFrame({'close': close})

        # Calculate indicators
        df['rsi'] = ta.rsi(df['close'], length=14)
        df['ema_20'] = ta.ema(df['close'], length=20)
        df['ema_50'] = ta.ema(df['close'], length=50)
        # df['macd'] = ta.macd(df['close'])['MACD_12_26_9']

        df.fillna(method='bfill', inplace=True)

        enriched.append(df[['close', 'rsi', 'ema_20', 'ema_50']].values)

    enriched_tensor = torch.tensor(np.stack(enriched), dtype=torch.float32).to(close_tensor.device)
    return enriched_tensor

class SklearnBatchMinMaxScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.scalers = []

    def scale(self, batch_tensor):
        B, T, F = batch_tensor.shape
        batch_np = batch_tensor.cpu().numpy()
        scaled_batches = []
        self.scalers = []

        for i in range(B):
            scaler = MinMaxScaler(feature_range=self.feature_range)
            scaled = scaler.fit_transform(batch_np[i])  # shape: (T, F)
            scaled_batches.append(scaled)
            self.scalers.append(scaler)

        scaled_tensor = torch.tensor(np.stack(scaled_batches), dtype=torch.float32).to(batch_tensor.device)
        return scaled_tensor

    def inverse_scale(self, scaled_tensor):
        B, T, F = scaled_tensor.shape
        scaled_np = scaled_tensor.cpu().numpy()
        restored_batches = []

        for i in range(B):
            scaler = self.scalers[i]
            restored = scaler.inverse_transform(scaled_np[i])  # shape: (T, F)
            restored_batches.append(restored)

        restored_tensor = torch.tensor(np.stack(restored_batches), dtype=torch.float32).to(scaled_tensor.device)
        return restored_tensor

class SklearnBatchStandardScaler:
    def __init__(self, feature_range=(-1, 1)):
        self.feature_range = feature_range
        self.scalers = []

    def scale(self, batch_tensor):
        B, T, F = batch_tensor.shape
        batch_np = batch_tensor.cpu().numpy()
        scaled_batches = []
        self.scalers = []

        for i in range(B):
            scaler = StandardScaler()
            scaled = scaler.fit_transform(batch_np[i])  # shape: (T, F)
            scaled_batches.append(scaled)
            self.scalers.append(scaler)

        scaled_tensor = torch.tensor(np.stack(scaled_batches), dtype=torch.float32).to(batch_tensor.device)
        return scaled_tensor

    def scale_withoutfit(self, batch_tensor):
        B, T, F = batch_tensor.shape
        batch_np = batch_tensor.cpu().numpy()
        scaled_batches = []

        for i in range(B):
            scaler = self.scalers[i]
            scaled = scaler.transform(batch_np[i])  # shape: (T, F)
            scaled_batches.append(scaled)

        scaled_tensor = torch.tensor(np.stack(scaled_batches), dtype=torch.float32).to(batch_tensor.device)
        return scaled_tensor

    def inverse_scale(self, scaled_tensor):
        B, T, F = scaled_tensor.shape
        scaled_np = scaled_tensor.cpu().numpy()
        restored_batches = []

        for i in range(B):
            scaler = self.scalers[i]
            restored = scaler.inverse_transform(scaled_np[i])  # shape: (T, F)
            restored_batches.append(restored)

        restored_tensor = torch.tensor(np.stack(restored_batches), dtype=torch.float32).to(scaled_tensor.device)
        return restored_tensor