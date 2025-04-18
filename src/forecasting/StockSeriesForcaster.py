import torch
import pandas_ta as ta
import pandas as pd

import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler

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

class StockSeriesForecaster:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = SklearnBatchMinMaxScaler()

    def enrich_tensor(self, close_tensor):
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

            # Fill NaNs (at beginning) to keep tensor shape
            df.fillna(method='bfill', inplace=True)

            enriched.append(df[['close', 'rsi', 'ema_20', 'ema_50']].values)

        enriched_tensor = torch.tensor(np.stack(enriched), dtype=torch.float32).to(close_tensor.device)
        enriched_tensor = self.scaler.scale(enriched_tensor)
        return enriched_tensor

    def enrich_tensor1(self, close_tensor):
        #B, window, _ = close_tensor.shape
        #enriched = []

        #for i in range(B):
        #    close = close_tensor[i, :, 0].cpu().numpy()
        #    df = pd.DataFrame({'close': close})

            # Calculate indicators
        #    df['rsi'] = ta.rsi(df['close'], length=14)
        #    df['ema_20'] = ta.ema(df['close'], length=20)
        #    df['ema_50'] = ta.ema(df['close'], length=50)
            # df['macd'] = ta.macd(df['close'])['MACD_12_26_9']

            # Fill NaNs (at beginning) to keep tensor shape
        #    df.fillna(method='bfill', inplace=True)

        #    enriched.append(df[['close', 'rsi', 'ema_20', 'ema_50']].values)

        #enriched_tensor = torch.tensor(np.stack(enriched), dtype=torch.float32).to(close_tensor.device)
        enriched_tensor = self.scaler.scale(close_tensor)
        return enriched_tensor

    def train(self, train_loader, epochs=10):
        print_every = 50
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader, 1):
                xy = self.enrich_tensor(torch.cat((x, y.unsqueeze(1)), dim=1))
                x = xy[:, :-1, :]  # shape: (B, T, F)
                y = xy[:, -1, :]
                x, y = x.to(self.device), y.to(self.device)
                tgt = torch.zeros_like(x).to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x, tgt, tgt_mask=self.generate_square_subsequent_mask(tgt.size(1)))
                preds = out[:, -1, :]  # Assuming output shape: [B, T, D]
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % print_every == 0:
                    print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

    @torch.no_grad()
    def autoregressive_predict(self, src, max_len):
        self.model.eval()
        src = self.enrich_tensor(src)
        src = src.to(self.device)

        # Start with a zero or special start token sequence [B, 1]
        B = src.size(0)
        generated = torch.zeros_like(src, dtype=src.dtype, device=self.device)

        for t in range(1, max_len + 1):
            tgt_mask = self.generate_square_subsequent_mask(generated.size(1)).to(self.device)

            out = self.model(src, generated, tgt_mask=tgt_mask)
            next_token = out[:, -1, :]

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

        return generated[:, 1:]

    def generate_square_subsequent_mask(self, sz):
        return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)
