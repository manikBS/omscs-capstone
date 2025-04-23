import torch
from src.forecasting.Scalers import SklearnBatchStandardScaler, enrich_tensor

class StockSeriesForecasterEncoder:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scaler = SklearnBatchStandardScaler()

    def train(self, train_loader, val_loader=None, epochs=10):
        print_every = 50
        train_losses = []
        val_losses = []
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for batch_idx, (x, y) in enumerate(train_loader, 1):
                xy = enrich_tensor(torch.cat((x, y.unsqueeze(1)), dim=1))
                xy = self.scaler.scale(xy)
                x = xy[:, :-1, :]  # shape: (B, T, F)
                y = xy[:, -1, :]
                x, y = x.to(self.device), y.to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x)
                loss = self.criterion(out, y[:, 0].unsqueeze(1))
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                if batch_idx % print_every == 0:
                    print(f"[Epoch {epoch} | Batch {batch_idx}] Loss: {loss.item():.4f}")

            avg_loss = total_loss / len(train_loader)
            train_losses.append(avg_loss)
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")

            if val_loader is not None:
                self.model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for x_val, y_val in val_loader:
                        xy_val = enrich_tensor(torch.cat((x_val, y_val.unsqueeze(1)), dim=1))
                        xy_val = self.scaler.scale(xy_val)
                        x_val = xy_val[:, :-1, :].to(self.device)
                        y_val = xy_val[:, -1, :].to(self.device)

                        out_val = self.model(x_val)
                        loss_val = self.criterion(out_val, y_val[:, 0].unsqueeze(1))
                        val_loss += loss_val.item()

                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                print(f"[Epoch {epoch}] Validation Loss: {avg_val_loss:.4f}")
        return train_losses, val_losses

    @torch.no_grad()
    def autoregressive_predict(self, src, max_len):
        self.model.eval()
        src = enrich_tensor(src)
        src = self.scaler.scale(src)
        src = src.to(self.device)

        predictions = []

        for t in range(1, max_len + 1):
            out = self.model(src)
            out_expanded = torch.zeros((src.shape[0], 1, src.shape[2]))
            out_expanded[:, :, 0] = out

            next_seq = torch.cat([src, out_expanded], dim=1)

            unscaled_next_seq = self.scaler.inverse_scale(next_seq)
            predictions.append(unscaled_next_seq[:, src.shape[1]:, 0].detach().cpu())
            unscaled_next_seq = enrich_tensor(unscaled_next_seq[:, :, 0].unsqueeze(-1))

            scaled_next_seq = self.scaler.scale_withoutfit(unscaled_next_seq)

            src = scaled_next_seq

        return torch.cat(predictions, dim=1)
