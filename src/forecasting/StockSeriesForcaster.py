import torch

class StockSeriesForecaster:
    def __init__(self, model, optimizer, criterion, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def train(self, train_loader, epochs=10):
        for epoch in range(1, epochs + 1):
            self.model.train()
            total_loss = 0.0

            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                tgt = torch.zeros_like(x).to(self.device)

                self.optimizer.zero_grad()
                out = self.model(x, tgt)
                preds = out[:, -1, 0]  # Assuming output shape: [B, T, D]
                loss = self.criterion(preds, y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"[Epoch {epoch}] Train Loss: {avg_loss:.4f}")
