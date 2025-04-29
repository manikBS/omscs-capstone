import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

class TimeSeries2DDataset(Dataset):
    def __init__(self, X, y, normalize=True):
        """
        X shape: (N, time_steps, channels)  → reshaped to (N, C, H, W)
        """
        X = torch.tensor(X, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.long)

        if X.ndim == 3:  # (N, 10, 4)
            X = X.permute(0, 2, 1)  # (N, 4, 10)
            X = X.view(-1, 4, 2, 5)  # reshape to (N, 4, 2, 5)

        if normalize:
            mean = X.view(X.size(0), X.size(1), -1).mean(dim=2, keepdim=True).unsqueeze(-1)
            std = X.view(X.size(0), X.size(1), -1).std(dim=2, keepdim=True).unsqueeze(-1) + 1e-8
            X = (X - mean) / std

        self.X = X
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN2DTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 num_classes=4, batch_size=32, lr=1e-3, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.train_loader = DataLoader(
            TimeSeries2DDataset(X_train, y_train, normalize=True),
            batch_size=batch_size, shuffle=True
        )

        self.val_loader = DataLoader(
            TimeSeries2DDataset(X_val, y_val, normalize=True),
            batch_size=batch_size, shuffle=False
        )

        self.train_losses = []
        self.val_losses = []

    def train(self, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            total_samples = 0
            all_preds = []
            all_targets = []

            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)

                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

            avg_loss = total_loss / total_samples
            self.train_losses.append(avg_loss)
            val_loss = self.evaluate(return_loss=True)
            self.val_losses.append(val_loss)

            all_preds_np = torch.cat(all_preds).cpu().numpy()
            all_targets_np = torch.cat(all_targets).cpu().numpy()

            train_acc = accuracy_score(all_targets_np, all_preds_np)
            train_f1 = f1_score(all_targets_np, all_preds_np, average='macro')
            print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Train Acc: {train_acc:.4f} | F1: {train_f1:.4f}")

    def evaluate(self, return_loss=False):
        self.model.eval()
        all_preds = []
        all_targets = []
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for X_batch, y_batch in self.val_loader:
                X_batch, y_batch = X_batch.to(self.device), y_batch.to(self.device)
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                total_loss += loss.item() * X_batch.size(0)
                total_samples += X_batch.size(0)

                preds = torch.argmax(outputs, dim=1)
                all_preds.append(preds.cpu())
                all_targets.append(y_batch.cpu())

        all_preds_np = torch.cat(all_preds).cpu().numpy()
        all_targets_np = torch.cat(all_targets).cpu().numpy()

        avg_val_loss = total_loss / total_samples
        val_acc = accuracy_score(all_targets_np, all_preds_np)
        val_f1 = f1_score(all_targets_np, all_preds_np, average='macro')
        print(f"Validation Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f} | F1:{val_f1:.4f}")

        if return_loss:
            return avg_val_loss
        return val_acc

    def plot_losses(self):
        plt.plot(self.train_losses, label="Train Loss")
        plt.plot(self.val_losses, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training & Validation Loss")
        plt.grid(True)
        plt.legend()
        plt.show()

    def save_model(self,path):
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved to: {path}")