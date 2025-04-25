import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

class TimeSeries1DDataset(Dataset):
    def __init__(self, X, y, normalize=True):
        """
        X shape: (N, time_steps, channels) or (N, channels, time_steps)
        y shape: (N,)
        """

        X = torch.tensor(X, dtype=torch.float32)  # e.g., (N, 15, 4) or (N, 4, 15)

        # Ensure shape is (N, C, T) → needed by Conv1D
        if X.ndim == 3 and X.shape[2] < X.shape[1]:  # shape: (N, 15, 4)
            X = X.permute(0, 2, 1)  # → (N, 4, 15)

        print(X.shape)

        if normalize:
            # Normalize per window, per channel
            mean = X.mean(dim=2, keepdim=True)
            std = X.std(dim=2, keepdim=True) + 1e-8
            X = (X - mean) / std

        self.X = X
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class CNN1DTrainer:
    def __init__(self, model, X_train, y_train, X_val, y_val,
                 num_classes=4, batch_size=32, lr=1e-3, device=None):
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = model.to(self.device)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.train_losses = []
        self.val_losses = []
        self.train_loader = DataLoader(
            TimeSeries1DDataset(X_train, y_train, normalize=True),
            batch_size=batch_size,
            shuffle=True
        )

        self.val_loader = DataLoader(
            TimeSeries1DDataset(X_val, y_val, normalize=True),
            batch_size=batch_size,
            shuffle=False
        )

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
            train_acc = accuracy_score(torch.cat(all_targets), torch.cat(all_preds))

            val_loss = self.evaluate(return_loss=True)
            self.train_losses.append(avg_loss)
            self.val_losses.append(val_loss)

            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}, Accuracy: {train_acc:.4f}")

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

        avg_val_loss = total_loss / total_samples
        val_acc = accuracy_score(torch.cat(all_targets), torch.cat(all_preds))
        print(f"Validation Loss: {avg_val_loss:.4f} | Accuracy: {val_acc:.4f}")

        if return_loss:
            return avg_val_loss
        return val_acc

    def predict(self, X_input):
        """
        X_input: (N, C, T) or (N, T, C) → auto-permuted and normalized
        """
        self.model.eval()
        X_tensor = torch.tensor(X_input, dtype=torch.float32)

        if X_tensor.ndim == 3 and X_tensor.shape[2] < X_tensor.shape[1]:  # shape: (N, 15, 4)
            X_tensor = X_tensor.permute(0, 2, 1)  # → (N, 4, 15)

        # Normalize per window
        mean = X_tensor.mean(dim=2, keepdim=True)
        std = X_tensor.std(dim=2, keepdim=True) + 1e-8
        X_tensor = (X_tensor - mean) / std
        X_tensor = X_tensor.to(self.device)

        with torch.no_grad():
            outputs = self.model(X_tensor)
            preds = torch.argmax(outputs, dim=1)

        return preds.cpu().numpy()

    def plot_losses(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.train_losses, label='Training Loss', marker='o')
        plt.plot(self.val_losses, label='Validation Loss', marker='x')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training vs Validation Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    def save_model(self, path):
        torch.save(self.model.state_dict(), path)
        print(f"✅ Model saved to: {path}")
