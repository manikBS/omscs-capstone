import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN1D_MultiChannel(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super(CNN1D_MultiChannel, self).__init__()

        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(64)
        self.pool1 = nn.MaxPool1d(kernel_size=2)

        self.conv2 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool2 = nn.MaxPool1d(kernel_size=2)

        self.flattened_size = 128 * 1  # Updated for input length = 10

        self.fc1 = nn.Linear(self.flattened_size, 64)
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        # x shape: (batch_size, 4, 10)
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))  # (B, 64, 4)
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))  # (B, 128, 1)
        x = x.view(x.size(0), -1)  # (B, 128)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)