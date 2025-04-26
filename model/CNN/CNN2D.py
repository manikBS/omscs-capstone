import torch
import torch.nn as nn
import torch.nn.functional as F
class CNN2D_MultiChannel(nn.Module):
    def __init__(self, in_channels=4, num_classes=4):
        super(CNN2D_MultiChannel, self).__init__()


        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=(2, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.Identity()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.AdaptiveAvgPool2d((1, 1))

        self.fc1 = nn.Linear(64, 64)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)