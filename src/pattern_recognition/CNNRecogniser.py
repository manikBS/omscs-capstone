# src/pattern_recognition/CNNRecogniser.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class ChartPatternCNN(object):
    def __init__(
        self,
        input_channels: int = 1,
        num_classes: int = 5,
        learning_rate: float = 1e-3,
    ):
        """
            CNN classifier for technical pattern recognition.
        """

        self.conv_block = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(1),  # Compress over time dimension
        )

        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes),
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self):
        pass