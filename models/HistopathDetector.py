import torch
import torch.nn as nn
import torch.nn.functional as F


class HistopathDetector(nn.Module):
    """Binary CNN that predicts whether an image is a histopathology image.

    Input shape: (batch, 3, 224, 224)
    Output shape: (batch,) unnormalized scalar logit.
    """

    def __init__(self):
        super(HistopathDetector, self).__init__()
        self.name = "HistopathDetector"

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.pool = nn.MaxPool2d(2, 2)

        # After 4 max-pooling layers: 224 -> 112 -> 56 -> 28 -> 14
        self.fc1 = nn.Linear(64 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        x = x.view(-1, 64 * 14 * 14)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        x = x.squeeze(1)
        return x