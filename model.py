import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MFCC(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_MFCC, self).__init__()

        # ---- Convolutional Blocks ----
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        # Pooling layer
        self.pool = nn.MaxPool2d(2, 2)

        # Will be calculated dynamically
        self.flatten_dim = None
        self.fc1 = None
        self.fc2 = None

        self.dropout = nn.Dropout(0.3)
        self.num_classes = num_classes

    def forward(self, x):
        # Block 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)

        # Block 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        # Block 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)

        # Lazy FC init
        if self.flatten_dim is None:
            self.flatten_dim = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(self.flatten_dim, 256).to(x.device)
            self.fc2 = nn.Linear(256, self.num_classes).to(x.device)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)

        return x
