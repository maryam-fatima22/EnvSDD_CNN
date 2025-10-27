import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN_MFCC(nn.Module):
    def __init__(self, num_classes=2):
        super(CNN_MFCC, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)

        # Placeholder for flatten dimension
        self.flatten_dim = None  
        self.fc1 = None
        self.fc2 = None

        # Will initialize fully connected layers later in forward()
        self.num_classes = num_classes

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)

        # Lazy initialization: compute flatten size dynamically
        if self.flatten_dim is None:
            self.flatten_dim = x.view(x.size(0), -1).size(1)
            self.fc1 = nn.Linear(self.flatten_dim, 128).to(x.device)
            self.fc2 = nn.Linear(128, self.num_classes).to(x.device)

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
