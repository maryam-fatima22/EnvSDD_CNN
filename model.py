import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN_MFCC(nn.Module):
    def __init__(self, num_classes, n_mfcc=40, max_frames=44):
        super(CNN_MFCC, self).__init__()
        # input shape = (batch, 1, n_mfcc, max_frames)
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(5, 5), padding=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool = nn.MaxPool2d((2, 2))

        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        # compute flattened size
        h = n_mfcc // 4  # two poolings
        w = max_frames // 4
        self.flat_dim = 32 * h * w

        self.fc1 = nn.Linear(self.flat_dim, 128)
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool(x)
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    # quick sanity check
    model = CNN_MFCC(num_classes=3)
    x = torch.randn(2, 1, 40, 44)
    y = model(x)
    print(y.shape)  # expect (2, 3)
