import torch
from torch.utils.data import Dataset
import numpy as np
from preprocess import compute_mfcc, mfcc_to_input


# Custom Dataset class
class SpeechDataset(Dataset):
    def __init__(self, filepaths, labels, classes):
        self.filepaths = filepaths
        self.labels = labels
        self.classes = classes
        self.class_to_idx = {cls: i for i, cls in enumerate(classes)}

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        # Fixed indentation + variable name
        path = self.filepaths[idx]
        label = self.labels[idx]

        # compute_mfcc expects the file path
        mfcc = compute_mfcc(path)
        x = mfcc_to_input(mfcc)

        y = self.class_to_idx[label]
        return x, y


# Collate function (combine samples into a batch)
def collate_fn(batch):
    xs, ys = zip(*batch)
    xs = torch.stack(xs)
    ys = torch.tensor(ys)
    return xs, ys


# Training loop for one epoch
def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss, correct, count = 0, 0, 0

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)

        optimizer.zero_grad()
        outputs = model(xb)
        loss = criterion(outputs, yb)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * xb.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == yb).sum().item()
        count += xb.size(0)

    return total_loss / count, correct / count


# Evaluation loop
def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, correct, count = 0, 0, 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            loss = criterion(outputs, yb)

            total_loss += loss.item() * xb.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == yb).sum().item()
            count += xb.size(0)

    return total_loss / count, correct / count

