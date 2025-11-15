import torch
import os
import csv
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

from dataset import find_audio_files
from train_utils import SpeechDataset, collate_fn
from model import CNN_MFCC

def test_model(model_path, test_dir, batch_size=16, out_dir="results"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test files
    test_files, test_labels, classes = find_audio_files(test_dir)
    print(f"Found {len(test_files)} test files.")
    print(f"Detected classes: {classes}")

    # Create dataset + loader
    test_dataset = SpeechDataset(test_files, test_labels, classes)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

    # Load trained model
    model = CNN_MFCC(num_classes=len(classes)).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    y_true = []
    y_pred = []
    predictions = []

    # Run inference
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            _, preds = torch.max(outputs, dim=1)

            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

            # Save per-sample prediction
            for i in range(len(preds)):
                predictions.append([
                    test_files[len(predictions)],
                    classes[preds[i].item()],
                    classes[yb[i].item()]
                ])

    # Accuracy
    correct = sum([int(a==b) for a,b in zip(y_true, y_pred)])
    total = len(y_true)
    accuracy = correct / total
    print(f"\n🎉 Test Accuracy: {accuracy * 100:.2f}%")

    # Classification report
    report = classification_report(y_true, y_pred, target_names=classes, digits=2)
    print("\n📋 Classification Report:\n")
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    os.makedirs(out_dir, exist_ok=True)
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.show()
    print(f"✅ Confusion matrix saved at '{cm_path}'")

    # Save predictions CSV
    csv_path = os.path.join(out_dir, "test_results.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "predicted", "actual"])
        writer.writerows(predictions)
    print(f"📁 Saved test predictions to {csv_path}")

    return accuracy, report


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model checkpoint (.pt)")
    parser.add_argument("--test_dir", type=str, required=True, help="Path to test folder")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--out_dir", type=str, default="results")

    args = parser.parse_args()
    test_model(args.model, args.test_dir, args.batch_size, args.out_dir)
