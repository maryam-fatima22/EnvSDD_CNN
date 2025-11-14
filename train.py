import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import matplotlib.pyplot as plt   # For plotting

from dataset import find_audio_files
from preprocess import compute_mfcc, mfcc_to_input
from model import CNN_MFCC
from utils import save_checkpoint
from train_utils import SpeechDataset, collate_fn, train_epoch, eval_epoch

def plot_confusion_matrix(model, data_loader, device, class_names=['fake', 'real'], save_path='plots/confusion_matrix.png'):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            outputs = model(xb)
            _, preds = torch.max(outputs, 1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(yb.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.show()
    print(f"✅ Confusion matrix saved at '{save_path}'")
def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Paths for training and validation folders
    train_dir = args.train_dir
    val_dir = args.val_dir

    # Load audio files
    train_files, train_labels, classes = find_audio_files(train_dir)
    val_files, val_labels, _ = find_audio_files(val_dir)

    if len(train_files) == 0:
        raise RuntimeError(f'❌ No training audio files found in {train_dir}')
    if len(val_files) == 0:
        raise RuntimeError(f'❌ No validation audio files found in {val_dir}')

    print(f"✅ Found {len(train_files)} training files and {len(val_files)} validation files.")
    print(f"Detected classes: {classes}")

    # Create Datasets
    train_dataset = SpeechDataset(train_files, train_labels, classes)
    val_dataset = SpeechDataset(val_files, val_labels, classes)

    # Create Data Loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    # Model setup
    model = CNN_MFCC(num_classes=len(classes))
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    # Lists to store training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []

    # Training loop
    for epoch in range(1, args.epochs + 1):

        # Clean epoch header
        print(f"\nEpoch {epoch}/{args.epochs}: 100%")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        # Save metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Clean summary line
        print(f"Epoch {epoch} | Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:.2f}%")

        # Save model only every N epochs
        if epoch % args.save_every == 0:
            is_best = val_acc > best_acc
            if is_best:
                best_acc = val_acc

            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, is_best, out_dir=args.out_dir, filename=f'checkpoint_epoch_{epoch}.pt')

    print(f'\n✅ Training finished. Best validation accuracy: {best_acc:.4f}')
    print(f'Models saved in "{args.out_dir}"')

    # Save training curves
    os.makedirs('plots', exist_ok=True)

    plt.figure()
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('plots/loss_curve.pdf')
    plt.close()

    plt.figure()
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(val_accs, label='Val Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig('plots/accuracy_curve.pdf')
    plt.close()

    print('📊 Plots saved in "plots/" folder.')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True,
                        help='Path to the training dataset (e.g., E:/selected)')
    parser.add_argument('--val_dir', type=str, required=True,
                        help='Path to the validation dataset (e.g., E:/selected2)')
    parser.add_argument('--out_dir', type=str, default='results',
                        help='Directory to save models and logs')
    parser.add_argument('--epochs', type=int, default=20,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--save_every', type=int, default=1,
                        help='Save checkpoint every N epochs')

    args = parser.parse_args()
    main(args)
