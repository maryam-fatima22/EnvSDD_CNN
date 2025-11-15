import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from dataset import find_audio_files
from preprocess import compute_mfcc, mfcc_to_input
from model import CNN_MFCC
from utils import save_checkpoint
from train_utils import SpeechDataset, collate_fn, train_epoch, eval_epoch


def plot_confusion_matrix(model, data_loader, device, class_names, save_path='plots/confusion_matrix.png'):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for xb, yb in data_loader:
            xb, yb = xb.to(device), yb.to(device)
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

    # Load audio files
    train_files, train_labels, classes = find_audio_files(args.train_dir)
    val_files, val_labels, _ = find_audio_files(args.val_dir)

    if not train_files or not val_files:
        raise RuntimeError("Training or validation files not found.")

    print(f"✅ Found {len(train_files)} training files and {len(val_files)} validation files.")
    print(f"Detected classes: {classes}")

    # Create datasets and loaders
    train_dataset = SpeechDataset(train_files, train_labels, classes)
    val_dataset = SpeechDataset(val_files, val_labels, classes)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model, criterion, optimizer
    model = CNN_MFCC(num_classes=len(classes)).to(device)
    criterion = nn.CrossEntropyLoss()

    if args.optimizer_type.lower() == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr_head)
    elif args.optimizer_type.lower() == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr_head, momentum=0.9)
    else:
        raise ValueError("Unsupported optimizer type. Use 'Adam' or 'SGD'.")

    # LR Scheduler (fixed: removed verbose)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='max',
        factor=0.5,
        patience=args.lr_scheduler_patience
    )

    # Training loop variables
    best_acc = 0.0
    early_stop_counter = 0
    train_losses, val_losses, train_accs, val_accs = [], [], [], []

    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc*100:.2f}% | "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc*100:.2f}%")

        # Save checkpoint
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc
            early_stop_counter = 0
        else:
            early_stop_counter += 1

        if epoch % args.save_every == 0 or is_best:
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc
            }, is_best, out_dir=args.out_dir, filename=f'checkpoint_epoch_{epoch}.pt')

        # Step LR scheduler
        scheduler.step(val_acc)

        # Early stopping
        if early_stop_counter >= args.early_stop_patience:
            print(f"⚠️ Early stopping triggered at epoch {epoch}")
            break

    print(f"\n✅ Training finished. Best validation accuracy: {best_acc:.4f}")
    print(f"Models saved in '{args.out_dir}'")

    # Plot curves
    os.makedirs('plots', exist_ok=True)
    plt.figure(); plt.plot(train_losses, label='Train Loss'); plt.plot(val_losses, label='Val Loss')
    plt.title('Loss Curve'); plt.xlabel('Epoch'); plt.ylabel('Loss'); plt.legend()
    plt.savefig('plots/loss_curve.pdf'); plt.close()

    plt.figure(); plt.plot(train_accs, label='Train Acc'); plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy Curve'); plt.xlabel('Epoch'); plt.ylabel('Accuracy'); plt.legend()
    plt.savefig('plots/accuracy_curve.pdf'); plt.close()
    print('📊 Plots saved in "plots/" folder.')

    # Confusion matrix
    plot_confusion_matrix(model, val_loader, device, class_names=classes)

    # Hyperparameters summary
    print("\n📋 Training Summary & Hyperparameters:")
    print(f"Sample Rate: 16000 Hz")
    print(f"Clip Length: {args.clip_length} sec")
    print(f"Batch Size: {args.batch_size}")
    print(f"Optimizer: {args.optimizer_type}")
    print(f"Learning Rate (Head): {args.lr_head}")
    print(f"Learning Rate (Backbone): {args.lr_backbone}")
    print(f"Epochs Run: {epoch}")
    print(f"Early Stopping Patience: {args.early_stop_patience}")
    print(f"LR Scheduler Patience: {args.lr_scheduler_patience}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_dir', type=str, required=True)
    parser.add_argument('--val_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default='results')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--save_every', type=int, default=1)

    # Hyperparameters
    parser.add_argument('--clip_length', type=float, default=2.0)
    parser.add_argument('--early_stop_patience', type=int, default=5)
    parser.add_argument('--lr_scheduler_patience', type=int, default=3)
    parser.add_argument('--optimizer_type', type=str, default='Adam')
    parser.add_argument('--lr_head', type=float, default=0.001)
    parser.add_argument('--lr_backbone', type=float, default=0.001)

    args = parser.parse_args()
    main(args)
