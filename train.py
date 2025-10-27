import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse

from dataset import find_audio_files, train_val_split, make_dummy_dataset
from preprocess import compute_mfcc, mfcc_to_input
from model import CNN_MFCC
from utils import save_checkpoint
from train_utils import SpeechDataset, collate_fn, train_epoch, eval_epoch


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Option to create a dummy dataset for testing
    if args.make_dummy:
        make_dummy_dataset(args.data_dir, n_classes=3, samples_per_class=20)
        print("Dummy dataset created.")

    # Load audio files
    files, labels, classes = find_audio_files(args.data_dir)
    if len(files) == 0:
        raise RuntimeError(f'No audio files found in {args.data_dir}. Check path or use --make_dummy')

    # Split dataset
    X_train, X_val, y_train, y_val = train_val_split(files, labels, test_size=0.2)

    # Create Datasets and Loaders
    train_dataset = SpeechDataset(X_train, y_train, classes)
    val_dataset = SpeechDataset(X_val, y_val, classes)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)

    # Model setup
    model = CNN_MFCC(num_classes=len(classes))
    model.to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0

    # Training loop
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)

        print(f'Epoch {epoch}: Train loss {train_loss:.4f} acc {train_acc:.4f} | Val loss {val_loss:.4f} acc {val_acc:.4f}')

        # Save model
        is_best = val_acc > best_acc
        if is_best:
            best_acc = val_acc

        save_checkpoint({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc
        }, is_best, out_dir=args.out_dir, filename=f'checkpoint_epoch_{epoch}.pt')

    print(f'\nTraining finished ✅ Best validation accuracy: {best_acc:.4f}')
    print(f'Models saved in "{args.out_dir}"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='data', help='Path to dataset directory')
    parser.add_argument('--out_dir', type=str, default='results', help='Directory to save models and logs')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--make_dummy', action='store_true', help='Create a dummy dataset for quick testing')

    args = parser.parse_args()
    main(args)
