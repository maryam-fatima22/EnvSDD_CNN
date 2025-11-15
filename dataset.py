import os
import random
import soundfile as sf
from sklearn.model_selection import train_test_split
import numpy as np


def find_audio_files(data_dir, exts=['.wav', '.flac', '.mp3']):
    items = []
    labels = []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    for label in classes:
        folder = os.path.join(data_dir, label)
        for fname in os.listdir(folder):
            if os.path.splitext(fname)[1].lower() in exts:
                path = os.path.join(folder, fname)

                # ==========================
                # Skip corrupted audio files
                # ==========================
                try:
                    sf.info(path)
                except Exception as e:
                    print(f"[WARNING] Skipping corrupted file: {path} ({e})")
                    continue

                items.append(path)
                labels.append(label)
    return items, labels, classes


def train_val_split(filepaths, labels, test_size=0.2, random_state=42):
    return train_test_split(filepaths, labels, test_size=test_size, stratify=labels, random_state=random_state)


def load_audio(path, sr=None):
    # -----------------------------------------------------------
    # TRY loading audio — skip corrupted files automatically
    # -----------------------------------------------------------
    try:
        y, sr = sf.read(path, dtype='float32')
    except Exception as e:
        print(f"[WARNING] Corrupted audio encountered in load_audio: {path} ({e})")
        return None, None

    # soundfile may return shape (n,) or (n, channels)
    if y is not None and y.ndim > 1:
        y = y.mean(axis=1)

    return y, sr


# Utility to create a tiny dummy dataset (sine waves) for testing
def make_dummy_dataset(out_dir, n_classes=3, samples_per_class=10, duration=1.0, sr=16000):
    os.makedirs(out_dir, exist_ok=True)
    for c in range(n_classes):
        label = f'class_{c}'
        folder = os.path.join(out_dir, label)
        os.makedirs(folder, exist_ok=True)
        freq = 300 + c * 200
        for i in range(samples_per_class):
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            y = 0.1 * np.sin(2 * np.pi * freq * t) * (1 + 0.1 * np.random.randn(len(t)))
            fname = os.path.join(folder, f'{label}_{i}.wav')
            sf.write(fname, y, sr)
    print(f"Dummy dataset created at {out_dir}")


# Example usage
if __name__ == "__main__":
    out_dir = "dummy_data"
    make_dummy_dataset(out_dir)
    items, labels, classes = find_audio_files(out_dir)
    print("Classes:", classes)
    print("Found", len(items), "audio files.")
