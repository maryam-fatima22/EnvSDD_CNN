import librosa
import numpy as np
import torch

# ===============================================================
# Function: compute_mfcc (Improved)
# ===============================================================
def compute_mfcc(
    path, 
    sr=16000, 
    n_mfcc=64, 
    n_fft=512, 
    hop_length=256, 
    max_duration=2.0
):
    """
    Loads audio, trims/pads it to fixed length, and computes MFCC + delta + delta-delta.
    
    Returns:
        np.ndarray: Shape (3 * n_mfcc, time_frames)
                    [MFCC ; Delta ; Delta-Delta]
        OR None if file is corrupted.
    """

    # -----------------------------------------------------------
    # Try loading audio — skip if corrupted
    # -----------------------------------------------------------
    try:
        y, sr = librosa.load(path, sr=sr)
    except Exception as e:
        print(f"[WARNING] Skipping corrupted file: {path} ({e})")
        return None

    # Ensure consistent duration
    max_len = int(max_duration * sr)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode="reflect")
    else:
        y = y[:max_len]

    # MFCC base
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr,
        n_mfcc=n_mfcc,
        n_fft=n_fft,
        hop_length=hop_length
    )

    # Delta & Delta-Delta
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    # Stack vertically → channels
    feats = np.vstack([mfcc, delta, delta2])  # shape: (3*n_mfcc, time)

    # Normalize per-feature
    feats = (feats - np.mean(feats, axis=1, keepdims=True)) / (np.std(feats, axis=1, keepdims=True) + 1e-6)

    return feats


# ===============================================================
# Function: mfcc_to_input
# ===============================================================
def mfcc_to_input(mfcc):
    """
    Convert MFCC into CNN input: (1, channels, time)
    """
    tensor = torch.tensor(mfcc, dtype=torch.float32)
    tensor = tensor.unsqueeze(0)   # → (1, C, T)
    return tensor
