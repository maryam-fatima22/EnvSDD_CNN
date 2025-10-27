import librosa
import numpy as np
import torch

# ===============================================================
# Function: compute_mfcc
# ===============================================================
def compute_mfcc(path, sr=16000, n_mfcc=13, n_fft=512, hop_length=256, max_duration=2.0):
    """
    Loads an audio file, pads/trims it to a fixed duration, and computes normalized MFCC features.

    Args:
        path (str): Path to the audio file.
        sr (int): Sampling rate.
        n_mfcc (int): Number of MFCC features to compute.
        n_fft (int): FFT window size.
        hop_length (int): Hop length for STFT.
        max_duration (float): Max duration (in seconds) to trim/pad audio for consistency.

    Returns:
        np.ndarray: Normalized MFCC array of shape (n_mfcc, time_frames)
    """
    # Load audio
    y, sr = librosa.load(path, sr=sr)

    # Ensure consistent duration
    max_len = int(max_duration * sr)
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)), mode='reflect')
    else:
        y = y[:max_len]

    # Compute MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

    # Normalize features (zero mean, unit variance)
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-6)

    return mfcc


# ===============================================================
# Function: mfcc_to_input
# ===============================================================
def mfcc_to_input(mfcc):
    """
    Converts MFCC numpy array to PyTorch tensor input for CNN.

    Args:
        mfcc (np.ndarray): MFCC feature array (n_mfcc, time_frames)

    Returns:
        torch.Tensor: Tensor of shape (1, n_mfcc, time_frames)
    """
    # Convert numpy array to float32 tensor
    tensor = torch.tensor(mfcc, dtype=torch.float32)

    # Add channel dimension for CNN input: (1, n_mfcc, time)
    tensor = tensor.unsqueeze(0)

    return tensor

