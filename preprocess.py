import numpy as np
import librosa

# Settings
N_MFCC = 40
MAX_FRAMES = 44  # number of time frames after which we pad/trim


def compute_mfcc(y, sr, n_mfcc=N_MFCC, max_frames=MAX_FRAMES):
    # compute MFCCs with librosa
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    # mfcc shape: (n_mfcc, T)
    
    # pad or trim to fixed number of frames
    if mfcc.shape[1] < max_frames:
        pad_width = max_frames - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    else:
        mfcc = mfcc[:, :max_frames]
    
    # normalize per-example
    mfcc = (mfcc - np.mean(mfcc)) / (np.std(mfcc) + 1e-9)
    
    return mfcc.astype(np.float32)


# For PyTorch conv2d we want shape (C=1, H=n_mfcc, W=max_frames)
def mfcc_to_input(mfcc):
    return np.expand_dims(mfcc, axis=0)
