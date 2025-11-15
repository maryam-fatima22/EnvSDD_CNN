from preprocess import compute_mfcc

mfcc_train = compute_mfcc(r"F:\Selected\fake\100263-2-0-137.wav")
mfcc_val = compute_mfcc(r"F:\Selected2\fake\12647-3-0-0noise_low.wav")

print("Train MFCC shape:", mfcc_train.shape)
print("Val MFCC shape:", mfcc_val.shape)
