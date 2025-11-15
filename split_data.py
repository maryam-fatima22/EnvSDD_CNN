import os
import shutil
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# --- Configuration ---
SOURCE_DIR = r"F:\Selected"        # Do NOT change
OUTPUT_TRAIN = r"F:\Selected_train"
OUTPUT_VAL = r"F:\Selected_val"
TRAIN_SPLIT = 0.8                  # 80% train, 20% val
RANDOM_SEED = 42


def collect_files(root_dir):
    """
    Read audio files from real/fake folders and return paths + labels.
    """
    all_files = []
    all_labels = []

    for class_name in ['real', 'fake']:
        class_path = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_path):
            print(f"Warning: Missing folder: {class_path}")
            continue

        label = 0 if class_name == 'real' else 1

        files = [
            os.path.join(class_path, f)
            for f in os.listdir(class_path)
            if f.lower().endswith(('.wav', '.flac', '.mp3'))
        ]

        all_files.extend(files)
        all_labels.extend([label] * len(files))

        print(f"Found {len(files)} files in '{class_name}'")

    return all_files, all_labels


def copy_files(file_list, target_root):
    os.makedirs(target_root, exist_ok=True)

    for file_path in tqdm(file_list, desc=f"Copying to {target_root}", ncols=80):
        class_name = os.path.basename(os.path.dirname(file_path))
        class_dir = os.path.join(target_root, class_name)
        os.makedirs(class_dir, exist_ok=True)

        dst = os.path.join(class_dir, os.path.basename(file_path))

        # Skip already copied files
        if os.path.exists(dst):
            continue

        shutil.copyfile(file_path, dst)


if __name__ == "__main__":
    # --- Step 1: Collect all file paths ---
    all_files, all_labels = collect_files(SOURCE_DIR)

    if not all_files:
        raise RuntimeError("No audio files found. Check SOURCE_DIR.")

    print(f"\nTotal files found: {len(all_files)}")

    # --- Step 2: Stratified split ---
    train_files, val_files = train_test_split(
        all_files,
        test_size=1 - TRAIN_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True,
        stratify=all_labels
    )

    print("\n--- Split Summary ---")
    print(f"Training:   {len(train_files)}")
    print(f"Validation: {len(val_files)}")

    # --- Step 3: Copy files ---
    print("\nCopying training files...")
    copy_files(train_files, OUTPUT_TRAIN)

    print("Copying validation files...")
    copy_files(val_files, OUTPUT_VAL)

    print("\n✅ Dataset split complete!")
