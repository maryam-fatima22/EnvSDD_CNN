import os
from collections import Counter

# CHANGE THESE PATHS IF NEEDED
train_dir = r"F:\Selected"
val_dir = r"F:\Selected2"

def count_classes(folder):
    class_counts = {}

    for cls in os.listdir(folder):
        cls_path = os.path.join(folder, cls)

        if not os.path.isdir(cls_path):
            continue

        class_counts[cls] = len(os.listdir(cls_path))

    return class_counts


print("Training Set Distribution:")
print(count_classes(train_dir))

print("\nValidation Set Distribution:")
print(count_classes(val_dir))
