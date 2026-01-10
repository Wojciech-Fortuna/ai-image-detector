import os
import shutil
import random

DATA_DIR = "data"
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)


def split_class(class_name):
    src_dir = os.path.join(DATA_DIR, class_name)
    files = [
        f for f in os.listdir(src_dir)
        if os.path.isfile(os.path.join(src_dir, f))
    ]

    random.shuffle(files)
    split_idx = int(len(files) * TRAIN_RATIO)

    train_files = files[:split_idx]
    test_files = files[split_idx:]

    for split, split_files in [("train", train_files), ("test", test_files)]:
        dst_dir = os.path.join(DATA_DIR, split, class_name)
        os.makedirs(dst_dir, exist_ok=True)

        for file in split_files:
            src_path = os.path.join(src_dir, file)
            dst_path = os.path.join(dst_dir, file)
            shutil.copy2(src_path, dst_path)

    print(f"{class_name}: {len(train_files)} train / {len(test_files)} test")


def main():
    for class_name in ["real", "ai"]:
        split_class(class_name)

    print("\nData split completed")


if __name__ == "__main__":
    main()
