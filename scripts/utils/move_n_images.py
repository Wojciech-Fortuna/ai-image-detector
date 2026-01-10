import shutil
import random
from pathlib import Path
import argparse

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".jfif",
    ".bmp", ".gif", ".tif", ".tiff", ".webp"
}

def collect_images(src_root: Path, dst_root: Path, n: int):
    dst_root.mkdir(exist_ok=True)

    root_images = [
        f for f in src_root.iterdir()
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    ]

    if root_images:
        files_to_move = random.sample(root_images, min(n, len(root_images)))
        print(f"Root directory: {src_root} -> moving {len(files_to_move)} files to {dst_root}")

        for f in files_to_move:
            shutil.move(str(f), str(dst_root / f.name))

    for subdir in src_root.iterdir():
        if not subdir.is_dir():
            continue

        dst_subdir = dst_root / subdir.name
        dst_subdir.mkdir(parents=True, exist_ok=True)

        image_files = [
            f for f in subdir.iterdir()
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_files:
            print(f"No images found in folder: {subdir}")
            continue

        files_to_move = random.sample(image_files, min(n, len(image_files)))
        print(f"Directory: {subdir} -> moving {len(files_to_move)} files to {dst_subdir}")

        for f in files_to_move:
            shutil.move(str(f), str(dst_subdir / f.name))

    print("Done!")


def main():
    parser = argparse.ArgumentParser(
        description="Randomly collect images from a directory and its subdirectories."
    )
    parser.add_argument(
        "src_root",
        type=Path,
        help="Source root directory"
    )
    parser.add_argument(
        "dst_root",
        type=Path,
        help="Destination root directory"
    )
    parser.add_argument(
        "-n", "--num-images",
        type=int,
        default=60,
        help="Number of images to move from each directory (default: 60)"
    )

    args = parser.parse_args()

    if not args.src_root.is_dir():
        print("Error: Source directory does not exist.")
        return

    collect_images(args.src_root, args.dst_root, args.num_images)


if __name__ == "__main__":
    main()
