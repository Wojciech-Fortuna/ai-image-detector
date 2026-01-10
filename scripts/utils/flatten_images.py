import os
import shutil
from pathlib import Path
import argparse

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".jfif",
    ".bmp", ".gif", ".tif", ".tiff", ".webp"
}

def get_free_name(dst_dir: Path, filename: str) -> Path:
    base = Path(filename).stem
    suffix = Path(filename).suffix
    candidate = dst_dir / filename
    counter = 1

    while candidate.exists():
        candidate = dst_dir / f"{base}_{counter}{suffix}"
        counter += 1

    return candidate


def move_images_to_root(root_dir: Path):
    if not root_dir.is_dir():
        print(f"The specified directory does not exist: {root_dir}")
        return

    print(f"Starting work in directory: {root_dir.resolve()}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        current_dir = Path(dirpath)

        if current_dir == root_dir:
            continue

        for name in filenames:
            src_file = current_dir / name
            if src_file.suffix.lower() not in IMAGE_EXTENSIONS:
                continue

            dst_file = get_free_name(root_dir, src_file.name)
            print(f"Moving: {src_file}  ->  {dst_file}")
            shutil.move(str(src_file), str(dst_file))

    removed_dirs = 0
    for dirpath, dirnames, filenames in os.walk(root_dir, topdown=False):
        current_dir = Path(dirpath)

        if current_dir == root_dir:
            continue

        if not dirnames and not filenames:
            try:
                current_dir.rmdir()
                removed_dirs += 1
                print(f"Removed empty directory: {current_dir}")
            except OSError:
                print(f"Failed to remove directory: {current_dir}")

    print(f"Done! Empty directories removed: {removed_dirs}")


def main():
    parser = argparse.ArgumentParser(
        description="Move all images from subdirectories into the root directory "
                    "and remove empty subdirectories."
    )
    parser.add_argument(
        "root_dir",
        type=Path,
        help="Root directory"
    )

    args = parser.parse_args()
    move_images_to_root(args.root_dir)


if __name__ == "__main__":
    main()
