import os
from pathlib import Path


def remove_empty_dirs(root: Path):
    removed_count = 0

    for dirpath, dirnames, filenames in os.walk(root, topdown=False):
        p = Path(dirpath)

        if p == root:
            continue

        if not any(p.iterdir()):
            print(f"[DEL] Removing empty directory: {p}")
            p.rmdir()
            removed_count += 1

    print(f"\nRemoved {removed_count} empty directories.")


if __name__ == "__main__":
    root_dir = Path(".").resolve()
    print(f"Starting cleanup of empty directories in:\n  {root_dir}\n")
    remove_empty_dirs(root_dir)
