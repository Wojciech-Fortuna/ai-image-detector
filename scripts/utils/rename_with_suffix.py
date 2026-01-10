import os
import sys

def add_suffix_to_files(root_dir, suffix):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            old_path = os.path.join(dirpath, filename)

            name, ext = os.path.splitext(filename)

            new_filename = f"{name}{suffix}{ext}"
            new_path = os.path.join(dirpath, new_filename)

            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python rename_with_suffix.py <directory> <suffix>")
        sys.exit(1)

    folder = sys.argv[1]
    suffix = sys.argv[2]

    if not os.path.isdir(folder):
        print("Error: The specified directory does not exist!")
        sys.exit(1)

    add_suffix_to_files(folder, suffix)
