import os
import sys

def count_images(folder):
    extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    count = 0
    for file in os.listdir(folder):
        _, ext = os.path.splitext(file)
        if ext.lower() in extensions:
            count += 1

    return count


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <folder_name>")
        sys.exit(1)

    folder_path = sys.argv[1]

    if not os.path.isdir(folder_path):
        print("The specified folder does not exist.")
        sys.exit(1)

    print("Number of images:", count_images(folder_path))
