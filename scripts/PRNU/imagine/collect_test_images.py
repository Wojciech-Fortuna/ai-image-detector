import argparse
import os
import shutil
from pathlib import Path


def copy_with_unique_name(src: Path, dst_dir: Path, base_name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_path = dst_dir / base_name
    if not dst_path.exists():
        shutil.copy2(src, dst_path)
        return dst_path

    stem, ext = os.path.splitext(base_name)
    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i}{ext}"
        if not candidate.exists():
            shutil.copy2(src, candidate)
            return candidate
        i += 1


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Copies all images from the test set (test.txt) into one directory, "
            "with filenames prefixed by the group name."
        )
    )
    ap.add_argument(
        "data_root",
        type=Path,
        help="Data directory (the same one used in split_folders_train_test.py)",
    )
    ap.add_argument(
        "results_root",
        type=Path,
        help="Split results directory (e.g. Results_Splits)",
    )
    ap.add_argument(
        "out_dir",
        type=Path,
        help="Destination directory where all test images will be copied",
    )

    args = ap.parse_args()

    data_root: Path = args.data_root.resolve()
    results_root: Path = args.results_root.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not data_root.exists():
        print(f"[ERROR] data_root does not exist: {data_root}")
        return

    if not results_root.exists():
        print(f"[ERROR] results_root does not exist: {results_root}")
        return

    print(f"[INFO] data_root    = {data_root}")
    print(f"[INFO] results_root = {results_root}")
    print(f"[INFO] out_dir      = {out_dir}")

    total_copied = 0
    total_missing = 0

    for group_dir in sorted(p for p in results_root.iterdir() if p.is_dir()):
        group_name = group_dir.name
        test_file = group_dir / "test.txt"

        if not test_file.exists():
            print(f"[WARN] No test.txt for group {group_name}, skipping.")
            continue

        lines = [
            ln.strip()
            for ln in test_file.read_text(encoding="utf-8").splitlines()
            if ln.strip()
        ]

        if not lines:
            print(f"[INFO] No test files in group {group_name}.")
            continue

        print(f"[INFO] Group {group_name}: {len(lines)} test files.")

        for rel_path in lines:
            src = (data_root / rel_path).resolve()

            if not src.exists():
                print(
                    f"[ERROR] File not found: {src} (from group {group_name} test.txt)"
                )
                total_missing += 1
                continue

            original_name = src.name
            new_name = f"{group_name}_{original_name}"

            _dst = copy_with_unique_name(src, out_dir, new_name)
            total_copied += 1

    print("\n[SUMMARY]")
    print(f"  Test files copied: {total_copied}")
    if total_missing > 0:
        print(f"  WARNING: missing files (not found in data_root): {total_missing}")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
