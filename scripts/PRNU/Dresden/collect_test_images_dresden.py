import argparse
import os
import shutil
import zipfile
from pathlib import Path


def copy_with_unique_name_from_zip(
    zf: zipfile.ZipFile, zi: zipfile.ZipInfo, dst_dir: Path, base_name: str
) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)

    dst_path = dst_dir / base_name
    if not dst_path.exists():
        with zf.open(zi) as src, open(dst_path, "wb") as dst:
            shutil.copyfileobj(src, dst)
        return dst_path

    stem, ext = os.path.splitext(base_name)
    i = 1
    while True:
        candidate = dst_dir / f"{stem}_{i}{ext}"
        if not candidate.exists():
            with zf.open(zi) as src, open(candidate, "wb") as dst:
                shutil.copyfileobj(src, dst)
            return candidate
        i += 1


def find_zip_member_for_rel_path(
    zf: zipfile.ZipFile, rel_path: str
) -> zipfile.ZipInfo | None:
    rel_norm = rel_path.replace("\\", "/").lstrip("/")

    candidates = []
    for zi in zf.infolist():
        if zi.is_dir():
            continue
        name_norm = zi.filename.replace("\\", "/").lstrip("/")
        if name_norm.endswith(rel_norm):
            candidates.append(zi)

    if not candidates:
        return None

    if len(candidates) > 1:
        print(
            f"[WARN] Multiple ZIP candidates found for rel_path='{rel_path}', "
            f"using the first one: {candidates[0].filename}"
        )

    return candidates[0]


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Copies (extracts from ZIP) all images from the test set (test.txt) "
            "into a single directory, with filenames prefixed by the group name."
        )
    )
    ap.add_argument(
        "zip_path",
        type=Path,
        help="Path to the Dresden ZIP archive (e.g., archive.zip)",
    )
    ap.add_argument(
        "splits_root",
        type=Path,
        help="Directory with split results (e.g., Results_Splits_Dresden)",
    )
    ap.add_argument(
        "out_dir",
        type=Path,
        help="Destination directory where all test images will be extracted",
    )

    args = ap.parse_args()

    zip_path: Path = args.zip_path.resolve()
    splits_root: Path = args.splits_root.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not zip_path.exists():
        print(f"[ERROR] zip_path does not exist: {zip_path}")
        return

    if not splits_root.exists():
        print(f"[ERROR] splits_root does not exist: {splits_root}")
        return

    print(f"[INFO] zip_path    = {zip_path}")
    print(f"[INFO] splits_root = {splits_root}")
    print(f"[INFO] out_dir     = {out_dir}")

    total_copied = 0
    total_missing = 0

    with zipfile.ZipFile(zip_path, "r") as zf:
        for group_dir in sorted(p for p in splits_root.iterdir() if p.is_dir()):
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
                zi = find_zip_member_for_rel_path(zf, rel_path)
                if zi is None:
                    print(
                        f"[ERROR] Could not find ZIP member matching: {rel_path} "
                        f"(group {group_name})"
                    )
                    total_missing += 1
                    continue

                original_name = Path(zi.filename).name
                new_name = f"{group_name}_{original_name}"

                _dst = copy_with_unique_name_from_zip(zf, zi, out_dir, new_name)
                total_copied += 1

    print("\n[SUMMARY]")
    print(f"  Test files extracted: {total_copied}")
    if total_missing > 0:
        print(f"  WARNING: missing in ZIP: {total_missing} files")
    print(f"  Output directory: {out_dir}")


if __name__ == "__main__":
    main()
