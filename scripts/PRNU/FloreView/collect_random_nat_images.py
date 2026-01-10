import argparse
import os
import random
import shutil
from pathlib import Path

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def collect_images_under_nat(nat_dir: Path):
    img_paths = []
    for root, _, files in os.walk(nat_dir):
        root_path = Path(root)
        for f in files:
            if f.lower().endswith(VALID_EXT):
                img_paths.append(root_path / f)
    return sorted(img_paths)


def copy_with_name(src: Path, dst_dir: Path, new_name: str) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    base = new_name
    dst_path = dst_dir / base

    if not dst_path.exists():
        shutil.copy2(src, dst_path)
        return dst_path

    stem, ext = os.path.splitext(base)
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
            "For each device, randomly selects one photo from the 'Nat' subdirectory "
            "and copies it into a shared output directory."
        )
    )
    ap.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("FloreView_Dataset"),
        help="Directory with device folders (default: 'FloreView_Dataset')",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("Nat_Samples"),
        help="Output directory for the sampled photos (default: 'Nat_Samples')",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional RNG seed for reproducible results (default: none)",
    )

    args = ap.parse_args()

    dataset_root: Path = args.dataset_root.resolve()
    out_dir: Path = args.out.resolve()
    seed = args.seed

    if seed is not None:
        random.seed(seed)

    if not dataset_root.exists():
        print(f"[ERROR] dataset_root directory does not exist: {dataset_root}")
        return

    print(f"[INFO] dataset_root = {dataset_root}")
    print(f"[INFO] out_dir      = {out_dir}")
    if seed is not None:
        print(f"[INFO] RNG seed     = {seed}")

    total_devices = 0
    total_copied = 0
    skipped_no_nat = 0
    skipped_empty = 0

    for dev_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        device_name = dev_dir.name
        total_devices += 1

        nat_dir = dev_dir / "Nat"
        if not nat_dir.exists() or not nat_dir.is_dir():
            print(f"[WARN] Device {device_name}: no 'Nat' directory, skipping.")
            skipped_no_nat += 1
            continue

        imgs = collect_images_under_nat(nat_dir)
        if not imgs:
            print(f"[WARN] Device {device_name}: no image files under 'Nat', skipping.")
            skipped_empty += 1
            continue

        chosen = random.choice(imgs)
        original_name = chosen.name
        new_name = f"{device_name}_{original_name}"

        dst = copy_with_name(chosen, out_dir, new_name)
        print(
            f"[OK] {device_name}: selected {chosen.relative_to(dataset_root)} -> {dst.name}"
        )
        total_copied += 1

    print("\n[SUMMARY]")
    print(f"  Devices found:                 {total_devices}")
    print(f"  Photos copied:                 {total_copied}")
    print(f"  Skipped (no 'Nat'):            {skipped_no_nat}")
    print(f"  Skipped (empty 'Nat'):         {skipped_empty}")
    print(f"  Output directory:              {out_dir}")


if __name__ == "__main__":
    main()
