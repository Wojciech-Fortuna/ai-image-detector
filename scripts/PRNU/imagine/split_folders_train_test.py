import argparse
import json
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple

import numpy as np
import cv2
from PIL import Image

STATE_FILE_NAME = ".incremental_zip_state.json"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def bytes_to_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


def dir_size_bytes(path: Path) -> int:
    total = 0
    for root, _, files in os.walk(path):
        for f in files:
            try:
                total += (Path(root) / f).stat().st_size
            except FileNotFoundError:
                pass
    return total


def count_images(path: Path) -> int:
    cnt = 0
    for root, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                cnt += 1
    return cnt


def filtered_order(groups: dict, process_root: bool) -> list:
    names = list(groups.keys())
    if not process_root and "__ROOT__" in names:
        names.remove("__ROOT__")
    names.sort()
    return names


def list_groups_on_disk(data_root: Path, include_root: bool) -> dict:
    groups = {}

    if include_root:
        groups["__ROOT__"] = data_root

    for entry in data_root.iterdir():
        if entry.is_dir():
            groups[entry.name] = entry

    return groups


def imread_gray_float(path: str) -> np.ndarray:
    try:
        with Image.open(path) as im:
            im = im.convert("L")
            img = np.array(im, dtype=np.float32) / 255.0
            return img
    except Exception as e:
        print(f"[WARN] PIL could not read {path} ({e}), trying cv2.imread...")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot open image: {path}")
    return img.astype(np.float32) / 255.0


def collect_all_images(extract_dir: Path) -> List[str]:
    img_paths: List[str] = []
    for root, _, files in os.walk(extract_dir):
        root_path = Path(root)
        for f in files:
            if f.lower().endswith(VALID_EXT):
                img_paths.append(str(root_path / f))
    return sorted(img_paths)


def load_state(state_path: Path) -> dict:
    if state_path.exists():
        try:
            return json.loads(state_path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_state(state_path: Path, data: dict) -> None:
    state_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def write_list(path: Path, items: List[str]) -> None:
    if not items:
        text = ""
    else:
        text = "\n".join(items)
    path.write_text(text, encoding="utf-8")


def check_images_chunk(
    paths: List[str],
) -> Tuple[List[Tuple[str, Tuple[int, int]]], List[str]]:
    valid_local: List[Tuple[str, Tuple[int, int]]] = []
    invalid_local: List[str] = []

    for p in paths:
        try:
            I = imread_gray_float(p)
            H, W = I.shape
            valid_local.append((p, (H, W)))
        except Exception as e:
            print(f"[WARN] Cannot read {p}: {e}")
            invalid_local.append(p)

    return valid_local, invalid_local


def find_valid_and_invalid(
    img_paths: List[str],
    workers: int = 4,
) -> Tuple[List[str], List[str], Tuple[int, int] or None]:
    if not img_paths:
        return [], [], None

    chunks = [[] for _ in range(max(1, workers))]
    for i, p in enumerate(img_paths):
        chunks[i % len(chunks)].append(p)

    all_valid: List[Tuple[str, Tuple[int, int]]] = []
    all_invalid: List[str] = []

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(check_images_chunk, ch) for ch in chunks if ch]
            for f in as_completed(futs):
                vloc, iloc = f.result()
                all_valid.extend(vloc)
                all_invalid.extend(iloc)
    else:
        vloc, iloc = check_images_chunk(img_paths)
        all_valid.extend(vloc)
        all_invalid.extend(iloc)

    if not all_valid:
        return [], all_invalid, None

    ref_shape = all_valid[0][1]

    valid_paths = []
    invalid_size = []
    for p, shape in all_valid:
        if shape == ref_shape:
            valid_paths.append(p)
        else:
            invalid_size.append(p)

    invalid_paths = all_invalid + invalid_size
    return sorted(valid_paths), sorted(invalid_paths), ref_shape


def split_train_test(
    valid_paths: List[str],
    test_ratio: float = 0.2,
    seed: int = 123,
) -> Tuple[List[str], List[str]]:
    n = len(valid_paths)
    if n == 0:
        return [], []

    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    shuffled = [valid_paths[i] for i in idx]

    if n >= 2:
        n_test = 1
    else:
        n_test = 0

    test_paths = shuffled[:n_test]
    train_paths = shuffled[n_test:]

    return sorted(train_paths), sorted(test_paths)


def process_group(
    group_name: str,
    extract_dir: Path,
    data_root: Path,
    results_dir: Path,
    workers: int,
    test_ratio: float,
    seed: int,
) -> Tuple[bool, str]:
    img_paths: List[str] = collect_all_images(extract_dir)
    if not img_paths:
        return False, "No images in the folder."

    print(f"[INFO] Found {len(img_paths)} images. Checking validity...")

    valid_paths_abs, invalid_paths_abs, ref_shape = find_valid_and_invalid(
        img_paths, workers=workers
    )

    if ref_shape is None:
        print("[WARN] No image could be read successfully.")
        valid_paths_abs = []
        invalid_paths_abs = sorted(set(img_paths))

    print(
        f"[INFO] valid={len(valid_paths_abs)}  invalid={len(invalid_paths_abs)}  "
        f"(ref_shape={ref_shape})"
    )

    train_abs, test_abs = split_train_test(
        valid_paths_abs, test_ratio=test_ratio, seed=seed
    )

    def rel(p: str) -> str:
        return str(Path(p).resolve().relative_to(data_root))

    train_rel = [rel(p) for p in train_abs]
    test_rel = [rel(p) for p in test_abs]
    invalid_rel = [rel(p) for p in invalid_paths_abs]

    group_res_dir = results_dir / group_name
    group_res_dir.mkdir(parents=True, exist_ok=True)

    write_list(group_res_dir / "train.txt", train_rel)
    write_list(group_res_dir / "test.txt", test_rel)
    write_list(group_res_dir / "invalid.txt", invalid_rel)

    summary = {
        "group": group_name,
        "ref_shape": ref_shape,
        "counts": {
            "total": len(img_paths),
            "valid": len(valid_paths_abs),
            "invalid": len(invalid_paths_abs),
            "train": len(train_rel),
            "test": len(test_rel),
        },
        "test_ratio_requested": test_ratio,
        "seed": seed,
    }

    (group_res_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    msg = (
        f"Saved split for group {group_name}: "
        f"train={len(train_rel)}, test={len(test_rel)}, invalid={len(invalid_rel)}"
    )
    return True, msg


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Train/test/invalid split from ready-made camera folders.\n"
            "Each first-level subdirectory in the data directory is treated as a separate group.\n"
            "Unreadable images or images with a size different from the reference go to 'invalid'."
        )
    )
    ap.add_argument(
        "data_root",
        type=Path,
        nargs="?",
        default=Path("."),
        help="Directory with camera subfolders (default: current directory)",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=Path("Results_Splits"),
        help="Output folder (train/test/invalid + summary.json)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous state if it exists",
    )
    ap.add_argument(
        "--root",
        action="store_true",
        help="Also process images directly inside data_root (without a subfolder)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the group list and exit without processing",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=(os.cpu_count() or 4),
        help="Number of CPU workers for reading/checking images",
    )
    ap.add_argument(
        "--test-ratio",
        type=float,
        default=0.2,
        help="Test data fraction (default 0.2 -> train:test ≈ 4:1)",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for the train/test split",
    )

    args = ap.parse_args()

    data_root: Path = args.data_root
    results_dir: Path = args.results

    if not data_root.exists():
        print(f"[ERROR] Data directory does not exist: {data_root}")
        sys.exit(1)

    data_root = data_root.resolve()
    results_dir.mkdir(parents=True, exist_ok=True)

    state_path = data_root / STATE_FILE_NAME
    state = load_state(state_path) if args.resume else {}

    groups = list_groups_on_disk(data_root, include_root=args.root)
    if not groups:
        print(
            f"[ERROR] No subdirectories (or root images if --root) found in: {data_root}"
        )
        sys.exit(1)

    order = filtered_order(groups, process_root=args.root)

    if args.dry_run:
        print("[DRY RUN] Groups to process (from disk):")
        for name in order:
            gdir = groups[name]
            total = dir_size_bytes(gdir)
            nimg = count_images(gdir)
            print(
                f"  - {name:30s} {bytes_to_human(total)}  "
                f"({nimg} image files)"
            )
        return

    already_done = set(state.get("done", []))
    to_process = [g for g in order if g not in already_done]

    if not to_process:
        print("No work to do. All selected groups have already been processed.")
        return

    print(f"Found {len(order)} groups; processing {len(to_process)}...")

    for group in to_process:
        gdir = groups[group]
        total_bytes = dir_size_bytes(gdir)
        img_count = count_images(gdir)
        print(
            f"\n==> Processing group: {group}  "
            f"({bytes_to_human(total_bytes)}, {img_count} image files)"
        )

        success, msg = process_group(
            group_name=group,
            extract_dir=gdir,
            data_root=data_root,
            results_dir=results_dir,
            workers=args.workers,
            test_ratio=args.test_ratio,
            seed=args.seed,
        )

        if success:
            print(f"[SPLIT][OK] {msg}")
        else:
            print(f"[SPLIT][ERROR] {msg}")

        already_done.add(group)
        save_state(state_path, {"done": sorted(list(already_done))})

    print("\nAll selected groups processed successfully.")


if __name__ == "__main__":
    main()
