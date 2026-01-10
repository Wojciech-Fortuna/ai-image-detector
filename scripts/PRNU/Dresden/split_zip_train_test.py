import argparse
import json
import os
import sys
import shutil
import zipfile
import re
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Set
from collections import defaultdict

import numpy as np
import cv2
from PIL import Image

STATE_FILE_NAME = ".incremental_zip_state.json"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

MODEL_DIR_RE = re.compile(r".+_\d+$")


def bytes_to_human(n: int) -> str:
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if n < 1024.0:
            return f"{n:.1f} {unit}"
        n /= 1024.0
    return f"{n:.1f} PB"


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
    text = "" if not items else "\n".join(items)
    path.write_text(text, encoding="utf-8")


def list_top_level_groups(zf: zipfile.ZipFile) -> dict:
    groups = {}
    for zi in zf.infolist():
        if zi.is_dir():
            continue
        name = zi.filename.replace("\\", "/").lstrip("/")
        if not name or name.startswith("__MACOSX/"):
            continue

        if name.startswith("Dresden_Exp/"):
            rel = name[len("Dresden_Exp/"):]
        else:
            rel = name

        parts = rel.split("/", 1)
        group = "__ROOT__" if len(parts) == 1 else parts[0]
        groups.setdefault(group, []).append(zi)
    return groups


def filtered_order(groups: dict, process_root: bool) -> list:
    names = list(groups.keys())
    if not process_root and "__ROOT__" in names:
        names.remove("__ROOT__")
    names.sort()
    return names


def extract_group(
    zf: zipfile.ZipFile,
    group_name: str,
    group_members: list,
    dest_root: Path,
) -> Path:
    target_dir = dest_root if group_name == "__ROOT__" else dest_root / group_name
    target_dir.mkdir(parents=True, exist_ok=True)

    for zi in group_members:
        name = zi.filename.replace("\\", "/").lstrip("/")
        if name.startswith("Dresden_Exp/"):
            rel_after_root = name[len("Dresden_Exp/"):]
        else:
            rel_after_root = name

        if group_name != "__ROOT__":
            parts = rel_after_root.split("/", 1)
            rel_inside = parts[1] if len(parts) > 1 else ""
        else:
            rel_inside = rel_after_root

        out_path = (target_dir / rel_inside).resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with zf.open(zi) as src, open(out_path, "wb") as dst:
            shutil.copyfileobj(src, dst, length=1024 * 1024)

    return target_dir


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


def split_foreign_model_subfolders(
    group_name: str,
    extract_dir: Path,
    img_paths: List[str],
) -> Tuple[List[str], List[str], Set[str]]:
    normal: List[str] = []
    foreign: List[str] = []
    foreign_names: Set[str] = set()

    for p in img_paths:
        try:
            rel = Path(p).resolve().relative_to(extract_dir.resolve())
        except Exception:
            normal.append(p)
            continue

        parts = rel.parts
        if len(parts) >= 2:
            first = parts[0]
            if first != group_name and MODEL_DIR_RE.fullmatch(first):
                foreign.append(p)
                foreign_names.add(first)
                continue

        normal.append(p)

    return normal, foreign, foreign_names


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


def analyze_images_by_size(
    img_paths: List[str],
    workers: int = 4,
) -> Tuple[Dict[Tuple[int, int], List[str]], List[str]]:
    if not img_paths:
        return {}, []

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

    size_to_paths: Dict[Tuple[int, int], List[str]] = defaultdict(list)
    for p, shape in all_valid:
        size_to_paths[shape].append(p)

    return size_to_paths, sorted(all_invalid)


def process_group(
    group_name: str,
    extract_dir: Path,
    work_dir: Path,
    results_dir: Path,
    workers: int,
    seed: int,
) -> Tuple[bool, str]:
    img_paths: List[str] = collect_all_images(extract_dir)
    if not img_paths:
        return False, "No images found in the folder."

    normal_paths, foreign_paths, foreign_models = split_foreign_model_subfolders(
        group_name=group_name,
        extract_dir=extract_dir,
        img_paths=img_paths,
    )

    if foreign_models:
        print(
            f"[WARN] Detected foreign model subfolders in group {group_name}: "
            f"{sorted(foreign_models)}"
        )
        print(
            f"[WARN] All images from those subfolders will go to invalid: "
            f"{len(foreign_paths)} files"
        )

    print(
        f"[INFO] Found {len(img_paths)} images. "
        f"Normal={len(normal_paths)}, foreign-model={len(foreign_paths)}. "
        f"Checking sizes of normal images..."
    )

    size_to_paths, invalid_read = analyze_images_by_size(normal_paths, workers=workers)

    work_root = work_dir.resolve()

    if not size_to_paths:
        print("[WARN] No normal image could be read successfully.")
        invalid_paths_abs = sorted(set(normal_paths + foreign_paths))
        valid_used_paths: List[str] = []
        train_abs: List[str] = []
        test_abs: List[str] = []
        ref_shapes_list = []
    else:
        shape_counter = {shape: len(paths) for shape, paths in size_to_paths.items()}
        print("[INFO] Size distribution (shape -> count):")
        for shape, cnt in sorted(shape_counter.items(), key=lambda x: x[1], reverse=True):
            print(f"   {shape}: {cnt} files")

        big_shapes = {shape: paths for shape, paths in size_to_paths.items() if len(paths) >= 21}
        small_shapes = {shape: paths for shape, paths in size_to_paths.items() if len(paths) < 21}

        invalid_size_paths: List[str] = []
        for paths in small_shapes.values():
            invalid_size_paths.extend(paths)

        invalid_paths_abs = sorted(set(invalid_read + invalid_size_paths + foreign_paths))

        rng_base = seed
        train_abs = []
        test_abs = []
        valid_used_paths = []

        for i, (shape, paths) in enumerate(sorted(big_shapes.items(), key=lambda x: x[0])):
            n = len(paths)
            if n == 0:
                continue
            valid_used_paths.extend(paths)

            rng = np.random.default_rng(rng_base + i)
            idx = np.arange(n)
            rng.shuffle(idx)

            test_abs.append(paths[idx[0]])
            for j in idx[1:]:
                train_abs.append(paths[j])

        ref_shapes_list = [[shape[0], shape[1]] for shape in sorted(big_shapes.keys())]

    print(
        f"[INFO] valid={len(valid_used_paths)}  invalid={len(invalid_paths_abs)}  "
        f"(ref_shapes={ref_shapes_list})"
    )

    def rel(p: str) -> str:
        return str(Path(p).resolve().relative_to(work_root))

    train_rel = [rel(p) for p in sorted(train_abs)]
    test_rel = [rel(p) for p in sorted(test_abs)]
    invalid_rel = [rel(p) for p in invalid_paths_abs]

    group_res_dir = results_dir / group_name
    group_res_dir.mkdir(parents=True, exist_ok=True)

    write_list(group_res_dir / "train.txt", train_rel)
    write_list(group_res_dir / "test.txt", test_rel)
    write_list(group_res_dir / "invalid.txt", invalid_rel)

    sizes_summary = {
        f"{shape[0]}x{shape[1]}": len(paths) for shape, paths in size_to_paths.items()
    }

    summary = {
        "group": group_name,
        "ref_shapes": ref_shapes_list,
        "counts": {
            "total": len(img_paths),
            "normal_total": len(normal_paths),
            "foreign_total": len(foreign_paths),
            "valid": len(valid_used_paths),
            "invalid": len(invalid_paths_abs),
            "train": len(train_rel),
            "test": len(test_rel),
        },
        "foreign_models_detected": sorted(list(foreign_models)),
        "seed": seed,
        "work_dir": str(work_root),
        "sizes": sizes_summary,
    }

    (group_res_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    msg = (
        f"Saved split for group {group_name}: "
        f"train={len(train_rel)}, test={len(test_rel)}, invalid={len(invalid_rel)} "
        f"(foreign={len(foreign_paths)})"
    )
    return True, msg


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Train/test/invalid split directly from a ZIP archive (Dresden format).\n"
            "Images are grouped by size; for each size with >=21 images,\n"
            "1 image goes to test, the rest to train; smaller sizes and unreadable files -> invalid.\n"
            "After processing a group, its extracted folder is removed from work_dir.\n"
            "Additionally: foreign model subfolders inside a group are automatically sent to invalid."
        )
    )
    ap.add_argument("zip_path", type=Path, help="Path to the ZIP archive (Dresden)")
    ap.add_argument("work_dir", type=Path, help="Working directory (images will be extracted here)")
    ap.add_argument(
        "--results",
        type=Path,
        default=Path("Results_Splits_Dresden"),
        help="Output folder (train/test/invalid + summary.json)",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="Resume from a previous state if it exists (work_dir/.incremental_zip_state.json)",
    )
    ap.add_argument(
        "--root",
        action="store_true",
        help="Also process files from the ROOT level (without a folder)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Print group list and exit without extracting or splitting",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=(os.cpu_count() or 4),
        help="Number of CPU workers for reading/checking images",
    )
    ap.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for the train/test split",
    )

    args = ap.parse_args()

    zip_path: Path = args.zip_path
    work_dir: Path = args.work_dir
    results_dir: Path = args.results

    if not zip_path.exists():
        print(f"[ERROR] ZIP not found: {zip_path}")
        sys.exit(1)

    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    state_path = work_dir / STATE_FILE_NAME
    state = load_state(state_path) if args.resume else {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        groups = list_top_level_groups(zf)
        if not groups:
            print("[ERROR] No files in the ZIP archive (after filtering).")
            sys.exit(1)

        order = filtered_order(groups, process_root=args.root)

        if args.dry_run:
            print("[DRY RUN] Groups to process:")
            for name in order:
                total = sum(m.file_size for m in groups[name])
                print(f"  - {name:30s} {bytes_to_human(total)}  ({len(groups[name])} files)")
            return

        already_done = set(state.get("done", []))
        to_process = [g for g in order if g not in already_done]

        if not to_process:
            print("No work to do. All selected groups have already been processed.")
            return

        print(f"Found {len(order)} groups; processing {len(to_process)}...")

        for group in to_process:
            members = groups[group]
            total_bytes = sum(m.file_size for m in members)
            print(
                f"\n==> Processing group: {group}  "
                f"({bytes_to_human(total_bytes)}, {len(members)} files in ZIP)"
            )

            try:
                target_dir = extract_group(zf, group, members, work_dir)
            except Exception as e:
                print(f"[ERROR] Extraction failed for {group}: {e}")
                already_done.add(group)
                save_state(state_path, {"done": sorted(list(already_done))})
                continue

            print(f"[OK] Extracted to: {target_dir}")

            success, msg = process_group(
                group_name=group,
                extract_dir=target_dir,
                work_dir=work_dir,
                results_dir=results_dir,
                workers=args.workers,
                seed=args.seed,
            )

            if success:
                print(f"[SPLIT][OK] {msg}")
            else:
                print(f"[SPLIT][ERROR] {msg}")

            print(f"[CLEAN] Removing extracted folder: {target_dir}")

            def onexc(func, path, exc_info):
                try:
                    os.chmod(path, 0o666)
                    func(path)
                except Exception:
                    pass

            try:
                shutil.rmtree(target_dir, onexc=onexc)
            except Exception as e:
                print(f"[WARN] Could not remove {target_dir}: {e}")

            already_done.add(group)
            save_state(state_path, {"done": sorted(list(already_done))})

        print("\nAll selected groups processed successfully.")


if __name__ == "__main__":
    main()
