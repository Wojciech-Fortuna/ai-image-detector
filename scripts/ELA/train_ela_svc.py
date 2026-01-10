from __future__ import annotations

import argparse
import io
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from tqdm import tqdm

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def list_images_recursive(root: Path) -> list[Path]:
    out: list[Path] = []
    if not root.exists():
        return out
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in IMG_EXTS:
            out.append(p)
    out.sort()
    return out


def open_rgb(path: Path) -> Image.Image:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def compute_ela_map(img_rgb: Image.Image, ela_quality: int, amplify: float) -> Image.Image:
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=int(ela_quality))
    buf.seek(0)
    jpg = Image.open(buf).convert("RGB")

    a = np.asarray(img_rgb).astype(np.int16)
    b = np.asarray(jpg).astype(np.int16)
    diff = np.abs(a - b).astype(np.float32) * float(amplify)
    diff = np.clip(diff, 0.0, 255.0)
    return Image.fromarray(diff.astype(np.uint8), mode="RGB")


def ela_features(path: Path, img_size: int, ela_quality: int, amplify: float) -> np.ndarray:
    img = open_rgb(path)
    ela = compute_ela_map(img, ela_quality=ela_quality, amplify=amplify)
    ela = ela.resize((img_size, img_size), Image.BILINEAR)
    ela01 = np.asarray(ela).astype(np.float32) / 255.0
    return ela01.reshape(-1).astype(np.float32)


def load_split(data_root: Path, split: str) -> tuple[list[Path], np.ndarray]:
    real_dir = data_root / split / "real"
    ai_dir = data_root / split / "ai"

    if not real_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {real_dir}")
    if not ai_dir.is_dir():
        raise FileNotFoundError(f"Missing directory: {ai_dir}")

    real_paths = list_images_recursive(real_dir)
    ai_paths = list_images_recursive(ai_dir)

    if len(real_paths) == 0:
        raise RuntimeError(f"No images found in: {real_dir}")
    if len(ai_paths) == 0:
        raise RuntimeError(f"No images found in: {ai_dir}")

    paths = real_paths + ai_paths
    y = np.array([0] * len(real_paths) + [1] * len(ai_paths), dtype=np.int64)
    return paths, y


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", required=True, help="DATA_ROOT containing train/ and test/")
    ap.add_argument("--img_size", type=int, default=128)
    ap.add_argument("--ela_quality", type=int, default=95)
    ap.add_argument("--ela_amplify", type=float, default=10.0)

    ap.add_argument("--kernel", default="rbf", choices=["rbf", "linear"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", default="scale")

    ap.add_argument("--out_dir", default="runs/ela_svc")
    ap.add_argument("--max_per_class_train", type=int, default=0,
                    help="0=use all, otherwise cap per class in train for quick tests")
    return ap.parse_args()


def cap_per_class(paths: list[Path], y: np.ndarray, max_per_class: int) -> tuple[list[Path], np.ndarray]:
    if max_per_class <= 0:
        return paths, y
    idx0 = [i for i, t in enumerate(y.tolist()) if t == 0][:max_per_class]
    idx1 = [i for i, t in enumerate(y.tolist()) if t == 1][:max_per_class]
    idx = idx0 + idx1
    paths2 = [paths[i] for i in idx]
    y2 = y[idx]
    return paths2, y2


def extract_matrix(paths: list[Path], y: np.ndarray, img_size: int, ela_quality: int, amplify: float) -> tuple[np.ndarray, np.ndarray]:
    X_list: list[np.ndarray] = []
    y_list: list[int] = []

    for p, lab in tqdm(list(zip(paths, y)), desc="ELA -> features"):
        try:
            feat = ela_features(p, img_size=img_size, ela_quality=ela_quality, amplify=amplify)
            X_list.append(feat)
            y_list.append(int(lab))
        except Exception as e:
            print(f"[WARN] skipping {p}: {e}")

    if not X_list:
        raise RuntimeError("No features extracted (all images failed?).")

    X = np.stack(X_list, axis=0)
    y_out = np.array(y_list, dtype=np.int64)
    return X, y_out


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_paths, y_train = load_split(data_root, "train")
    test_paths, y_test = load_split(data_root, "test")

    train_paths, y_train = cap_per_class(train_paths, y_train, args.max_per_class_train)

    X_train, y_train2 = extract_matrix(train_paths, y_train, args.img_size, args.ela_quality, args.ela_amplify)
    X_test, y_test2 = extract_matrix(test_paths, y_test, args.img_size, args.ela_quality, args.ela_amplify)

    model = Pipeline([
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("svc", SVC(kernel=args.kernel, C=args.C, gamma=args.gamma)),
    ])

    model.fit(X_train, y_train2)
    y_pred = model.predict(X_test)

    acc = float(accuracy_score(y_test2, y_pred))
    report = classification_report(y_test2, y_pred, digits=4)
    cm = confusion_matrix(y_test2, y_pred).tolist()

    print("Accuracy:", acc)
    print(report)
    print("Confusion matrix:", cm)

    joblib.dump(model, out_dir / "ela_svc_model.joblib")
    with open(out_dir / "preprocess.json", "w", encoding="utf-8") as f:
        json.dump({
            "img_size": args.img_size,
            "ela_quality": args.ela_quality,
            "ela_amplify": args.ela_amplify,
        }, f, indent=2)

    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({
            "accuracy": acc,
            "confusion_matrix": cm,
            "report": report,
            "args": vars(args),
            "train_counts": {"real": int((y_train2 == 0).sum()), "ai": int((y_train2 == 1).sum())},
            "test_counts": {"real": int((y_test2 == 0).sum()), "ai": int((y_test2 == 1).sum())},
        }, f, indent=2)

    print(f"Saved model to: {out_dir / 'ela_svc_model.joblib'}")


if __name__ == "__main__":
    main()
