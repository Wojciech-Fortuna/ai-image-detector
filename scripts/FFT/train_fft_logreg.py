import os
from typing import List, Tuple

import numpy as np
from PIL import Image, ImageOps
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, accuracy_score


REAL_DIR = "data/real"
AI_DIR   = "data/ai"

TARGET_SIZE = 512
NUM_BANDS = 8

RANDOM_STATE = 42


def prepare_gray(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    return img.convert("L")


def compute_fft_features(path: str) -> np.ndarray:
    img = Image.open(path)
    gray = prepare_gray(img)

    gray = gray.resize((TARGET_SIZE, TARGET_SIZE), Image.LANCZOS)

    arr = np.asarray(gray, dtype=np.float32)

    f = np.fft.fft2(arr)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)

    h, w = arr.shape
    cy, cx = h // 2, w // 2

    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    r_max = dist.max() + 1e-12

    band_edges = np.linspace(0.0, r_max, NUM_BANDS + 1, dtype=np.float32)

    total_energy = float(magnitude.sum()) + 1e-12
    band_fracs: List[float] = []

    for i in range(NUM_BANDS):
        r0 = band_edges[i]
        r1 = band_edges[i + 1]
        mask_band = (dist >= r0) & (dist < r1)
        band_energy = float(magnitude[mask_band].sum())
        band_fracs.append(band_energy / total_energy)

    half = NUM_BANDS // 2
    hf_ratio = float(sum(band_fracs[half:]))

    mag_log = np.log1p(magnitude)
    mag_log_norm = mag_log / (mag_log.max() + 1e-12)

    fft_mean = float(mag_log_norm.mean())
    fft_std = float(mag_log_norm.std())

    feats = [hf_ratio, fft_mean, fft_std] + band_fracs
    return np.array(feats, dtype=np.float32)


def load_images_from_dir(dir_path: str) -> List[str]:
    exts = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
    paths = []
    for root, _, files in os.walk(dir_path):
        for f in files:
            ext = os.path.splitext(f.lower())[1]
            if ext in exts:
                paths.append(os.path.join(root, f))
    return paths


def main():
    real_paths = load_images_from_dir(REAL_DIR)
    ai_paths = load_images_from_dir(AI_DIR)

    print(f"Found {len(real_paths)} real, {len(ai_paths)} ai.")

    if len(real_paths) == 0 or len(ai_paths) == 0:
        print("No data in one of the folders. Check REAL_DIR and AI_DIR.")
        return

    X = []
    y = []

    for p in real_paths:
        try:
            feats = compute_fft_features(p)
            X.append(feats)
            y.append(0)
        except Exception as e:
            print(f"[REAL] Error for {p}: {e}")

    for p in ai_paths:
        try:
            feats = compute_fft_features(p)
            X.append(feats)
            y.append(1)
        except Exception as e:
            print(f"[AI] Error for {p}: {e}")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    print(f"Features ready: X.shape = {X.shape}, y.shape = {y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    clf = LogisticRegression(
        solver="liblinear",
        random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)

    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy (test): {acc:.4f}")
    print(f"ROC AUC  (test): {auc:.4f}")

    intercept = float(clf.intercept_[0])
    coefs = list(map(float, clf.coef_[0]))

    coef_hf = coefs[0]
    coef_mean = coefs[1]
    coef_std = coefs[2]
    coef_bands = coefs[3:]

    print("\n== COEFFICIENTS TO PASTE INTO FFTMethod v2 ==")
    print(f"COEF_A    = {intercept:.9f}")
    print(f"COEF_HF   = {coef_hf:.9f}")
    print(f"COEF_MEAN = {coef_mean:.9f}")
    print(f"COEF_STD  = {coef_std:.9f}")
    for i, cb in enumerate(coef_bands):
        print(f"COEF_BAND_{i} = {cb:.9f}")

    print("\nCopy this into FFTMethod.analyze:")
    print("COEF_A    = ...")
    print("COEF_HF   = ...")
    print("COEF_MEAN = ...")
    print("COEF_STD  = ...")
    print("COEF_BANDS = [COEF_BAND_0, COEF_BAND_1, ..., COEF_BAND_7]")


if __name__ == "__main__":
    main()
