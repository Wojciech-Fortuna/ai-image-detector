import argparse
import csv
import os
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import cv2
from PIL import Image
import pywt

from concurrent.futures import ProcessPoolExecutor

VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def imread_rgb_float(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        return np.asarray(im, dtype=np.float32) / 255.0


def resize_rgb_float(rgb01: np.ndarray, target_hw: Tuple[int, int]) -> np.ndarray:
    th, tw = int(target_hw[0]), int(target_hw[1])
    h, w, _ = rgb01.shape
    if (h, w) == (th, tw):
        return rgb01
    out = cv2.resize(rgb01, (tw, th), interpolation=cv2.INTER_LINEAR)
    return out.astype(np.float32)


def center_crop_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    H, W, _ = arr.shape
    if H < size or W < size:
        raise ValueError(f"Image too small: {H}x{W} < {size}x{size}")
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    return arr[y0:y0 + size, x0:x0 + size, :]


def rgb_to_luma_y(rgb01: np.ndarray) -> np.ndarray:
    return (
        0.299 * rgb01[..., 0] + 0.587 * rgb01[..., 1] + 0.114 * rgb01[..., 2]
    ).astype(np.float32)


def saturation_mask_cpu(img: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    return ((img > lo) & (img < hi)).astype(np.float32)


def local_zero_mean_cpu(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return img - cv2.blur(img, (ksize, ksize))


def noise_extract_wavelet(
    img: np.ndarray,
    wavelet: str = "db8",
    level: int = 4,
    mode: str = "soft",
    sigma: float = None,
) -> np.ndarray:
    img = img.astype(np.float32)
    coeffs = pywt.wavedec2(img, wavelet=wavelet, level=level)
    cA, details = coeffs[0], coeffs[1:]

    if sigma is None:
        cH1, cV1, cD1 = details[-1]
        mad = np.median(np.abs(cD1 - np.median(cD1)))
        sigma = float(mad / 0.6745 + 1e-12)

    N = img.size
    thr = sigma * np.sqrt(2.0 * np.log(max(N, 2)))

    def thresh_detail(d):
        return pywt.threshold(d, value=thr, mode=mode).astype(np.float32)

    new_details = []
    for (cH, cV, cD) in details:
        new_details.append((thresh_detail(cH), thresh_detail(cV), thresh_detail(cD)))

    den = pywt.waverec2([cA] + new_details, wavelet=wavelet).astype(np.float32)
    den = den[: img.shape[0], : img.shape[1]]
    return (img - den).astype(np.float32)


def ncc_surface_fft_standard(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x = x - float(x.mean())
    y = y - float(y.mean())
    X = np.fft.fft2(x)
    Y = np.fft.fft2(y)
    corr = np.fft.ifft2(X * np.conj(Y)).real.astype(np.float32)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)) + eps)
    return (corr / denom).astype(np.float32)


def pce_from_corr(corr: np.ndarray, neigh: int = 11, eps: float = 1e-12):
    H, W = corr.shape
    idx = int(np.argmax(corr))
    py, px = divmod(idx, W)
    peak = float(corr[py, px])
    h = neigh // 2
    mask = np.ones_like(corr, dtype=bool)
    mask[max(0, py - h):py + h + 1, max(0, px - h):px + h + 1] = False
    energy = float(np.mean(corr[mask] ** 2) + eps)
    pce = float(np.sign(peak) * (peak * peak) / energy)
    return pce, peak, (py, px)


def list_images(path: Path) -> List[Path]:
    if path.is_file():
        return [path]
    out: List[Path] = []
    for r, _, files in os.walk(path):
        for f in files:
            if f.lower().endswith(VALID_EXT):
                out.append(Path(r) / f)
    return sorted(out)


def list_fingerprints(path: Path) -> List[Path]:
    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Fingerprint file must be .npz, got: {path}")
        return [path]
    fps = sorted([p for p in path.glob("*.npz") if p.is_file()])
    if not fps:
        raise RuntimeError(f"No .npz fingerprints found in: {path}")
    return fps


def load_fingerprint_npz(path: Path):
    data = np.load(path)
    if "K" not in data:
        raise ValueError(f"{path} missing 'K' array")
    K = data["K"].astype(np.float32)
    M = data["M"].astype(np.float32) if "M" in data else np.ones_like(K, dtype=np.float32)
    native_hw = None
    if "native_hw" in data:
        arr = np.asarray(data["native_hw"]).astype(np.int64).ravel()
        if arr.size >= 2:
            native_hw = (int(arr[0]), int(arr[1]))
    return K, M, native_hw


_FP_DATA = None
_ARGS = None
_CROP_SIZE = None


def _init_worker(fp_paths: List[str], args_dict: dict):
    global _FP_DATA, _ARGS, _CROP_SIZE

    _ARGS = args_dict

    fp_data = []
    with_native = 0
    for p in fp_paths:
        fp = Path(p)
        K, M, native_hw = load_fingerprint_npz(fp)
        if native_hw is not None:
            with_native += 1
        fp_data.append((fp, K, M, native_hw))

    K0, M0, _ = fp_data[0][1], fp_data[0][2], fp_data[0][3]
    if K0.ndim != 2 or K0.shape[0] != K0.shape[1]:
        raise ValueError(f"Fingerprint K must be square 2D, got shape={K0.shape}")
    for fp, K, M, _ in fp_data[1:]:
        if K.shape != K0.shape or M.shape != M0.shape:
            raise ValueError(
                "Fingerprint size mismatch:\n"
                f" first: K={K0.shape}, M={M0.shape}\n"
                f" {fp.name}: K={K.shape}, M={M.shape}\n"
                "All fingerprints must have identical shapes."
            )

    _FP_DATA = fp_data
    _CROP_SIZE = int(K0.shape[0])


def _score_image(ip_str: str):
    ip = Path(ip_str)
    try:
        rgb01_original = imread_rgb_float(str(ip))
    except Exception as e:
        return {"ok": False, "img_path": str(ip), "err": f"cannot read ({e})"}

    rgb_cache: Dict[Tuple[int, int], np.ndarray] = {}
    scores = []

    for fp, K, M_ref, native_hw in _FP_DATA:
        try:
            if native_hw is not None:
                if native_hw not in rgb_cache:
                    rgb_cache[native_hw] = resize_rgb_float(rgb01_original, native_hw)
                rgb01 = rgb_cache[native_hw]
            else:
                rgb01 = rgb01_original

            rgb = center_crop_rgb(rgb01, _CROP_SIZE)

            I = rgb_to_luma_y(rgb).astype(np.float32)

            Il = local_zero_mean_cpu(I, ksize=3)
            Wn = noise_extract_wavelet(
                Il,
                wavelet=_ARGS["wavelet"],
                level=_ARGS["wavelet_level"],
                mode=_ARGS["wavelet_mode"],
                sigma=None,
            )

            M = (M_ref * saturation_mask_cpu(I)).astype(np.float32)
            cov = float(M.mean())

            if cov < float(_ARGS["cov_floor"]):
                pce, peak, py, px = 0.0, 0.0, 0, 0
            else:
                X = (Wn * M).astype(np.float32)
                xstd = float(X.std())

                if xstd < float(_ARGS["xstd_floor"]):
                    pce, peak, py, px = 0.0, 0.0, 0, 0
                else:
                    Y = (I * K * M).astype(np.float32)
                    corr = ncc_surface_fft_standard(X, Y)
                    pce, peak, (py, px) = pce_from_corr(corr, neigh=_ARGS["pce_neigh"])

            scores.append((fp, float(pce), float(peak), int(py), int(px), float(cov)))
        except Exception:
            continue

    if not scores:
        return {"ok": False, "img_path": str(ip), "err": "no scores (all failed?)"}

    scores_sorted = sorted(scores, key=lambda t: t[1], reverse=True)

    rows = []
    for rank_idx, (fp, pce, peak, py, px, cov) in enumerate(scores_sorted, 1):
        rows.append([
            str(ip),
            fp.stem,
            f"{pce:.10g}",
            rank_idx,
            f"{peak:.10g}",
            py,
            px,
            f"{cov:.6g}",
        ])

    best_fp, best_pce, best_peak, _, _, best_cov = scores_sorted[0]
    return {
        "ok": True,
        "img_path": str(ip),
        "rows": rows,
        "best": (best_fp.stem, best_pce, best_peak, best_cov),
    }


def main():
    ap = argparse.ArgumentParser(
        "PRNU batch matching (wavelet only) - standard NCC-surface + PCE + multiprocessing"
    )
    ap.add_argument("--fingerprints-dir", type=Path, required=True,
                    help="Folder with .npz fingerprints or a single .npz file")
    ap.add_argument("--images", type=Path, required=True,
                    help="Folder with images (recursive) or a single file")
    ap.add_argument("--out", type=Path, required=True,
                    help="Output CSV: img_path,fingerprint,pce,rank,...")

    ap.add_argument("--pce-neigh", type=int, default=11)
    ap.add_argument("--cov-floor", type=float, default=0.01,
                    help="Minimum mean mask coverage M, otherwise score=0")

    ap.add_argument("--xstd-floor", type=float, default=0.001,
                    help="If std(X)=std(Wn*M) < xstd_floor then PCE is zeroed (after passing cov-floor).")

    ap.add_argument("--wavelet", type=str, default="db8")
    ap.add_argument("--wavelet-level", type=int, default=4)
    ap.add_argument("--wavelet-mode", type=str, default="soft", choices=["soft", "hard"])

    ap.add_argument("--progress-every", type=int, default=25,
                    help="Print progress every N images (0=never)")

    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 1)),
                    help="How many parallel processes (default: cpu_count)")

    ap.add_argument("--chunksize", type=int, default=1,
                    help="Chunksize for multiprocessing (>=1)")

    args = ap.parse_args()

    fingerprints = list_fingerprints(args.fingerprints_dir)
    images = list_images(args.images)
    if not images:
        raise RuntimeError(f"No images found in: {args.images}")

    K0, M0, _ = load_fingerprint_npz(fingerprints[0])
    if K0.ndim != 2 or K0.shape[0] != K0.shape[1]:
        raise ValueError(f"Fingerprint K must be square 2D, got shape={K0.shape}")
    for fp in fingerprints[1:]:
        Kt, Mt, _ = load_fingerprint_npz(fp)
        if Kt.shape != K0.shape or Mt.shape != M0.shape:
            raise ValueError(
                "Fingerprint size mismatch:\n"
                f" {fingerprints[0].name}: K={K0.shape}, M={M0.shape}\n"
                f" {fp.name}: K={Kt.shape}, M={Mt.shape}\n"
                "All fingerprints must have identical shapes."
            )

    with_native = 0
    for fp in fingerprints:
        data = np.load(fp)
        if "native_hw" in data:
            arr = np.asarray(data["native_hw"]).astype(np.int64).ravel()
            if arr.size >= 2:
                with_native += 1

    print(f"[INFO] Images: {len(images)} | Fingerprints: {len(fingerprints)} | Crop: {K0.shape[0]}x{K0.shape[0]}")
    print(f"[INFO] residual=wavelet use_I=True cov_floor={args.cov_floor} xstd_floor={args.xstd_floor} pce_neigh={args.pce_neigh}")
    print(f"[INFO] Fingerprints with native_hw: {with_native}/{len(fingerprints)}")
    print(f"[INFO] workers={args.workers} chunksize={args.chunksize}")

    args.out.parent.mkdir(parents=True, exist_ok=True)

    args_dict = {
        "pce_neigh": int(args.pce_neigh),
        "cov_floor": float(args.cov_floor),
        "xstd_floor": float(args.xstd_floor),  # >>> MOD
        "wavelet": str(args.wavelet),
        "wavelet_level": int(args.wavelet_level),
        "wavelet_mode": str(args.wavelet_mode),
    }

    fp_paths = [str(p) for p in fingerprints]
    img_paths = [str(p) for p in images]

    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["img_path", "fingerprint", "pce", "rank", "peak", "peak_y", "peak_x", "cov"])

        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=_init_worker,
            initargs=(fp_paths, args_dict),
        ) as ex:
            for i, res in enumerate(ex.map(_score_image, img_paths, chunksize=max(1, int(args.chunksize))), 1):
                if not res.get("ok", False):
                    err = res.get("err", "unknown error")
                    print(f"[SKIP] {res.get('img_path', '?')}: {err}")
                    continue

                for row in res["rows"]:
                    w.writerow(row)

                if args.progress_every > 0 and (i % args.progress_every == 0 or i == len(images)):
                    best_fp, best_pce, best_peak, best_cov = res["best"]
                    ip_name = Path(res["img_path"]).name
                    print(f"[INFO] {i}/{len(images)}: {ip_name} best={best_fp} pce={best_pce:.4g} peak={best_peak:.4g} cov={best_cov:.3f}")

    print(f"\nDone → {args.out}")


if __name__ == "__main__":
    main()
