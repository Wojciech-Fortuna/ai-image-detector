import argparse
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Tuple, Dict, Optional

import numpy as np
import cv2
from PIL import Image
import pywt

# torch optional
try:
    import torch
except Exception:
    torch = None

STATE_FILE_NAME = ".incremental_zip_state.json"
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


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
    state_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


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


def extract_group(zf: zipfile.ZipFile, group_name: str, group_members: list, dest_root: Path) -> Path:
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


def read_list(path: Path) -> List[str]:
    if not path.exists():
        return []
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return [ln.strip() for ln in lines if ln.strip()]


def load_train_paths_for_group(group_name: str, work_dir: Path, splits_root: Path) -> List[str]:
    train_file = splits_root / group_name / "train.txt"
    if not train_file.exists():
        return []
    rels = read_list(train_file)
    out: List[str] = []
    for rel in rels:
        p = Path(rel)
        if p.is_absolute():
            if p.exists():
                out.append(str(p.resolve()))
        else:
            cand = (work_dir / p).resolve()
            if cand.exists():
                out.append(str(cand))
    return out


def probe_image_hw(path: str) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            w, h = im.size
            return int(h), int(w)
    except Exception:
        return None


def group_paths_by_native_hw(paths: List[str]) -> Dict[Tuple[int, int], List[str]]:
    buckets: Dict[Tuple[int, int], List[str]] = {}
    for p in paths:
        hw = probe_image_hw(p)
        if hw is None:
            continue
        buckets.setdefault(hw, []).append(p)
    return buckets


def imread_rgb_float(path: str) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        return arr


def center_crop_rgb(arr: np.ndarray, out_hw: Tuple[int, int]) -> np.ndarray:
    H, W, _ = arr.shape
    oh, ow = out_hw
    if H < oh or W < ow:
        raise ValueError(f"Image too small for crop: {H}x{W} < {oh}x{ow}")

    y0 = (H - oh) // 2
    x0 = (W - ow) // 2
    return arr[y0:y0 + oh, x0:x0 + ow, :]
    

def rgb_to_luma_y(rgb01: np.ndarray) -> np.ndarray:
    r = rgb01[..., 0]
    g = rgb01[..., 1]
    b = rgb01[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    return y.astype(np.float32)


def saturation_mask_cpu(img: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    return ((img > lo) & (img < hi)).astype(np.float32)


def local_zero_mean_cpu(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return img - cv2.blur(img, (ksize, ksize))


def noise_extract_gauss(img: np.ndarray, sigma: float = 2.0) -> np.ndarray:
    den = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma, sigmaY=sigma)
    return img - den


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


def load_didn_model(model_path: Path, device: str = "cuda"):
    if torch is None:
        raise RuntimeError("PyTorch not available. Install it to use --residual didn.")

    model_path = model_path.resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    dev = torch.device(device if (device.startswith("cuda") and torch.cuda.is_available()) else "cpu")

    # 1) TorchScript
    try:
        m = torch.jit.load(str(model_path), map_location=dev)
        m.eval()
        return m, dev, "torchscript"
    except Exception:
        pass

    # 2) state_dict
    try:
        from color_model import _NetG as DIDN
    except Exception as e:
        raise RuntimeError(
            "Checkpoint is not TorchScript and DIDN class is not importable.\n"
            "If color_model.pth is state_dict, add module with DIDN class\n"
            "or export checkpoint as TorchScript.\n"
            f"Details: {e}"
        )

    m = DIDN()
    ckpt = torch.load(str(model_path), map_location="cpu")
    sd = ckpt["state_dict"] if (isinstance(ckpt, dict) and "state_dict" in ckpt) else ckpt

    new_sd = {}
    for k, v in sd.items():
        nk = k[len("module."):] if k.startswith("module.") else k
        new_sd[nk] = v

    m.load_state_dict(new_sd, strict=False)
    m.to(dev)
    m.eval()
    return m, dev, "state_dict"


@torch.no_grad() if torch is not None else (lambda f: f)
def didn_residual_luma(model, device, rgb01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    if torch is None:
        raise RuntimeError("torch is required for DIDN residual")

    I_y = rgb_to_luma_y(rgb01)

    x = torch.from_numpy(rgb01.transpose(2, 0, 1)).unsqueeze(0).to(device=device, dtype=torch.float32)
    y = model(x)
    if isinstance(y, (list, tuple)):
        y = y[0]
    y = y.clamp(0.0, 1.0)
    den = y.squeeze(0).detach().cpu().numpy().transpose(1, 2, 0).astype(np.float32)

    W_rgb = rgb01.astype(np.float32) - den
    W_y = rgb_to_luma_y(W_rgb)
    return I_y.astype(np.float32), W_y.astype(np.float32)


def rsc_enhance(K: np.ndarray, wiener_eps: float = 1e-8) -> np.ndarray:
    K = K.astype(np.float32)
    K = K - K.mean(axis=1, keepdims=True)
    K = K - K.mean(axis=0, keepdims=True)

    F = np.fft.fft2(K)
    P = (np.abs(F) ** 2).astype(np.float32)

    N = float(np.median(P)) + wiener_eps
    S = np.maximum(P - N, 0.0)
    G = S / (S + N)

    Fw = F * G
    Kw = np.fft.ifft2(Fw).real.astype(np.float32)
    Kw -= float(Kw.mean())
    Kw /= float(Kw.std() + 1e-8)
    return Kw


def sea_enhance(K: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    K = K.astype(np.float32)
    H, W = K.shape

    F = np.fft.fftshift(np.fft.fft2(K))
    mag = np.abs(F)

    yy, xx = np.indices((H, W))
    cy, cx = H // 2, W // 2
    rr = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2).astype(np.int32)
    rr_max = rr.max()

    radial_sum = np.bincount(rr.ravel(), weights=mag.ravel(), minlength=rr_max + 1)
    radial_cnt = np.bincount(rr.ravel(), minlength=rr_max + 1)
    radial_avg = radial_sum / (radial_cnt + eps)

    eq = radial_avg[rr]
    F_eq = F / (eq + eps)

    K_eq = np.fft.ifft2(np.fft.ifftshift(F_eq)).real.astype(np.float32)
    K_eq -= float(K_eq.mean())
    K_eq /= float(K_eq.std() + 1e-8)
    return K_eq


def _mp_median(q: float, ngrid: int = 20000) -> float:
    q = float(q)
    if q <= 0:
        return 1.0
    a = (1.0 - np.sqrt(q)) ** 2
    b = (1.0 + np.sqrt(q)) ** 2
    xs = np.linspace(a, b, ngrid, dtype=np.float64)

    rad = np.maximum((b - xs) * (xs - a), 0.0)
    fx = (1.0 / (2.0 * np.pi * q * np.maximum(xs, 1e-12))) * np.sqrt(rad)

    cdf = np.cumsum(fx) * (xs[1] - xs[0])
    cdf /= cdf[-1]
    idx = int(np.searchsorted(cdf, 0.5))
    idx = max(0, min(idx, len(xs) - 1))
    return float(xs[idx])


def dc_enhance_decorrelation(K: np.ndarray, patch: int = 16, stride: int = 16) -> np.ndarray:
    K = K.astype(np.float32)
    H, W = K.shape
    ph = pw = int(patch)
    st = int(stride)
    if ph <= 0 or pw <= 0 or st <= 0:
        return K

    ys = list(range(0, H - ph + 1, st))
    xs = list(range(0, W - pw + 1, st))
    if not ys or not xs:
        return K

    patches = []
    for y in ys:
        for x in xs:
            patches.append(K[y:y + ph, x:x + pw].reshape(-1))
    X = np.asarray(patches, dtype=np.float32)
    n, d = X.shape
    if n < 4 or d < 4:
        return K

    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu

    try:
        _, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    except Exception:
        return K

    eigvals = (S.astype(np.float64) ** 2) / max(1.0, float(n - 1))
    q = float(d) / float(n) if n > 0 else 1.0
    q = min(max(q, 1e-6), 1.0)

    mp_med = _mp_median(q)
    med_eig = float(np.median(eigvals))
    sigma2 = med_eig / max(mp_med, 1e-12)
    lambda_plus = sigma2 * (1.0 + np.sqrt(q)) ** 2

    bad = np.where(eigvals > lambda_plus)[0]
    if bad.size == 0:
        out = K.copy()
        out -= float(out.mean())
        out /= float(out.std() + 1e-8)
        return out.astype(np.float32)

    V_bad = Vt[bad, :].astype(np.float32)
    proj = (Xc @ V_bad.T) @ V_bad
    X_clean = Xc - proj + mu

    out = np.zeros((H, W), dtype=np.float32)
    wgt = np.zeros((H, W), dtype=np.float32)
    idx = 0
    for y in ys:
        for x in xs:
            patch_vec = X_clean[idx].reshape(ph, pw)
            out[y:y + ph, x:x + pw] += patch_vec
            wgt[y:y + ph, x:x + pw] += 1.0
            idx += 1

    out = np.where(wgt > 0, out / np.maximum(wgt, 1e-12), K)
    out -= float(out.mean())
    out /= float(out.std() + 1e-8)
    return out.astype(np.float32)


def apply_base(K: np.ndarray, base: str, dc_patch: int = 16, dc_stride: int = 16) -> np.ndarray:
    base = base.lower()
    if base == "none":
        return K.astype(np.float32)
    if base == "rsc":
        return rsc_enhance(K)
    if base == "sea":
        return sea_enhance(K)
    if base == "dc":
        return dc_enhance_decorrelation(K, patch=dc_patch, stride=dc_stride)
    raise ValueError(f"Unknown base enhancement: {base}")


def guided_filter(I: np.ndarray, p: np.ndarray, r: int = 2, eps: float = 1e-2) -> np.ndarray:
    I = I.astype(np.float32)
    p = p.astype(np.float32)

    def mean_f(x):
        return cv2.boxFilter(x, ddepth=-1, ksize=(2 * r + 1, 2 * r + 1), normalize=True)

    mean_I = mean_f(I)
    mean_p = mean_f(p)
    mean_Ip = mean_f(I * p)
    mean_II = mean_f(I * I)

    cov_Ip = mean_Ip - mean_I * mean_p
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = mean_f(a)
    mean_b = mean_f(b)

    return mean_a * I + mean_b


def lowpass_fft_cycles_per_image(K: np.ndarray, cutoff: float) -> np.ndarray:
    K = K.astype(np.float32)
    H, W = K.shape

    Fy = np.fft.fftfreq(H) * H
    Fx = np.fft.fftfreq(W) * W
    fy, fx = np.meshgrid(Fy, Fx, indexing="ij")
    fr = np.sqrt(fx * fx + fy * fy)

    F = np.fft.fft2(K)
    mask = (fr <= float(cutoff)).astype(np.float32)
    return np.fft.ifft2(F * mask).real.astype(np.float32)


def paper7701_hf_enhance(
    K_raw: np.ndarray,
    cutoff_cpi: float,
    gf_diameter: int,
    gf_eps: float,
    lam: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    K_raw = K_raw.astype(np.float32)

    K_u10 = lowpass_fft_cycles_per_image(K_raw, cutoff=float(cutoff_cpi))
    r = max(1, (int(gf_diameter) - 1) // 2)
    K_low = guided_filter(I=K_u10, p=K_raw, r=r, eps=float(gf_eps)).astype(np.float32)

    K_high = (K_raw - K_low).astype(np.float32)
    K_adv = (float(lam) * K_high + K_raw).astype(np.float32)

    K_adv -= float(K_adv.mean())
    K_adv /= float(K_adv.std() + 1e-8)

    return K_u10, K_low, K_adv


def accumulate_chunk_gauss(paths: List[str], target_hw: Tuple[int, int], sigma: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    H, W = target_hw
    K_num = np.zeros((H, W), dtype=np.float32)
    K_den = np.zeros((H, W), dtype=np.float32)
    M_sum = np.zeros((H, W), dtype=np.float32)
    used = 0

    for p in paths:
        try:
            rgb = imread_rgb_float(p)
            rgb = center_crop_rgb(rgb, (H, W))
            I = rgb_to_luma_y(rgb)
        except Exception:
            continue

        M = saturation_mask_cpu(I)
        Il = local_zero_mean_cpu(I, ksize=3)
        Wn = noise_extract_gauss(Il, sigma=sigma)

        K_num += (Wn * I) * M
        K_den += (I * I) * M
        M_sum += M
        used += 1

    return K_num, K_den, M_sum, used


def accumulate_chunk_wavelet(
    paths: List[str],
    target_hw: Tuple[int, int],
    wavelet: str,
    level: int,
    mode: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    H, W = target_hw
    K_num = np.zeros((H, W), dtype=np.float32)
    K_den = np.zeros((H, W), dtype=np.float32)
    M_sum = np.zeros((H, W), dtype=np.float32)
    used = 0

    for p in paths:
        try:
            rgb = imread_rgb_float(p)
            rgb = center_crop_rgb(rgb, (H, W))
            I = rgb_to_luma_y(rgb)
        except Exception:
            continue

        M = saturation_mask_cpu(I)

        Il = local_zero_mean_cpu(I, ksize=3)

        Wn = noise_extract_wavelet(Il, wavelet=wavelet, level=level, mode=mode)

        K_num += (Wn * I) * M
        K_den += (I * I) * M
        M_sum += M
        used += 1

    return K_num, K_den, M_sum, used


def estimate_prnu_parallel_gauss(
    img_paths: List[str],
    target_hw: Tuple[int, int],
    sigma: float = 2.0,
    workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if not img_paths:
        raise ValueError("No images for PRNU.")

    H, W = target_hw

    chunks = [[] for _ in range(max(1, workers))]
    for i, p in enumerate(img_paths):
        chunks[i % len(chunks)].append(p)

    partials = []
    used_total = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(accumulate_chunk_gauss, ch, (H, W), sigma) for ch in chunks if ch]
            for f in as_completed(futs):
                kn, kd, ms, used = f.result()
                partials.append((kn, kd, ms))
                used_total += used
    else:
        kn, kd, ms, used = accumulate_chunk_gauss(img_paths, (H, W), sigma)
        partials.append((kn, kd, ms))
        used_total += used

    skipped_total = len(img_paths) - used_total

    K_num = np.zeros((H, W), dtype=np.float32)
    K_den = np.zeros((H, W), dtype=np.float32)
    M_sum = np.zeros((H, W), dtype=np.float32)
    for kn, kd, ms in partials:
        K_num += kn
        K_den += kd
        M_sum += ms

    K_raw = K_num / (K_den + 1e-8)
    thresh = max(1, int(0.2 * max(1, used_total)))
    M_valid = (M_sum >= thresh).astype(np.float32)

    return K_raw.astype(np.float32), M_valid, used_total, skipped_total


def estimate_prnu_serial_didn(
    img_paths: List[str],
    target_hw: Tuple[int, int],
    didn_model_path: Path,
    device_str: str,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    H, W = target_hw
    K_num = np.zeros((H, W), dtype=np.float32)
    K_den = np.zeros((H, W), dtype=np.float32)
    M_sum = np.zeros((H, W), dtype=np.float32)
    used = 0

    model, device, how = load_didn_model(didn_model_path, device=device_str)
    print(f"[DIDN] Loaded via {how} on device={device}")

    for p in img_paths:
        try:
            rgb = imread_rgb_float(p)
            rgb = center_crop_rgb(rgb, (H, W))
        except Exception:
            continue

        try:
            I, Wn = didn_residual_luma(model, device, rgb)
        except Exception:
            continue

        M = saturation_mask_cpu(I)
        K_num += (Wn * I) * M
        K_den += (I * I) * M
        M_sum += M
        used += 1

    skipped = len(img_paths) - used
    K_raw = K_num / (K_den + 1e-8)

    thresh = max(1, int(0.2 * max(1, used)))
    M_valid = (M_sum >= thresh).astype(np.float32)
    return K_raw.astype(np.float32), M_valid, used, skipped


def estimate_prnu_parallel_wavelet(
    img_paths: List[str],
    target_hw: Tuple[int, int],
    wavelet: str = "db8",
    level: int = 4,
    mode: str = "soft",
    workers: int = 4,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    if not img_paths:
        raise ValueError("No images for PRNU.")

    H, W = target_hw

    chunks = [[] for _ in range(max(1, workers))]
    for i, p in enumerate(img_paths):
        chunks[i % len(chunks)].append(p)

    partials = []
    used_total = 0

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = [
                ex.submit(accumulate_chunk_wavelet, ch, (H, W), wavelet, level, mode)
                for ch in chunks if ch
            ]
            for f in as_completed(futs):
                kn, kd, ms, used = f.result()
                partials.append((kn, kd, ms))
                used_total += used
    else:
        kn, kd, ms, used = accumulate_chunk_wavelet(img_paths, (H, W), wavelet, level, mode)
        partials.append((kn, kd, ms))
        used_total += used

    skipped_total = len(img_paths) - used_total

    K_num = np.zeros((H, W), dtype=np.float32)
    K_den = np.zeros((H, W), dtype=np.float32)
    M_sum = np.zeros((H, W), dtype=np.float32)
    for kn, kd, ms in partials:
        K_num += kn
        K_den += kd
        M_sum += ms

    K_raw = K_num / (K_den + 1e-8)
    thresh = max(1, int(0.2 * max(1, used_total)))
    M_valid = (M_sum >= thresh).astype(np.float32)

    return K_raw.astype(np.float32), M_valid, used_total, skipped_total


def do_prnu_and_save_variants(
    img_paths_for_reference: List[str],
    group_name_for_output: str,
    results_dir: Path,
    sigma: float,
    size: int,
    bases: List[str],
    cutoff_cpi: float,
    gf_diameter: int,
    gf_eps: float,
    lam: float,
    residual: str,
    workers: int,
    didn_model_path: Path,
    device: str,
    dc_patch: int,
    dc_stride: int,
    save_baseline: bool,
    native_hw: Tuple[int, int],
    wv: str = "db8",
    wv_level: int = 4,
    wv_mode: str = "soft",
) -> Tuple[bool, str]:
    if not img_paths_for_reference:
        return False, "No reference images."

    target_hw = (int(size), int(size))
    H0, W0 = int(native_hw[0]), int(native_hw[1])
    native_hw_arr = np.asarray([H0, W0], dtype=np.int32)

    try:
        if residual == "didn":
            K_raw, M, used, skipped = estimate_prnu_serial_didn(
                img_paths_for_reference,
                target_hw=target_hw,
                didn_model_path=didn_model_path,
                device_str=device,
            )
        elif residual == "wavelet":
            K_raw, M, used, skipped = estimate_prnu_parallel_wavelet(
                img_paths_for_reference,
                target_hw=target_hw,
                wavelet=wv,
                level=wv_level,
                mode=wv_mode,
                workers=workers,
            )
        else:
            K_raw, M, used, skipped = estimate_prnu_parallel_gauss(
                img_paths_for_reference,
                target_hw=target_hw,
                sigma=sigma,
                workers=workers,
            )
    except Exception as e:
        return False, f"PRNU estimation error: {e}"

    results_dir.mkdir(parents=True, exist_ok=True)
    msgs = [f"size={size} used={used} skipped={skipped} native={H0}x{W0}"]

    if save_baseline:
        base_out = results_dir / f"{group_name_for_output}__{size}__baseline_raw__{residual}.npz"
        np.savez_compressed(
            base_out,
            K=K_raw.astype(np.float32),
            M=M.astype(np.float32),
            native_hw=native_hw_arr,
        )
        msgs.append(f"baseline={base_out.name}")
    else:
        msgs.append("baseline=SKIPPED")

    for b in bases:
        try:
            K_base = apply_base(K_raw, b, dc_patch=dc_patch, dc_stride=dc_stride)

            _, _, K_adv = paper7701_hf_enhance(
                K_raw=K_base,
                cutoff_cpi=cutoff_cpi,
                gf_diameter=gf_diameter,
                gf_eps=gf_eps,
                lam=lam,
            )

            K_adv = (K_adv * M).astype(np.float32)

            out = results_dir / f"{group_name_for_output}__{size}__{b}_advHF__{residual}.npz"
            np.savez_compressed(
                out,
                K=K_adv.astype(np.float32),
                M=M.astype(np.float32),
                native_hw=native_hw_arr,
            )
            msgs.append(f"{b}_advHF={out.name}")
        except Exception as e:
            msgs.append(f"{b}_advHF_ERROR={e}")

    return True, " | ".join(msgs)


def parse_square_sizes(s: str) -> List[int]:
    out: List[int] = []
    for raw in (s or "").split(","):
        tok = raw.strip()
        if not tok:
            continue
        if not tok.isdigit():
            raise ValueError(f"Invalid --sizes token: '{raw}' (expected comma-separated ints, e.g. 128,256,512)")
        val = int(tok)
        if val <= 0:
            raise ValueError(f"Invalid crop size: {val}")
        out.append(val)
    # uniq + stable
    seen = set()
    uniq = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Incremental ZIP extractor + PRNU builder (paper7701 HF enhancement Eq.9-11).\n"
            "Reference images ALWAYS from <splits>/<group>/train.txt (relative to work_dir).\n"
            "NEW: If group has mixed native sizes, fingerprints are computed separately per native HxW bucket.\n"
            "Crop sizes logic matches 'imagine': --sizes is comma-separated INTS => square center crops."
        )
    )
    ap.add_argument("zip_path", type=Path, help="Ścieżka do archiwum ZIP")
    ap.add_argument("work_dir", type=Path, help="Katalog roboczy (tymczasowe wypakowanie)")
    ap.add_argument("--results", type=Path, default=Path("Results_Zip_Paper7701_DIDN"))

    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--root", action="store_true")
    ap.add_argument("--dry-run", action="store_true")

    ap.add_argument(
        "--splits",
        type=Path,
        required=True,
        help="Folder z podziałem (Results_Splits_...). Używamy <splits>/<group>/train.txt.",
    )

    ap.add_argument(
        "--sizes",
        type=str,
        default="128,256,512",
        help="Square crops (center), comma-separated ints. Default: 128,256,512",
    )
    ap.add_argument("--base", type=str, default="rsc,sea,dc",
                    help="Base enhancements: none,rsc,sea,dc (comma-separated)")

    ap.add_argument("--sigma", type=float, default=2.0)
    ap.add_argument("--workers", type=int, default=(os.cpu_count() or 4))

    ap.add_argument("--cutoff-hz", type=float, default=10.0,
                    help="Paper threshold (10 Hz) mapped here to cycles-per-image cutoff.")
    ap.add_argument("--gf-diameter", type=int, default=5)
    ap.add_argument("--gf-eps", type=float, default=0.01)
    ap.add_argument("--lambda", dest="lam", type=float, default=5.0,
                    help="Paper lambda in Eq.(11): K_adv = lambda*K_high + K_raw")

    ap.add_argument("--residual", type=str, default="didn",
                    choices=["didn", "gauss", "wavelet"],
                    help="Residual extractor: wavelet, gauss, didn.")
    ap.add_argument("--didn-model", type=Path, default=Path("color_model.pth"),
                    help="Path to DIDN RGB checkpoint (TorchScript or state_dict).")
    ap.add_argument("--device", type=str, default="cuda",
                    help="Device for DIDN: cuda or cpu")

    ap.add_argument("--dc-patch", type=int, default=16, help="DC patch size")
    ap.add_argument("--dc-stride", type=int, default=16, help="DC patch stride (16=non-overlapping)")

    ap.add_argument(
        "--save-baseline",
        action="store_true",
        help="If set, also save baseline_raw fingerprint file. Default: do NOT save baseline.",
    )

    ap.add_argument("--wv", type=str, default="db8", help="Wavelet name (pywt), e.g. db8,sym8")
    ap.add_argument("--wv-level", type=int, default=4, help="Wavelet decomposition level")
    ap.add_argument("--wv-mode", type=str, default="soft", choices=["soft", "hard"], help="Thresholding mode")

    args = ap.parse_args()

    zip_path = args.zip_path
    work_dir = args.work_dir.resolve()
    results_dir = args.results.resolve()
    splits_root = args.splits.resolve()

    if not zip_path.exists():
        print(f"[ERROR] ZIP not found: {zip_path}")
        sys.exit(1)
    if not splits_root.exists():
        print(f"[ERROR] --splits does not exist: {splits_root}")
        sys.exit(1)

    work_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    try:
        sizes = parse_square_sizes(args.sizes)
        if not sizes:
            raise ValueError("Empty --sizes list.")
    except Exception as e:
        print(f"[ERROR] --sizes parse error: {e}")
        sys.exit(1)

    bases = [b.strip().lower() for b in args.base.split(",") if b.strip()]
    for b in bases:
        if b not in {"none", "rsc", "sea", "dc"}:
            print(f"[ERROR] Unknown base: {b}")
            sys.exit(1)

    if args.residual == "didn":
        if torch is None:
            print("[ERROR] --residual didn requires PyTorch.")
            sys.exit(1)
        if args.workers != 1:
            print("[INFO] residual=DIDN -> forcing workers=1 (no multiprocessing).")
            args.workers = 1

    state_path = work_dir / STATE_FILE_NAME
    state = load_state(state_path) if args.resume else {}

    with zipfile.ZipFile(zip_path, "r") as zf:
        groups = list_top_level_groups(zf)
        order = filtered_order(groups, process_root=args.root)

        if args.dry_run:
            print("[DRY RUN] Grupy do przetworzenia:")
            for name in order:
                total = sum(m.file_size for m in groups[name])
                train_file = splits_root / name / "train.txt"
                train_ok = train_file.exists()
                print(f"  - {name:30s} {bytes_to_human(total)} ({len(groups[name])} plików) train.txt={'OK' if train_ok else 'MISSING'}")
            return

        already_done = set(state.get("done", []))
        to_process = [g for g in order if g not in already_done]
        if not to_process:
            print("Nothing to do. All selected groups already processed.")
            return

        print(f"Found {len(order)} groups; processing {len(to_process)}...")

        for group in to_process:
            members = groups[group]
            total_bytes = sum(m.file_size for m in members)
            print(f"\n==> Extracting group: {group} ({bytes_to_human(total_bytes)}, {len(members)} files)")

            try:
                target_dir = extract_group(zf, group, members, work_dir)
            except Exception as e:
                print(f"[ERROR] Extraction failed for {group}: {e}")
                already_done.add(group)
                save_state(state_path, {"done": sorted(list(already_done))})
                continue

            print(f"[OK] Extracted to: {target_dir}")

            # --- ALWAYS train.txt ---
            ref_paths = load_train_paths_for_group(group, work_dir, splits_root)
            if not ref_paths:
                print(f"[SKIP] No valid paths in: {splits_root / group / 'train.txt'}")
            else:
                buckets = group_paths_by_native_hw(ref_paths)
                if not buckets:
                    print("[SKIP] Could not read sizes of reference images (all unreadable?)")
                else:
                    bucket_keys = sorted(buckets.keys(), key=lambda x: (x[0], x[1]))

                    if len(bucket_keys) > 1:
                        print(
                            f"[INFO] Detected {len(bucket_keys)} native resolutions in group '{group}': "
                            + ", ".join([f"{h}x{w}({len(buckets[(h, w)])})" for (h, w) in bucket_keys])
                        )
                    else:
                        h0, w0 = bucket_keys[0]
                        print(f"[INFO] Native resolution in group '{group}': {h0}x{w0} ({len(buckets[(h0, w0)])} refs)")

                    for (H0, W0) in bucket_keys:
                        ref_hw_paths = buckets[(H0, W0)]
                        hw_tag = f"{H0}x{W0}"
                        group_out = f"{group}__{hw_tag}"

                        sizes_ok = [sz for sz in sizes if sz <= H0 and sz <= W0]
                        sizes_bad = [sz for sz in sizes if sz > H0 or sz > W0]
                        if sizes_bad:
                            print(f"[INFO] {group}/{hw_tag}: skipping crop sizes not fitting: {sizes_bad}")
                        if not sizes_ok:
                            print(f"[SKIP] {group}/{hw_tag}: no crop sizes fit into {H0}x{W0}")
                            continue

                        for sz in sizes_ok:
                            print(
                                f"[PRNU] group={group} native={hw_tag} crop={sz} "
                                f"refs={len(ref_hw_paths)} bases={bases} residual={args.residual} save_baseline={args.save_baseline}"
                            )
                            ok, msg = do_prnu_and_save_variants(
                                img_paths_for_reference=ref_hw_paths,
                                group_name_for_output=group_out,
                                results_dir=results_dir,
                                sigma=args.sigma,
                                size=sz,
                                bases=bases,
                                cutoff_cpi=args.cutoff_hz,
                                gf_diameter=args.gf_diameter,
                                gf_eps=args.gf_eps,
                                lam=args.lam,
                                residual=args.residual,
                                workers=args.workers,
                                didn_model_path=args.didn_model,
                                device=args.device,
                                dc_patch=args.dc_patch,
                                dc_stride=args.dc_stride,
                                save_baseline=bool(args.save_baseline),
                                native_hw=(H0, W0),
                                wv=args.wv,
                                wv_level=args.wv_level,
                                wv_mode=args.wv_mode,
                            )
                            if ok:
                                print(f"[OK] {msg}")
                            else:
                                print(f"[ERROR] {msg}")

            if not args.keep:
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
                    print(f"[WARN] Nie udało się usunąć {target_dir}: {e}")

            already_done.add(group)
            save_state(state_path, {"done": sorted(list(already_done))})

        print("\nDone.")


if __name__ == "__main__":
    main()
