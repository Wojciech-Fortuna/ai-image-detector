from __future__ import annotations

import base64
import io
from pathlib import Path
from typing import Dict, Tuple, Any, Optional

import numpy as np
import cv2
from PIL import Image
import pywt

from .base import BaseMethod, MethodResult, register


FINGERPRINTS_DIR = Path("models") / "fingerprints_1024"

WAVELET = "db8"
WAVELET_LEVEL = 4
WAVELET_MODE = "soft"

PCE_NEIGH = 11
COV_FLOOR = 0.01
XSTD_FLOOR = 0.001

PCE_THRESHOLD = 67.582400


def _img_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _pil_to_rgb_u8(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("RGB"), dtype=np.uint8)


def _center_crop_2d(arr: np.ndarray, size: int) -> np.ndarray:
    H, W = arr.shape
    if H < size or W < size:
        raise ValueError(f"Image too small: {H}x{W} < {size}x{size}")
    y0 = (H - size) // 2
    x0 = (W - size) // 2
    return arr[y0:y0 + size, x0:x0 + size]


def _luma_center_crop_from_rgb_u8(
    rgb_u8: np.ndarray,
    crop_size: int,
    native_hw: Optional[Tuple[int, int]],
) -> np.ndarray:
    crop_size = int(crop_size)

    if native_hw is None:
        r = _center_crop_2d(rgb_u8[..., 0], crop_size).astype(np.float32) / 255.0
        I = 0.299 * r
        g = _center_crop_2d(rgb_u8[..., 1], crop_size).astype(np.float32) / 255.0
        I += 0.587 * g
        b = _center_crop_2d(rgb_u8[..., 2], crop_size).astype(np.float32) / 255.0
        I += 0.114 * b
        return I.astype(np.float32)

    th, tw = int(native_hw[0]), int(native_hw[1])
    if th < crop_size or tw < crop_size:
        raise ValueError(f"native_hw too small: {th}x{tw} < {crop_size}x{crop_size}")

    y0 = (th - crop_size) // 2
    x0 = (tw - crop_size) // 2
    y1 = y0 + crop_size
    x1 = x0 + crop_size

    I = np.zeros((crop_size, crop_size), dtype=np.float32)

    # Channel 0 (R)
    ch0 = rgb_u8[..., 0].astype(np.float32) / 255.0
    ch0r = cv2.resize(ch0, (tw, th), interpolation=cv2.INTER_LINEAR)
    I += 0.299 * ch0r[y0:y1, x0:x1]
    del ch0, ch0r

    # Channel 1 (G)
    ch1 = rgb_u8[..., 1].astype(np.float32) / 255.0
    ch1r = cv2.resize(ch1, (tw, th), interpolation=cv2.INTER_LINEAR)
    I += 0.587 * ch1r[y0:y1, x0:x1]
    del ch1, ch1r

    # Channel 2 (B)
    ch2 = rgb_u8[..., 2].astype(np.float32) / 255.0
    ch2r = cv2.resize(ch2, (tw, th), interpolation=cv2.INTER_LINEAR)
    I += 0.114 * ch2r[y0:y1, x0:x1]
    del ch2, ch2r

    return I.astype(np.float32)


def _saturation_mask_cpu(img: np.ndarray, lo: float = 0.05, hi: float = 0.95) -> np.ndarray:
    return ((img > lo) & (img < hi)).astype(np.float32)


def _local_zero_mean_cpu(img: np.ndarray, ksize: int = 3) -> np.ndarray:
    return img - cv2.blur(img, (ksize, ksize))


def _noise_extract_wavelet(
    img: np.ndarray,
    wavelet: str = WAVELET,
    level: int = WAVELET_LEVEL,
    mode: str = WAVELET_MODE,
    sigma: Optional[float] = None,
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


def _ncc_surface_fft_standard(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    x = x.astype(np.float32)
    y = y.astype(np.float32)
    x = x - float(x.mean())
    y = y - float(y.mean())

    X = np.fft.fft2(x)
    Y = np.fft.fft2(y)
    corr = np.fft.ifft2(X * np.conj(Y)).real.astype(np.float32)
    denom = float(np.sqrt(np.sum(x * x) * np.sum(y * y)) + eps)
    return (corr / denom).astype(np.float32)


def _pce_from_corr(
    corr: np.ndarray,
    neigh: int = PCE_NEIGH,
    eps: float = 1e-12
) -> Tuple[float, float, Tuple[int, int], float]:
    H, W = corr.shape
    idx = int(np.argmax(corr))
    py, px = divmod(idx, W)
    peak = float(corr[py, px])

    h = neigh // 2
    mask = np.ones_like(corr, dtype=bool)
    mask[max(0, py - h):py + h + 1, max(0, px - h):px + h + 1] = False
    energy = float(np.mean(corr[mask] ** 2) + eps)
    pce = float(np.sign(peak) * (peak * peak) / energy)
    return pce, peak, (py, px), energy


def _circular_peak_offsets(py: int, px: int, H: int, W: int) -> Tuple[float, float, float]:
    dr = float(py if py < H / 2 else py - H)
    dc = float(px if px < W / 2 else px - W)
    return dr, dc, float((dr * dr + dc * dc) ** 0.5)


def _load_fingerprints(
    results_dir: Path
) -> Dict[str, Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]]:
    fps: Dict[str, Tuple[np.ndarray, np.ndarray, Optional[Tuple[int, int]]]] = {}
    paths = sorted(results_dir.glob("*.npz"))
    if not paths:
        return fps

    ref_shape: Optional[Tuple[int, int]] = None

    for p in paths:
        try:
            data = np.load(p)
        except Exception:
            continue

        if "K" not in data:
            continue

        K = data["K"].astype(np.float32)
        if "M" in data:
            M = data["M"].astype(np.float32)
        else:
            M = np.ones_like(K, dtype=np.float32)

        if K.ndim != 2 or K.shape[0] != K.shape[1]:
            continue
        if M.shape != K.shape:
            continue

        if ref_shape is None:
            ref_shape = K.shape
        else:
            if K.shape != ref_shape:
                continue

        native_hw = None
        if "native_hw" in data:
            arr = np.asarray(data["native_hw"]).astype(np.int64).ravel()
            if arr.size >= 2:
                native_hw = (int(arr[0]), int(arr[1]))

        fps[p.stem] = (K, M, native_hw)

    return fps


def _make_corr_map_visual(C: np.ndarray) -> Image.Image:
    C = C.astype(np.float32)
    C_disp = np.fft.fftshift(C)

    A = np.log1p(np.abs(C_disp))

    p1 = float(np.percentile(A, 1.0))
    p99 = float(np.percentile(A, 99.0))
    if p99 - p1 < 1e-12:
        gray = np.zeros_like(A, dtype=np.uint8)
    else:
        A = np.clip(A, p1, p99)
        norm = (A - p1) / (p99 - p1)
        gray = (norm * 255.0).astype(np.uint8)

    return Image.fromarray(gray, mode="L").convert("RGB")


@register
class PRNUMethod(BaseMethod):
    name = "prnu"
    description = "Checks for a camera-specific sensor fingerprint (PRNU) to verify whether an image likely comes from a real camera."

    how_title = "PRNU (Camera Sensor Fingerprint)"
    how_text = (
        "Real cameras leave a subtle, device-specific sensor pattern in photos, known as PRNU. "
        "Each camera model (and often each individual device) has its own unique fingerprint. "
        "This method compares the image against known camera fingerprints; a strong match supports "
        "a real-camera origin. A weak or missing match does NOT mean the image is AI-generated — "
        "it simply means the origin cannot be determined using PRNU alone."
    )

    def analyze(
        self,
        img: Image.Image,
        score_only: bool = False,
        **kwargs: Any,
    ) -> MethodResult:
        results_dir = Path(FINGERPRINTS_DIR).resolve()

        try:
            if not results_dir.exists():
                raise FileNotFoundError(f"Fingerprints directory does not exist: {results_dir}")

            fp_dict = _load_fingerprints(results_dir)
            if not fp_dict:
                raise RuntimeError(f"No valid fingerprints in directory: {results_dir}")

            rgb_u8_original = _pil_to_rgb_u8(img)
            H0, W0, _ = rgb_u8_original.shape

            def _fp_area(name: str) -> int:
                _K, _M_ref, _native_hw = fp_dict[name]
                if _native_hw is not None:
                    return int(_native_hw[0]) * int(_native_hw[1])
                return int(H0) * int(W0)

            fpnames = sorted(fp_dict.keys(), key=lambda n: (_fp_area(n), n))

            last_hw: Optional[Tuple[int, int]] = None
            last_crop_size: Optional[int] = None
            last_I: Optional[np.ndarray] = None

            best_name: Optional[str] = None
            best_pce: Optional[float] = None

            best_corr: Optional[np.ndarray] = None
            best_peak_pos: Tuple[int, int] = (0, 0)
            best_peak_val: float = 0.0
            best_noise_energy: float = 0.0

            best_cov: float = 0.0
            best_xstd: float = 0.0
            best_gate_cov_failed: int = 0
            best_gate_xstd_failed: int = 0

            best_res_mean: float = 0.0
            best_res_std: float = 0.0

            best_mask_ref_cov: float = 0.0
            best_mask_test_cov: float = 0.0

            best_native_hw_used: Optional[Tuple[int, int]] = None
            best_crop_size: int = 0

            for name in fpnames:
                K, M_ref, native_hw = fp_dict[name]
                crop_size = int(K.shape[0])

                if (native_hw == last_hw) and (crop_size == last_crop_size) and (last_I is not None):
                    I = last_I
                    native_hw_used = native_hw
                else:
                    I = _luma_center_crop_from_rgb_u8(
                        rgb_u8_original, crop_size=crop_size, native_hw=native_hw
                    )
                    last_hw = native_hw
                    last_crop_size = crop_size
                    last_I = I
                    native_hw_used = native_hw

                Il = _local_zero_mean_cpu(I, ksize=3)

                Wn = _noise_extract_wavelet(
                    Il,
                    wavelet=WAVELET,
                    level=WAVELET_LEVEL,
                    mode=WAVELET_MODE,
                    sigma=None,
                )

                M_test = _saturation_mask_cpu(I).astype(np.float32)
                M = (M_ref * M_test).astype(np.float32)

                cov = float(M.mean())

                gate_cov_failed = 0
                gate_xstd_failed = 0
                xstd = 0.0

                if cov < float(COV_FLOOR):
                    gate_cov_failed = 1
                    pce = 0.0
                    peak_val = 0.0
                    peak_pos = (0, 0)
                    noise_energy = 0.0
                    corr = None
                else:
                    X = (Wn * M).astype(np.float32)
                    xstd = float(X.std())

                    if xstd < float(XSTD_FLOOR):
                        gate_xstd_failed = 1
                        pce = 0.0
                        peak_val = 0.0
                        peak_pos = (0, 0)
                        noise_energy = 0.0
                        corr = None
                    else:
                        Y = (I * K * M).astype(np.float32)
                        corr = _ncc_surface_fft_standard(X, Y)
                        pce, peak_val, peak_pos, noise_energy = _pce_from_corr(corr, neigh=PCE_NEIGH)

                if best_pce is None or pce > best_pce:
                    best_pce = float(pce)
                    best_name = name

                    best_corr = corr
                    best_peak_pos = peak_pos
                    best_peak_val = float(peak_val)
                    best_noise_energy = float(noise_energy)

                    best_cov = float(cov)
                    best_xstd = float(xstd)
                    best_gate_cov_failed = int(gate_cov_failed)
                    best_gate_xstd_failed = int(gate_xstd_failed)

                    best_res_mean = float(Wn.mean())
                    best_res_std = float(Wn.std())
                    best_mask_ref_cov = float((M_ref > 0.5).mean())
                    best_mask_test_cov = float((M_test > 0.5).mean())

                    best_native_hw_used = native_hw_used
                    best_crop_size = int(crop_size)

            if best_pce is None or best_name is None:
                raise RuntimeError("No PCE values computed for PRNU method.")

            score_val = 0.1 if best_pce >= PCE_THRESHOLD else float("nan")

            if score_only:
                return MethodResult(
                    name=self.name,
                    task="detection",
                    score=score_val,
                    metrics={},
                    visuals_b64={},
                )

            visuals: Dict[str, str] = {}
            if best_corr is not None:
                H, W = best_corr.shape
                py, px = best_peak_pos
                dr, dc, dn = _circular_peak_offsets(py, px, H, W)
                corr_img = _make_corr_map_visual(best_corr)
                visuals["corr_map"] = _img_to_b64_png(corr_img)
            else:
                H = W = best_crop_size if best_crop_size > 0 else 256
                dummy = np.zeros((H, W), dtype=np.float32)
                corr_img = _make_corr_map_visual(dummy)
                visuals["corr_map"] = _img_to_b64_png(corr_img)
                dr = dc = dn = 0.0

            metrics: Dict[str, Any] = {
                "pce": float(best_pce),
                "peak": float(best_peak_val),
                "noise_energy": float(best_noise_energy),
                "cov": float(best_cov),
                "xstd": float(best_xstd),
                "gate_cov_failed": int(best_gate_cov_failed),
                "gate_xstd_failed": int(best_gate_xstd_failed),
                "peak_offset_r": float(dr),
                "peak_offset_c": float(dc),
                "peak_offset_norm": float(dn),
                "mask_ref_coverage": float(best_mask_ref_cov),
                "mask_test_coverage": float(best_mask_test_cov),
                "res_mean": float(best_res_mean),
                "res_std": float(best_res_std),
                "best_fingerprint_name": (None if float(best_pce) == 0.0 else best_name),
                "native_hw_used": list(best_native_hw_used) if best_native_hw_used is not None else None,
                "crop_size": int(best_crop_size),
                "pce_threshold": float(PCE_THRESHOLD),
                "wavelet": str(WAVELET),
                "wavelet_level": int(WAVELET_LEVEL),
                "wavelet_mode": str(WAVELET_MODE),
                "pce_neigh": int(PCE_NEIGH),
                "cov_floor": float(COV_FLOOR),
                "xstd_floor": float(XSTD_FLOOR),
            }

            return MethodResult(
                name=self.name,
                task="detection",
                score=score_val,
                metrics=metrics,
                visuals_b64=visuals,
            )

        except Exception as e:
            return MethodResult(
                name=self.name,
                task="detection",
                score=float("nan"),
                metrics={"error": 1.0, "error_msg": str(e)},
                visuals_b64={},
            )
