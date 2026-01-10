from __future__ import annotations

import base64
import io
import math
from typing import Dict, Any, List

import numpy as np
from PIL import Image, ImageOps
from .base import BaseMethod, MethodResult, register


def _img_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _prepare_gray(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    return img.convert("L")


@register
class FFTMethod(BaseMethod):
    name = "fft"
    description = "Looks for unusual frequency patterns often found in AI-generated images."

    how_title = "FFT (Frequency Analysis)"
    how_text = (
        "The image is analyzed in the frequency domain to measure how visual energy is distributed "
        "across low and high frequencies. AI-generated images often show characteristic frequency "
        "patterns that differ from natural photographs."
    )

    def analyze(
        self,
        img: Image.Image,
        score_only: bool = False,
        **_: Any,
    ) -> MethodResult:
        TARGET_SIZE = 512

        NUM_BANDS = 8

        COEF_A = -0.223128569
        COEF_HF = -2.189326782
        COEF_MEAN = 0.654943422
        COEF_STD = 1.233249336
        COEF_BANDS: List[float] = [
            1.145075448,
            2.459305114,
            0.613219691,
            -2.251596373,
            -2.510267124,
            -0.497114361,
            0.530085277,
            0.287969487,
        ]

        try:
            gray = _prepare_gray(img)

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

            z = (
                COEF_A
                + COEF_HF * hf_ratio
                + COEF_MEAN * fft_mean
                + COEF_STD * fft_std
            )
            for w_band, band_val in zip(COEF_BANDS, band_fracs):
                z += w_band * float(band_val)

            p_ai = 1.0 / (1.0 + math.exp(-z))

            if score_only:
                return MethodResult(
                    name=self.name,
                    task="detection",
                    score=float(p_ai),
                    metrics={},
                    visuals_b64={},
                )

            vis = Image.fromarray(
                (mag_log_norm * 255.0).astype(np.uint8)
            ).convert("L")

            metrics: Dict[str, float] = {
                "hf_ratio": float(hf_ratio),
                "fft_mean": float(fft_mean),
                "fft_std": float(fft_std),
                "logit_z": float(z),
                "target_size": float(TARGET_SIZE),
                "num_bands": float(NUM_BANDS),
            }

            for i, v in enumerate(band_fracs):
                metrics[f"band_{i}"] = float(v)

            metrics.update(
                {
                    "coef_a": float(COEF_A),
                    "coef_hf": float(COEF_HF),
                    "coef_mean": float(COEF_MEAN),
                    "coef_std": float(COEF_STD),
                }
            )
            for i, w_band in enumerate(COEF_BANDS):
                metrics[f"coef_band_{i}"] = float(w_band)

            visuals = {"spectrum": _img_to_b64_png(vis)}

            return MethodResult(
                name=self.name,
                task="detection",
                score=float(p_ai),
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
