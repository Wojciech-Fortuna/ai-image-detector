from __future__ import annotations

import base64
import io
import json
import math
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps
import joblib

from .base import BaseMethod, MethodResult, register


_MODEL_PATH = Path("models") / "ela_svc" / "ela_svc_model.joblib"
_PREPROCESS_PATH = Path("models") / "ela_svc" / "preprocess.json"

def _img_to_b64_png(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _prepare_rgb(img: Image.Image) -> Image.Image:
    img = ImageOps.exif_transpose(img)
    if img.mode in ("RGBA", "LA"):
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[-1])
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _compute_ela_map(base_rgb: Image.Image, quality: int, amplify: float) -> Image.Image:
    buf = io.BytesIO()
    base_rgb.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)

    jpg = Image.open(buf)
    jpg.load()
    jpg_rgb = _prepare_rgb(jpg)

    a = np.asarray(base_rgb).astype(np.int16)
    b = np.asarray(jpg_rgb).astype(np.int16)
    diff = np.abs(a - b).astype(np.float32) * float(amplify)
    diff = np.clip(diff, 0.0, 255.0)

    return Image.fromarray(diff.astype(np.uint8), mode="RGB")


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _ela_gray_from_model_map(ela_rgb_uint8: np.ndarray) -> np.ndarray:
    r = ela_rgb_uint8[..., 0].astype(np.float32)
    g = ela_rgb_uint8[..., 1].astype(np.float32)
    b = ela_rgb_uint8[..., 2].astype(np.float32)
    return 0.299 * r + 0.587 * g + 0.114 * b


def _energy_top1pct_frac(gray_255: np.ndarray) -> float:
    x = gray_255.reshape(-1).astype(np.float32)
    total = float(x.sum())
    if total <= 0.0:
        return 0.0

    n = x.size
    k = max(1, int(math.ceil(0.01 * n)))

    topk = np.partition(x, n - k)[n - k :]
    return float(topk.sum()) / total


def _hot_frac_t_adaptive(gray_255: np.ndarray) -> float:
    x = gray_255.reshape(-1).astype(np.float32)
    if x.size == 0:
        return 0.0
    t = float(np.percentile(x, 99))
    return float(np.mean(x >= t))


def _model_map_saturation_frac(ela_rgb_uint8: np.ndarray) -> float:
    sat = np.any(ela_rgb_uint8 >= 255, axis=2)
    return float(np.mean(sat))


@register
class ELAMethod(BaseMethod):
    name = "ela"
    description = "Detects editing or synthesis artifacts by analyzing JPEG compression inconsistencies."

    how_title = "ELA (Error Level Analysis)"
    how_text = (
        "Recompresses the image as JPEG and measures the pixel-wise difference (ELA map). "
        "High/structured error patterns can indicate edits or synthetic generation; "
        "the ELA map is resized/flattened and scored by an SVC (sigmoid of decision function)."
    )

    _model = None
    _pp: Optional[Dict[str, Any]] = None

    @classmethod
    def _lazy_load(cls) -> None:
        if cls._model is None:
            cls._model = joblib.load(_MODEL_PATH)

        if cls._pp is None:
            with open(_PREPROCESS_PATH, "r", encoding="utf-8") as f:
                cls._pp = json.load(f)

            for k in ("img_size", "ela_quality", "ela_amplify"):
                if k not in cls._pp:
                    raise ValueError(f"Missing '{k}' in preprocess.json")

    @classmethod
    def _features_from_image(cls, img: Image.Image) -> Tuple[np.ndarray, Image.Image, Dict[str, float]]:
        cls._lazy_load()
        assert cls._pp is not None

        img_size = int(cls._pp["img_size"])
        quality = int(cls._pp["ela_quality"])
        amplify = float(cls._pp["ela_amplify"])

        base = _prepare_rgb(img)

        ela_rgb = _compute_ela_map(base, quality=quality, amplify=amplify)

        ela_rgb_arr = np.asarray(ela_rgb, dtype=np.uint8)
        gray = _ela_gray_from_model_map(ela_rgb_arr)

        metrics: Dict[str, float] = {
            "ela_mean": float(gray.mean()),
            "ela_std": float(gray.std()),
            "ela_p90": float(np.percentile(gray, 90)),
            "ela_p99": float(np.percentile(gray, 99)),
            "ela_median": float(np.percentile(gray, 50)),
            "energy_top1pct_frac": float(_energy_top1pct_frac(gray)),
            "hot_frac_t_adaptive": float(_hot_frac_t_adaptive(gray)),
            "model_map_saturation_frac": float(_model_map_saturation_frac(ela_rgb_arr)),
        }

        ela_rgb_resized = ela_rgb.resize((img_size, img_size), Image.BILINEAR)
        ela01 = np.asarray(ela_rgb_resized).astype(np.float32) / 255.0
        feat = ela01.reshape(1, -1)

        return feat, ela_rgb, metrics

    def analyze(
        self,
        img: Image.Image,
        score_only: bool = False,
        **_: Any,
    ) -> MethodResult:
        try:
            feat, ela_vis, extra_metrics = self._features_from_image(img)
            model = self._model
            assert model is not None

            decision = float(model.decision_function(feat)[0])
            p_ai = float(_sigmoid(decision))
            pred = int(model.predict(feat)[0])

            metrics: Dict[str, float] = {
                **extra_metrics,
                "decision_score": decision,
                "pred": float(pred),
            }

            if score_only:
                return MethodResult(
                    name=self.name,
                    task="detection",
                    score=p_ai,
                    metrics={},
                    visuals_b64={},
                )

            return MethodResult(
                name=self.name,
                task="detection",
                score=p_ai,
                metrics=metrics,
                visuals_b64={
                    "compression_artifacts_map": _img_to_b64_png(ela_vis),
                },
            )

        except Exception as e:
            return MethodResult(
                name=self.name,
                task="detection",
                score=float("nan"),
                metrics={"error": 1.0, "error_msg": str(e)},
                visuals_b64={},
            )
