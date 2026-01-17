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

import torch
import open_clip

from .base import BaseMethod, MethodResult, register


_MODEL_PATH = Path("models") / "clip_svm" / "v1.pkl"
_PREPROCESS_PATH = Path("models") / "clip_svm" / "preprocess.json"

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


def _sigmoid(z: float) -> float:
    if z >= 0:
        ez = math.exp(-z)
        return 1.0 / (1.0 + ez)
    ez = math.exp(z)
    return ez / (1.0 + ez)


def _l2_normalize(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / (n + eps)


def _get_vit_patch_embeddings_openclip(clip_model, image_input: torch.Tensor):
    visual = clip_model.visual

    if image_input.dtype != visual.conv1.weight.dtype:
        image_input = image_input.to(dtype=visual.conv1.weight.dtype)

    x = visual.conv1(image_input)              # [B, C, H', W']
    x = x.reshape(x.shape[0], x.shape[1], -1)  # [B, C, HW]
    x = x.permute(0, 2, 1)                     # [B, HW, C]

    cls = visual.class_embedding.to(x.dtype)
    cls = cls + torch.zeros(x.shape[0], 1, x.shape[2], dtype=x.dtype, device=x.device)
    x = torch.cat([cls, x], dim=1)             # [B, 1+HW, C]

    x = x + visual.positional_embedding.to(x.dtype)
    x = visual.ln_pre(x)

    x = x.permute(1, 0, 2)                     # [1+HW, B, C]
    x = visual.transformer(x)
    x = x.permute(1, 0, 2)                     # [B, 1+HW, C]

    x = visual.ln_post(x)

    if hasattr(visual, "proj") and visual.proj is not None:
        x = x @ visual.proj

    patch = x[:, 1:, :]                        # [B, HW, D]

    n = patch.shape[1]
    side = int(math.sqrt(n))
    if side * side == n:
        H = W = side
    else:
        H, W = 1, n

    patch = patch / (patch.norm(dim=-1, keepdim=True) + 1e-8)
    return patch, H, W


def _svm_patch_heatmap_overlay(
    img: Image.Image,
    clip_model,
    clip_preprocess,
    device: str,
    svm,
    alpha: float = 0.5,
) -> Tuple[Image.Image, Dict[str, float]]:
    base = _prepare_rgb(img)
    image_input = clip_preprocess(base).unsqueeze(0).to(device)

    with torch.inference_mode():
        patch_emb, H_p, W_p = _get_vit_patch_embeddings_openclip(clip_model, image_input)

    if not hasattr(svm, "coef_"):
        raise ValueError("SVM has no coef_. Heatmap requires a linear model (e.g. LinearSVC).")

    w = svm.coef_.reshape(-1).astype(np.float32)  # [D]
    w_t = torch.from_numpy(w).to(device)

    with torch.inference_mode():
        contrib = torch.einsum("bnd,d->bn", patch_emb, w_t)[0]  # [HW]

    contrib_np = contrib.detach().float().cpu().numpy().reshape(H_p, W_p)
    c_min, c_max = float(contrib_np.min()), float(contrib_np.max())

    if c_max - c_min < 1e-8:
        norm = np.zeros_like(contrib_np, dtype=np.float32)
    else:
        norm = (contrib_np - c_min) / (c_max - c_min)  # 0..1

    heat_small = (norm * 255).astype(np.uint8)
    heat_img = Image.fromarray(heat_small, "L").resize(base.size, Image.BILINEAR)

    arr = np.array(heat_img, dtype=np.float32) / 255.0
    rgb = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    rgb[..., 0] = (arr * 255).astype(np.uint8)          # red
    rgb[..., 2] = ((1.0 - arr) * 255).astype(np.uint8)  # blue

    heat_rgb = Image.fromarray(rgb, "RGB")
    overlay = Image.blend(base.convert("RGB"), heat_rgb, alpha=float(alpha))

    metrics = {
        "patch_contrib_min": c_min,
        "patch_contrib_max": c_max,
        "patch_contrib_mean": float(contrib_np.mean()),
        "patch_contrib_std": float(contrib_np.std()),
        "patch_h": float(H_p),
        "patch_w": float(W_p),
    }
    return overlay, metrics


@register
class CLIPSVMMethod(BaseMethod):
    name = "clip_svm"
    description = "Detects AI-generated images using CLIP image embeddings + a trained linear SVM."

    how_title = "CLIP + Linear SVM"
    how_text = (
        "Encodes the image with a CLIP vision model and scores it using a pretrained linear SVM. "
    )

    _svm = None
    _pp: Optional[Dict[str, Any]] = None
    _clip_model = None
    _clip_preprocess = None
    _device: Optional[str] = None

    @classmethod
    def _lazy_load(cls) -> None:
        if cls._pp is None:
            with open(_PREPROCESS_PATH, "r", encoding="utf-8") as f:
                cls._pp = json.load(f)

            cls._pp.setdefault("clip_model", "hf-hub:laion/CLIP-ViT-L-14-DataComp.XL-s13B-b90K")
            cls._pp.setdefault("device", "auto")
            cls._pp.setdefault("normalize_embedding", True)

        assert cls._pp is not None

        if cls._device is None:
            d = str(cls._pp.get("device", "auto")).lower()
            if d == "auto":
                cls._device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                cls._device = d

        if cls._svm is None:
            cls._svm = joblib.load(_MODEL_PATH)

        if cls._clip_model is None or cls._clip_preprocess is None:
            model_name = str(cls._pp["clip_model"])
            model, _, preprocess_val = open_clip.create_model_and_transforms(model_name)
            model.eval().to(cls._device)
            cls._clip_model = model
            cls._clip_preprocess = preprocess_val

    def analyze(
        self,
        img: Image.Image,
        score_only: bool = False,
        **_: Any,
    ) -> MethodResult:
        try:
            type(self)._lazy_load()
            pp = type(self)._pp
            svm = type(self)._svm
            clip_model = type(self)._clip_model
            clip_preprocess = type(self)._clip_preprocess
            device = type(self)._device

            assert pp is not None and svm is not None and clip_model is not None and clip_preprocess is not None and device is not None

            base = _prepare_rgb(img)
            x = clip_preprocess(base).unsqueeze(0).to(device)

            with torch.inference_mode():
                emb = clip_model.encode_image(x).float().cpu().numpy()  # [1, D]

            if bool(pp.get("normalize_embedding", True)):
                emb = _l2_normalize(emb).astype(np.float32, copy=False)
            else:
                emb = emb.astype(np.float32, copy=False)

            decision = float(svm.decision_function(emb)[0])
            pred = int(svm.predict(emb)[0])

            score = float(_sigmoid(decision))

            if score_only:
                return MethodResult(
                    name=self.name,
                    task="detection",
                    score=score,
                    metrics={},
                    visuals_b64={},
                )

            metrics: Dict[str, float] = {
                "decision": float(decision),
                "pred": float(pred),
                "embedding_dim": float(emb.shape[1]),
            }

            visuals: Dict[str, str] = {}
            try:
                overlay, hm = _svm_patch_heatmap_overlay(
                    img=img,
                    clip_model=clip_model,
                    clip_preprocess=clip_preprocess,
                    device=device,
                    svm=svm,
                    alpha=0.5,
                )
                visuals["heatmap"] = _img_to_b64_png(overlay)
                metrics.update({k: float(v) for k, v in hm.items()})
            except Exception as e_hm:
                metrics.update({"heat_error": 1.0, "heat_error_msg": str(e_hm)})

            return MethodResult(
                name=self.name,
                task="detection",
                score=score,
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
