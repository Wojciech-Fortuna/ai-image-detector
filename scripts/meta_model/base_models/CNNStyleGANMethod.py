from __future__ import annotations

import base64
import io
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from .base import BaseMethod, MethodResult, register


_WEIGHTS_PATH = Path("models") / "model_cnn_stylegan.pt"

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


def _safe_logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return float(math.log(p / (1.0 - p)))


def _entropy_bernoulli(p: float, eps: float = 1e-12) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def _energy_binary_from_logit(z1: float, z0: float = 0.0) -> float:
    m = max(z0, z1)
    return float(-(m + math.log(math.exp(z0 - m) + math.exp(z1 - m))))


def _to_grayscale_heatmap(cam01: np.ndarray) -> Image.Image:
    cam01 = np.clip(cam01, 0.0, 1.0)
    arr = (cam01 * 255.0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def _overlay_red(base_rgb: Image.Image, heat_l: Image.Image, alpha: float = 0.55) -> Image.Image:
    base = base_rgb.convert("RGB")
    heat = heat_l.resize(base.size, Image.BILINEAR)

    red = Image.new("RGB", base.size, (255, 0, 0))

    mask = np.asarray(heat).astype(np.float32) / 255.0
    mask = np.clip(mask * float(alpha), 0.0, 1.0)
    mask_u8 = (mask * 255.0).astype(np.uint8)
    mask_img = Image.fromarray(mask_u8, mode="L")

    return Image.composite(red, base, mask_img)


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)

        self.fc1 = nn.Linear(256 * 14 * 14, 256)
        self.fc2 = nn.Linear(256, 1)

    def forward(self, x, return_logits: bool = False):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))

        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        logits = self.fc2(x)
        prob = torch.sigmoid(logits)

        if return_logits:
            return prob, logits
        return prob


@register
class CNNStyleGANMethod(BaseMethod):
    name = "cnn_stylegan"
    description = "Face-focused AI vs real classifier (CNN). Best for portrait photos."
    
    how_title = "CNN (Portrait-focused)"
    how_text = (
        "A CNN classifier trained to distinguish AI-generated vs real images in portrait photos. "
        "Use it mainly when a clear face is visible; results on non-face images can be unreliable."
    )

    OUTPUT_IS_P_REAL = True

    _model: Optional[CNN] = None
    _device: Optional[str] = None
    _transform = None

    @classmethod
    def _lazy_load(cls) -> None:
        if cls._device is None:
            cls._device = "cuda" if torch.cuda.is_available() else "cpu"

        if cls._transform is None:
            cls._transform = transforms.Compose(
                [
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ]
            )

        if cls._model is None:
            if not _WEIGHTS_PATH.exists():
                raise FileNotFoundError(f"Missing weights: {_WEIGHTS_PATH}")

            model = CNN().to(cls._device)
            state = torch.load(str(_WEIGHTS_PATH), map_location=cls._device, weights_only=True)
            model.load_state_dict(state)
            model.eval()
            cls._model = model

    @classmethod
    def _preprocess(cls, img: Image.Image) -> Tuple[torch.Tensor, Image.Image]:
        cls._lazy_load()
        assert cls._transform is not None

        base_rgb = _prepare_rgb(img)
        x = cls._transform(base_rgb).unsqueeze(0)
        return x, base_rgb

    @classmethod
    def _p_ai_from_model_outputs(cls, prob: float, logit: float) -> Tuple[float, float, float, float]:
        p = float(prob)
        z = float(logit)

        if cls.OUTPUT_IS_P_REAL:
            p_real = p
            p_ai = 1.0 - p_real
            logit_real = z
            logit_ai = -z
        else:
            p_ai = p
            p_real = 1.0 - p_ai
            logit_ai = z
            logit_real = -z

        p_ai = float(np.clip(p_ai, 0.0, 1.0))
        p_real = float(np.clip(p_real, 0.0, 1.0))
        return p_ai, p_real, logit_ai, logit_real

    @classmethod
    def _gradcam(
        cls,
        x: torch.Tensor,
        base_rgb: Image.Image,
        target_is_ai: bool = True,
    ) -> Image.Image:
        cls._lazy_load()
        assert cls._model is not None
        assert cls._device is not None

        model = cls._model

        feats = {}
        grads = {}

        def fwd_hook(_m, _inp, out):
            feats["value"] = out

        def bwd_hook(_m, _gin, gout):
            grads["value"] = gout[0]

        h1 = model.bn4.register_forward_hook(fwd_hook)
        h2 = model.bn4.register_full_backward_hook(bwd_hook)

        try:
            x = x.to(cls._device)
            x.requires_grad_(True)

            prob_t, logits_t = model(x, return_logits=True)
            z = logits_t[0, 0]

            if cls.OUTPUT_IS_P_REAL:
                target_logit = -z if target_is_ai else z
            else:
                target_logit = z if target_is_ai else -z

            model.zero_grad(set_to_none=True)
            target_logit.backward(retain_graph=False)

            A = feats.get("value")
            dA = grads.get("value")
            if A is None or dA is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            weights = torch.mean(dA, dim=(2, 3), keepdim=True)
            cam = torch.sum(weights * A, dim=1, keepdim=False)
            cam = F.relu(cam)

            cam_np = cam.detach().cpu().numpy()[0]
            cam_np = cam_np - cam_np.min()
            denom = cam_np.max() if cam_np.max() > 1e-12 else 1.0
            cam01 = cam_np / denom

            heat_l = _to_grayscale_heatmap(cam01).resize(base_rgb.size, Image.BILINEAR)
            overlay = _overlay_red(base_rgb, heat_l, alpha=0.55)
            return overlay

        finally:
            h1.remove()
            h2.remove()

    def analyze(
        self,
        img: Image.Image,
        score_only: bool = False,
        **_: Any,
    ) -> MethodResult:
        try:
            x, base_rgb = self._preprocess(img)
            assert self._model is not None
            assert self._device is not None

            if score_only:
                with torch.no_grad():
                    prob_t, logits_t = self._model(x.to(self._device), return_logits=True)
                    prob = float(prob_t[0, 0].item())
                    z = float(logits_t[0, 0].item())

                p_ai, p_real, logit_ai, logit_real = self._p_ai_from_model_outputs(prob, z)

                return MethodResult(
                    name=self.name,
                    task="detection",
                    score=float(p_ai),
                    metrics={},
                    visuals_b64={},
                )

            x = x.to(self._device)

            prob_t, logits_t = self._model(x, return_logits=True)
            prob = float(prob_t[0, 0].detach().cpu().item())
            z = float(logits_t[0, 0].detach().cpu().item())

            p_ai, p_real, logit_ai, logit_real = self._p_ai_from_model_outputs(prob, z)

            overlay = self._gradcam(x, base_rgb, target_is_ai=True)

            metrics = {
                "p_ai": float(p_ai),
                "p_real": float(p_real),
                "logit_ai": float(logit_ai),
                "logit_real": float(logit_real),
                "entropy": _entropy_bernoulli(p_ai),
                "prob_margin": float(abs(p_ai - 0.5) * 2.0),
                "logit_margin": float(abs(logit_ai)),
                "energy": _energy_binary_from_logit(logit_ai, 0.0),
            }

            return MethodResult(
                name=self.name,
                task="detection",
                score=float(p_ai),
                metrics=metrics,
                visuals_b64={
                    "gradcam": _img_to_b64_png(overlay),
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
