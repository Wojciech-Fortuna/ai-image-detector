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

import timm

from .base import BaseMethod, MethodResult, register


_WEIGHTS_PATH = Path("models") / "convnext_stylegan.pth"

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


def _entropy_bernoulli(p: float, eps: float = 1e-12) -> float:
    p = min(max(float(p), eps), 1.0 - eps)
    return float(-(p * math.log(p) + (1.0 - p) * math.log(1.0 - p)))


def _energy_binary_from_logits(z1: float, z0: float = 0.0) -> float:
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


class ClassificationHead(nn.Module):
    def __init__(self, in_features: int):
        super().__init__()
        self.norm = nn.LayerNorm(in_features)
        self.dropout1 = nn.Dropout(0.4)
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.ReLU(inplace=True)
        self.dropout2 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.norm(x)
        x = self.dropout1(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        return x


class ConvNeXtBinaryClassifier(nn.Module):

    def __init__(self, backbone: nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = ClassificationHead(in_features=backbone.num_features)

    def forward(self, x: torch.Tensor, return_logits: bool = False):
        feats = self.backbone(x)
        logits = self.head(feats)
        probs = torch.sigmoid(logits)
        if return_logits:
            return probs, logits
        return probs


@register
class ConvNeXtStyleGANMethod(BaseMethod):
    name = "convnext_stylegan1"
    description = "Face-focused AI vs real classifier (ConvNeXt). Best for portrait photos."
    
    how_title = "ConvNeXt (Portrait-focused)"
    how_text = (
        "A ConvNeXt-based classifier trained to distinguish AI-generated vs real images in portrait photos. "
        "Use it mainly when a clear face is visible; results on non-face images can be unreliable."
    )

    AI_CLASS_INDEX: int = 0  # AI = "fake"

    _model: Optional[ConvNeXtBinaryClassifier] = None
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

            backbone = timm.create_model(
                "convnext_tiny",
                pretrained=False,
                num_classes=0,
                global_pool="avg",
            )
            model = ConvNeXtBinaryClassifier(backbone=backbone).to(cls._device)

            state_dict = torch.load(str(_WEIGHTS_PATH), map_location=cls._device, weights_only=True)
            model.load_state_dict(state_dict, strict=True)

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
    def _p_ai_from_prob_class1(cls, p1: float, logit1: float) -> Tuple[float, float, float, float]:
        p1 = float(np.clip(p1, 0.0, 1.0))
        z1 = float(logit1)

        if cls.AI_CLASS_INDEX == 1:
            p_ai = p1
            p_real = 1.0 - p1
            logit_ai = z1
            logit_real = -z1
        else:
            p_ai = 1.0 - p1
            p_real = p1
            logit_ai = -z1
            logit_real = z1

        return (
            float(np.clip(p_ai, 0.0, 1.0)),
            float(np.clip(p_real, 0.0, 1.0)),
            float(logit_ai),
            float(logit_real),
        )

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

        feats: Dict[str, torch.Tensor] = {}
        grads: Dict[str, torch.Tensor] = {}

        if not hasattr(model.backbone, "stages"):
            raise RuntimeError("Backbone does not have .stages; cannot attach Grad-CAM hook.")

        stage = model.backbone.stages[3]
        target_layer = stage[-1] if hasattr(stage, "__getitem__") else stage

        def fwd_hook(_m, _inp, out):
            feats["value"] = out

        def bwd_hook(_m, _gin, gout):
            g = gout[0] if isinstance(gout, (tuple, list)) and len(gout) else None
            grads["value"] = g

        h1 = target_layer.register_forward_hook(fwd_hook)
        h2 = target_layer.register_full_backward_hook(bwd_hook)

        try:
            x = x.to(cls._device)
            x.requires_grad_(True)

            prob_t, logits_t = model(x, return_logits=True)
            z1 = logits_t[0, 0]

            if target_is_ai:
                target_logit = -z1 if cls.AI_CLASS_INDEX == 0 else z1
            else:
                target_logit = z1 if cls.AI_CLASS_INDEX == 0 else -z1

            model.zero_grad(set_to_none=True)
            target_logit.backward(retain_graph=False)

            A = feats.get("value")
            dA = grads.get("value")
            if A is None or dA is None:
                raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

            if isinstance(A, torch.Tensor) and isinstance(dA, torch.Tensor):
                if A.ndim == 4 and dA.ndim == 4:
                    if A.shape[1] != dA.shape[1] and A.shape[-1] == dA.shape[-1]:
                        A = A.permute(0, 3, 1, 2).contiguous()
                        dA = dA.permute(0, 3, 1, 2).contiguous()

            if not (isinstance(A, torch.Tensor) and isinstance(dA, torch.Tensor)):
                raise RuntimeError("Grad-CAM expected tensor activations and gradients.")

            if A.ndim != 4 or dA.ndim != 4:
                raise RuntimeError(f"Grad-CAM expected 4D tensors, got A.ndim={A.ndim}, dA.ndim={dA.ndim}")

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
                    p1 = float(prob_t[0, 0].item())  # P(real)
                    z1 = float(logits_t[0, 0].item())  # logit(real)

                p_ai, p_real, logit_ai, logit_real = self._p_ai_from_prob_class1(p1, z1)

                return MethodResult(
                    name=self.name,
                    task="detection",
                    score=float(p_ai),
                    metrics={},
                    visuals_b64={},
                )

            x = x.to(self._device)

            with torch.no_grad():
                prob_t, logits_t = self._model(x, return_logits=True)

            p1 = float(prob_t[0, 0].detach().cpu().item())
            z1 = float(logits_t[0, 0].detach().cpu().item())

            p_ai, p_real, logit_ai, logit_real = self._p_ai_from_prob_class1(p1, z1)

            overlay = self._gradcam(x, base_rgb, target_is_ai=True)

            metrics = {
                "p_ai": p_ai,
                "p_real": p_real,
                "logit_ai": logit_ai,
                "logit_real": logit_real,
                "entropy": _entropy_bernoulli(p_ai),
                "prob_margin": float(abs(p_ai - 0.5) * 2.0),
                "logit_margin": float(abs(logit_ai)),
                "energy": _energy_binary_from_logits(logit_ai, 0.0),
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
