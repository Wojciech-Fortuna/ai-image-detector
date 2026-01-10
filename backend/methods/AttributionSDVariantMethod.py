from __future__ import annotations

import base64
import io
import math
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image, ImageOps
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

from methods.base import BaseMethod, MethodResult, register


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_CLASSES = 8

ID2LABEL_RAW = {
    0: "real",
    1: "sd1",
    2: "sd2",
    3: "sdxl",
    4: "sd3",
    5: "ssd",
    6: "pixart",
    7: "other_diffusion",
}

WEIGHTS_PATH_DEFAULT = Path("models") / "dragon_model_final.pt"

TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

_MODEL = None
_MODEL_DEVICE = None


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


def build_model(num_classes: int = NUM_CLASSES) -> nn.Module:
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


def _load_model(weights_path: str, device: str | None = None) -> nn.Module:
    global _MODEL, _MODEL_DEVICE
    if _MODEL is not None:
        return _MODEL

    if device is None:
        device = DEVICE

    model = build_model().to(device)
    state = torch.load(weights_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    _MODEL = model
    _MODEL_DEVICE = device
    return model


def _softmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.softmax(logits, dim=1)


def _entropy(probs: torch.Tensor, eps: float = 1e-12) -> float:
    p = probs.clamp(min=eps)
    return float(-(p * p.log()).sum(dim=1).item())


def _top2_stats(probs: torch.Tensor, logits: torch.Tensor) -> Tuple[float, float]:
    p_sorted, _ = probs.sort(dim=1, descending=True)
    z_sorted, _ = logits.sort(dim=1, descending=True)
    prob_margin = float((p_sorted[:, 0] - p_sorted[:, 1]).item()) if probs.shape[1] >= 2 else 0.0
    logit_margin = float((z_sorted[:, 0] - z_sorted[:, 1]).item()) if logits.shape[1] >= 2 else 0.0
    return prob_margin, logit_margin


def _energy(logits: torch.Tensor, T: float = 1.0) -> float:
    return float((-T * torch.logsumexp(logits / T, dim=1)).item())


def _unknown_score(confidence: float, entropy_val: float, num_classes: int) -> float:
    max_ent = math.log(max(num_classes, 2))
    ent_norm = float(entropy_val / max_ent) if max_ent > 0 else 0.0
    return float(0.5 * (1.0 - confidence) + 0.5 * ent_norm)


def _gradcam_overlay(
    model: nn.Module,
    img_pil: Image.Image,
    target_class: int,
) -> Tuple[Image.Image, Dict[str, float]]:
    base = _prepare_rgb(img_pil)
    x = TEST_TRANSFORM(base).unsqueeze(0).to(_MODEL_DEVICE)

    target_layer = model.features[-1]

    activations = []
    gradients = []

    def fwd_hook(_, __, output):
        activations.append(output)

    def bwd_hook(_, grad_input, grad_output):
        gradients.append(grad_output[0])

    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)

    model.zero_grad(set_to_none=True)
    logits = model(x)
    score = logits[:, target_class].sum()
    score.backward()

    h1.remove()
    h2.remove()

    act = activations[0]
    grad = gradients[0]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))

    cam_min = cam.min()
    cam_max = cam.max()
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    cam_np = cam_norm[0, 0].detach().cpu().numpy()
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L")
    cam_img = cam_img.resize(base.size, Image.BILINEAR)

    arr = (np.array(cam_img, dtype=np.float32) / 255.0)
    heat = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (arr * 255).astype(np.uint8)
    heat[..., 2] = ((1.0 - arr) * 80).astype(np.uint8)

    overlay = Image.blend(base.convert("RGB"), Image.fromarray(heat, "RGB"), alpha=0.45)

    metrics = {
        "cam_min": float(cam_min.item()),
        "cam_max": float(cam_max.item()),
        "cam_mean": float(cam_norm.mean().item()),
        "cam_std": float(cam_norm.std().item()),
    }
    return overlay, metrics


@register
class AttributionSDVariantMethod(BaseMethod):
    name = "attrib_sd_variant"
    description = "Identifies the most likely diffusion variant."

    how_title = "attrib_sd_variant (Diffusion model variant recognition)"
    how_text = (
        "This method looks for visual signatures characteristic of different diffusion model families "
        "and variants (e.g., SD1/SD2/SDXL/SD3, etc.). It estimates which variant best matches the image, "
        "or falls back to “other diffusion” when no specific known variant fits well."
    )

    def analyze(self, img: Image.Image, score_only: bool = False, **kwargs: Any) -> MethodResult:
        weights_path: str = kwargs.get("weights_path", WEIGHTS_PATH_DEFAULT)
        device: str | None = kwargs.get("device", None)

        try:
            model = _load_model(weights_path=weights_path, device=device)

            base = _prepare_rgb(img)
            x = TEST_TRANSFORM(base).unsqueeze(0).to(_MODEL_DEVICE)

            with torch.no_grad():
                logits = model(x)
                probs = _softmax_probs(logits)

            pred_id = int(probs.argmax(dim=1).item())
            confidence = float(probs.max(dim=1).values.item())

            pred_label = ID2LABEL_RAW.get(pred_id, str(pred_id))

            if score_only:
                return MethodResult(
                    name=self.name,
                    task="attribution",
                    pred_label=pred_label,
                    confidence=confidence,
                    metrics={},
                    visuals_b64={},
                )

            ent = _entropy(probs)
            eff_classes = float(math.exp(ent))
            prob_margin, logit_margin = _top2_stats(probs, logits)
            energy = _energy(logits, T=1.0)
            unk = _unknown_score(confidence, ent, NUM_CLASSES)

            p = probs[0].detach().cpu().numpy().tolist()
            p_real, p_sd1, p_sd2, p_sdxl, p_sd3, p_ssd, p_pixart, p_other_raw = p

            p_other_diffusion = float(p_real + p_other_raw)

            eps = 1e-12
            p_sd_family = float(p_sd1 + p_sd2 + p_sdxl + p_sd3)
            p_alt_diff = float(p_ssd + p_pixart)
            p_known_diffusion_total = float(1.0 - p_other_diffusion)  # known variants mass
            sd_share_of_known = float(p_sd_family / max(p_known_diffusion_total, eps)) if p_known_diffusion_total > 0 else 0.0

            metrics: Dict[str, float] = {
                "entropy": float(ent),
                "effective_classes": float(eff_classes),
                "top2_margin": float(prob_margin),
                "logit_margin": float(logit_margin),
                "energy": float(energy),
                "unknown_score": float(unk),
                "p_sd1": float(p_sd1),
                "p_sd2": float(p_sd2),
                "p_sdxl": float(p_sdxl),
                "p_sd3": float(p_sd3),
                "p_ssd": float(p_ssd),
                "p_pixart": float(p_pixart),
                "p_other_diffusion": float(p_other_diffusion),
                "p_known_diffusion_total": float(p_known_diffusion_total),
                "p_sd_family_total": float(p_sd_family),
                "p_alt_diffusion_total": float(p_alt_diff),
                "sd_family_share_of_known_diffusion": float(sd_share_of_known),
            }

            visuals: Dict[str, str] = {}
            try:
                overlay, cam_m = _gradcam_overlay(model, img, target_class=pred_id)
                visuals["gradcam"] = _img_to_b64_png(overlay)
                metrics.update(cam_m)
            except Exception:
                metrics["cam_error"] = 1.0

            return MethodResult(
                name=self.name,
                task="attribution",
                pred_label=pred_label,
                confidence=confidence,
                metrics=metrics,
                visuals_b64=visuals,
            )

        except Exception as e:
            return MethodResult(
                name=self.name,
                task="attribution",
                pred_label="error",
                confidence=float("nan"),
                metrics={"error": 1.0, "error_msg": str(e)},
                visuals_b64={},
            )
