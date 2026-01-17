from __future__ import annotations

import base64
import io
import math
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Tuple, List, Optional

import numpy as np
from PIL import Image, ImageOps

import torch
import torch.nn as nn

import clip
from torchvision import models, transforms

from methods.base import BaseMethod, MethodResult, register


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CLIP_LABEL_MAP = {
    1: "sd21",
    2: "sdxl",
    3: "sd3",
    4: "dalle3",
    5: "midjourney",
}
CLIP_CLASSES: List[str] = list(CLIP_LABEL_MAP.values())
CLIP_NUM_CLASSES = len(CLIP_CLASSES)
CLIP_IDX2CLASS = {i: v for i, v in enumerate(CLIP_CLASSES)}

CLIP_WEIGHTS_PATH_DEFAULT = Path("models") / "clip_probe_finetuned.pt"
CLIP_BACKBONE_DEFAULT = "ViT-B/32"

DRAGON_NUM_CLASSES = 8
DRAGON_ID2LABEL_RAW = {
    0: "real",
    1: "sd1",
    2: "sd2",
    3: "sdxl",
    4: "sd3",
    5: "ssd",
    6: "pixart",
    7: "other_diffusion",
}
DRAGON_WEIGHTS_PATH_DEFAULT = Path("models") / "dragon_model_final.pt"

DRAGON_TEST_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

FUSED_CLASSES: List[str] = [
    "sd1",
    "sd2",
    "sdxl",
    "sd3",
    "dalle3",
    "midjourney",
    "ssd",
    "pixart",
    "other_diffusion",
]

CLIP_TO_FUSED: Dict[str, str] = {
    "sd21": "sd2",
    "sdxl": "sdxl",
    "sd3": "sd3",
    "dalle3": "dalle3",
    "midjourney": "midjourney",
}

FUSE_FLOOR_LAMBDA = 0.01


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


def _overlay_heatmap_redblue(base: Image.Image, heat_2d: np.ndarray, alpha: float = 0.45) -> Image.Image:
    base = base.convert("RGB")
    h = Image.fromarray((heat_2d * 255).astype(np.uint8), mode="L").resize(base.size, Image.BILINEAR)

    arr = (np.array(h, dtype=np.float32) / 255.0)
    heat = np.zeros((arr.shape[0], arr.shape[1], 3), dtype=np.uint8)
    heat[..., 0] = (arr * 255).astype(np.uint8)
    heat[..., 2] = ((1.0 - arr) * 80).astype(np.uint8)

    return Image.blend(base, Image.fromarray(heat, "RGB"), alpha=alpha)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    s = np.sum(e)
    return e / s if s > 0 else np.ones_like(e) / len(e)


def _fuse_product_of_experts(
    fused_classes: List[str],
    probs_clip: Dict[str, float],
    probs_dragon: Dict[str, float],
    alpha: float = 1.0,
    beta: float = 1.0,
    eps: float = 1e-9,
) -> Tuple[str, float, Dict[str, float]]:
    n = max(len(fused_classes), 1)
    floor = float(FUSE_FLOOR_LAMBDA) / float(n) if FUSE_FLOOR_LAMBDA > 0 else 0.0

    scores = np.zeros((n,), dtype=np.float64)

    for i, cls in enumerate(fused_classes):
        s = 0.0
        if cls in probs_clip:
            p = float(probs_clip[cls])
            p = max(p, floor, eps)
            s += alpha * math.log(p)
        if cls in probs_dragon:
            p = float(probs_dragon[cls])
            p = max(p, floor, eps)
            s += beta * math.log(p)
        scores[i] = s

    p = _softmax_np(scores)
    fused_probs = {cls: float(p[i]) for i, cls in enumerate(fused_classes)}
    pred_label = max(fused_probs, key=fused_probs.get)
    confidence = float(fused_probs[pred_label])
    return pred_label, confidence, fused_probs


class LinearProbe(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(input_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


_CLIP_CACHE: Dict[Tuple[str, str, str], Tuple[nn.Module, nn.Module, Any]] = {}


def _load_clip_and_probe(
    weights_path: str | Path,
    device: Optional[str] = None,
    clip_backbone: str = CLIP_BACKBONE_DEFAULT,
) -> Tuple[nn.Module, nn.Module, Any, str]:
    if device is None:
        device = DEVICE

    key = (str(weights_path), str(device), str(clip_backbone))
    if key in _CLIP_CACHE:
        clip_model, probe, preprocess = _CLIP_CACHE[key]
        return clip_model, probe, preprocess, device

    clip_model, preprocess = clip.load(clip_backbone, device=device)
    clip_model.eval()

    probe = LinearProbe(512, CLIP_NUM_CLASSES).to(device)
    probe.eval()

    ckpt = torch.load(str(weights_path), map_location=device)
    clip_model.load_state_dict(ckpt["clip"])
    probe.load_state_dict(ckpt["probe"])

    _CLIP_CACHE[key] = (clip_model, probe, preprocess)
    return clip_model, probe, preprocess, device


@contextmanager
def _force_mha_need_weights(clip_model: nn.Module):
    visual = getattr(clip_model, "visual", None)
    if visual is None or not hasattr(visual, "transformer") or not hasattr(visual.transformer, "resblocks"):
        yield
        return

    patched: List[Tuple[nn.Module, Any]] = []

    for rb in visual.transformer.resblocks:
        if not hasattr(rb, "attn"):
            continue
        attn = rb.attn
        if not isinstance(attn, nn.MultiheadAttention):
            continue

        orig_forward = attn.forward

        def wrapped_forward(*args, __orig=orig_forward, **kwargs):
            kwargs["need_weights"] = True
            if "average_attn_weights" in kwargs:
                kwargs["average_attn_weights"] = False
            try:
                return __orig(*args, **kwargs)
            except TypeError:
                kwargs.pop("average_attn_weights", None)
                return __orig(*args, **kwargs)

        attn.forward = wrapped_forward
        patched.append((attn, orig_forward))

    try:
        yield
    finally:
        for attn, orig in patched:
            attn.forward = orig


@torch.no_grad()
def _attention_rollout_map_clip_vit(
    clip_model: nn.Module,
    image_tensor: torch.Tensor,
) -> torch.Tensor:
    visual = clip_model.visual
    if not hasattr(visual, "transformer") or not hasattr(visual.transformer, "resblocks"):
        raise RuntimeError("CLIP visual backbone does not look like ViT (missing transformer.resblocks).")

    attn_mats: List[torch.Tensor] = []

    def _hook_attn(_mod, _inp, out):
        if isinstance(out, (tuple, list)) and len(out) >= 2 and out[1] is not None:
            attn_mats.append(out[1].detach())

    hooks = []
    for rb in visual.transformer.resblocks:
        if hasattr(rb, "attn") and isinstance(rb.attn, nn.MultiheadAttention):
            hooks.append(rb.attn.register_forward_hook(_hook_attn))

    with _force_mha_need_weights(clip_model):
        _ = clip_model.encode_image(image_tensor)

    for h in hooks:
        h.remove()

    if len(attn_mats) == 0:
        raise RuntimeError("Could not collect attention matrices (attn_weights).")

    mats: List[torch.Tensor] = []
    for A in attn_mats:
        if A.dim() == 4:
            A = A.mean(dim=1)
        elif A.dim() == 3:
            if A.shape[0] != image_tensor.shape[0] and A.shape[1] == image_tensor.shape[0]:
                A = A.permute(1, 0, 2).contiguous()
        else:
            continue
        mats.append(A)

    if len(mats) == 0:
        raise RuntimeError("Collected attentions have unsupported shape.")

    B = mats[0].shape[0]
    tokens = mats[0].shape[-1]
    device = mats[0].device

    eye = torch.eye(tokens, device=device).unsqueeze(0).expand(B, tokens, tokens)
    joint = eye.clone()

    for A in mats:
        if A.shape[-1] != tokens or A.shape[-2] != tokens:
            continue
        A = A + eye
        A = A / A.sum(dim=-1, keepdim=True).clamp(min=1e-6)
        joint = torch.bmm(A, joint)

    cls_to_patches = joint[:, 0, 1:]
    n_patches = cls_to_patches.shape[-1]
    side = int(math.sqrt(n_patches))
    if side * side != n_patches:
        raise RuntimeError(f"Non-square number of patches: {n_patches}")

    amap = cls_to_patches.reshape(B, side, side)
    amap = amap - amap.amin(dim=(1, 2), keepdim=True)
    amap = amap / amap.amax(dim=(1, 2), keepdim=True).clamp(min=1e-6)
    return amap[0]


def _clip_overlay(
    clip_model: nn.Module,
    preprocess,
    clip_device: str,
    img_pil: Image.Image,
) -> Tuple[Image.Image, Dict[str, float]]:
    base = _prepare_rgb(img_pil)
    x = preprocess(base).unsqueeze(0).to(clip_device)

    amap = _attention_rollout_map_clip_vit(clip_model, x)
    amap_np = amap.detach().float().cpu().numpy()

    overlay = _overlay_heatmap_redblue(base, amap_np, alpha=0.45)
    metrics = {
        "attn_map_min": float(amap_np.min()),
        "attn_map_max": float(amap_np.max()),
        "attn_map_mean": float(amap_np.mean()),
        "attn_map_std": float(amap_np.std()),
        "attn_map_side": float(amap_np.shape[0]),
    }
    return overlay, metrics


def _run_clip_probe(
    img: Image.Image,
    weights_path: str | Path,
    device: Optional[str],
    clip_backbone: str,
) -> Tuple[str, float, Dict[str, float], Optional[Image.Image], Dict[str, float]]:
    clip_model, probe, preprocess, clip_device = _load_clip_and_probe(
        weights_path=weights_path,
        device=device,
        clip_backbone=clip_backbone,
    )

    base = _prepare_rgb(img)
    x = preprocess(base).unsqueeze(0).to(clip_device)

    with torch.inference_mode():
        emb = clip_model.encode_image(x).float()
        logits = probe(emb)
        probs = _softmax_probs(logits)

    pred_id = int(probs.argmax(dim=1).item())
    confidence = float(probs.max(dim=1).values.item())
    pred_label = CLIP_IDX2CLASS.get(pred_id, str(pred_id))

    ent = _entropy(probs)
    eff_classes = float(math.exp(ent))
    prob_margin, logit_margin = _top2_stats(probs, logits)
    energy = _energy(logits, T=1.0)
    unk = _unknown_score(confidence, ent, CLIP_NUM_CLASSES)

    emb_norm = float(torch.linalg.norm(emb, dim=1).item())
    logits_l2 = float(torch.linalg.norm(logits, dim=1).item())
    probe_w_norm = float(torch.linalg.norm(probe.fc.weight).item())
    probe_b_norm = float(torch.linalg.norm(probe.fc.bias).item()) if probe.fc.bias is not None else 0.0

    p = probs[0].detach().cpu().numpy().tolist()
    metrics: Dict[str, float] = {
        "pred_id": float(pred_id),
        "confidence": float(confidence),
        "entropy": float(ent),
        "effective_classes": float(eff_classes),
        "top2_margin": float(prob_margin),
        "logit_margin": float(logit_margin),
        "energy": float(energy),
        "unknown_score": float(unk),
        "emb_norm_l2": float(emb_norm),
        "logits_norm_l2": float(logits_l2),
        "probe_weight_norm_l2": float(probe_w_norm),
        "probe_bias_norm_l2": float(probe_b_norm),
    }
    for i, cls_name in CLIP_IDX2CLASS.items():
        metrics[f"p_{cls_name}"] = float(p[i])

    overlay_img: Optional[Image.Image] = None
    try:
        overlay_img, viz_m = _clip_overlay(clip_model, preprocess, clip_device, img)
        metrics.update(viz_m)
    except Exception:
        metrics["viz_error"] = 1.0

    clip_probs_fused: Dict[str, float] = {}
    for cls_name in CLIP_CLASSES:
        raw_p = float(metrics.get(f"p_{cls_name}", 0.0))
        fused_name = CLIP_TO_FUSED.get(cls_name)
        if fused_name is not None:
            clip_probs_fused[fused_name] = clip_probs_fused.get(fused_name, 0.0) + raw_p

    return pred_label, confidence, metrics, overlay_img, clip_probs_fused


def build_dragon_model(num_classes: int = DRAGON_NUM_CLASSES) -> nn.Module:
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(in_features, num_classes),
    )
    return model


_DRAGON_CACHE: Dict[Tuple[str, str], nn.Module] = {}


def _load_dragon_model(weights_path: str | Path, device: Optional[str] = None) -> Tuple[nn.Module, str]:
    if device is None:
        device = DEVICE
    key = (str(weights_path), str(device))
    if key in _DRAGON_CACHE:
        return _DRAGON_CACHE[key], device

    model = build_dragon_model().to(device)
    try:
        state = torch.load(str(weights_path), map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(str(weights_path), map_location=device)

    model.load_state_dict(state)
    model.eval()

    _DRAGON_CACHE[key] = model
    return model, device


def _gradcam_overlay_dragon(
    model: nn.Module,
    device: str,
    img_pil: Image.Image,
    target_class: int,
) -> Tuple[Image.Image, Dict[str, float]]:
    base = _prepare_rgb(img_pil)
    x = DRAGON_TEST_TRANSFORM(base).unsqueeze(0).to(device)

    target_layer = model.features[-1]

    activations: List[torch.Tensor] = []
    gradients: List[torch.Tensor] = []

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
    if act.dim() != 4 or grad.dim() != 4:
        raise RuntimeError(f"GradCAM expects BCHW, got act={tuple(act.shape)}, grad={tuple(grad.shape)}")

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((weights * act).sum(dim=1, keepdim=True))

    cam_min = cam.min()
    cam_max = cam.max()
    cam_norm = (cam - cam_min) / (cam_max - cam_min + 1e-8)

    cam_np = cam_norm[0, 0].detach().cpu().numpy()
    cam_img = Image.fromarray((cam_np * 255).astype(np.uint8), mode="L").resize(base.size, Image.BILINEAR)

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


def _run_dragon(
    img: Image.Image,
    weights_path: str | Path,
    device: Optional[str],
) -> Tuple[str, float, Dict[str, float], Optional[Image.Image], Dict[str, float]]:
    model, dragon_device = _load_dragon_model(weights_path=weights_path, device=device)

    base = _prepare_rgb(img)
    x = DRAGON_TEST_TRANSFORM(base).unsqueeze(0).to(dragon_device)

    with torch.inference_mode():
        logits = model(x)
        probs = _softmax_probs(logits)

    pred_id_raw = int(probs.argmax(dim=1).item())
    confidence_raw = float(probs.max(dim=1).values.item())
    pred_label_raw = DRAGON_ID2LABEL_RAW.get(pred_id_raw, str(pred_id_raw))

    ent = _entropy(probs)
    eff_classes = float(math.exp(ent))
    prob_margin, logit_margin = _top2_stats(probs, logits)
    energy = _energy(logits, T=1.0)
    unk = _unknown_score(confidence_raw, ent, DRAGON_NUM_CLASSES)

    p = probs[0].detach().cpu().numpy().tolist()
    while len(p) < DRAGON_NUM_CLASSES:
        p.append(0.0)

    p_real, p_sd1, p_sd2, p_sdxl, p_sd3, p_ssd, p_pixart, p_other_raw = p[:8]

    p_other_diffusion = float(p_real + p_other_raw)

    merged: Dict[str, float] = {
        "sd1": float(p_sd1),
        "sd2": float(p_sd2),
        "sdxl": float(p_sdxl),
        "sd3": float(p_sd3),
        "ssd": float(p_ssd),
        "pixart": float(p_pixart),
        "other_diffusion": float(p_other_diffusion),
    }
    pred_label_merged = max(merged, key=merged.get)
    confidence_merged = float(merged[pred_label_merged])

    eps = 1e-12
    p_sd_family = float(p_sd1 + p_sd2 + p_sdxl + p_sd3)
    p_alt_diff = float(p_ssd + p_pixart)
    p_known_diffusion_total = float(1.0 - p_other_diffusion)
    sd_share_of_known = float(p_sd_family / max(p_known_diffusion_total, eps)) if p_known_diffusion_total > 0 else 0.0

    metrics: Dict[str, float] = {
        "pred_id_raw": float(pred_id_raw),
        "pred_label_raw_is_real": 1.0 if pred_label_raw == "real" else 0.0,
        "confidence_raw": float(confidence_raw),
        "confidence_merged": float(confidence_merged),

        "entropy": float(ent),
        "effective_classes": float(eff_classes),
        "top2_margin": float(prob_margin),
        "logit_margin": float(logit_margin),
        "energy": float(energy),
        "unknown_score": float(unk),

        "p_real": float(p_real),
        "p_sd1": float(p_sd1),
        "p_sd2": float(p_sd2),
        "p_sdxl": float(p_sdxl),
        "p_sd3": float(p_sd3),
        "p_ssd": float(p_ssd),
        "p_pixart": float(p_pixart),
        "p_other_raw": float(p_other_raw),

        "p_other_diffusion": float(p_other_diffusion),

        "p_known_diffusion_total": float(p_known_diffusion_total),
        "p_sd_family_total": float(p_sd_family),
        "p_alt_diffusion_total": float(p_alt_diff),
        "sd_family_share_of_known_diffusion": float(sd_share_of_known),
    }

    overlay_img: Optional[Image.Image] = None
    try:
        overlay_img, cam_m = _gradcam_overlay_dragon(model, dragon_device, img, target_class=pred_id_raw)
        metrics.update(cam_m)
    except Exception:
        metrics["cam_error"] = 1.0

    dragon_probs_fused = merged.copy()
    return pred_label_merged, confidence_merged, metrics, overlay_img, dragon_probs_fused


@register
class AttributionCombinedMethod(BaseMethod):
    name = "attrib_sd_variant"
    description = "Identifies the most likely diffusion model variant."

    how_title = "attrib_sd_variant (Diffusion model variant recognition)"
    how_text = (
        "This method identifies the most likely diffusion model variant that generated an image, "
        "such as SD1, SD2, SDXL, SD3, DALL·E 3, Midjourney, or other diffusion-based models. "
        "It combines the predictions of two independent methods: a CLIP-based image classifier "
        "and a ConvNeXt-based image classifier trained to recognize diffusion model variants."
    )

    def analyze(self, img: Image.Image, score_only: bool = False, **kwargs: Any) -> MethodResult:
        clip_weights_path = kwargs.get("clip_weights_path", CLIP_WEIGHTS_PATH_DEFAULT)
        clip_device = kwargs.get("clip_device", None)
        clip_backbone = kwargs.get("clip_backbone", CLIP_BACKBONE_DEFAULT)

        dragon_weights_path = kwargs.get("dragon_weights_path", DRAGON_WEIGHTS_PATH_DEFAULT)
        dragon_device = kwargs.get("dragon_device", None)

        fuse_alpha = float(kwargs.get("fuse_alpha", 1.0))
        fuse_beta = float(kwargs.get("fuse_beta", 1.0))

        try:
            clip_pred, clip_conf, clip_metrics, clip_overlay, clip_probs = _run_clip_probe(
                img=img,
                weights_path=clip_weights_path,
                device=clip_device,
                clip_backbone=clip_backbone,
            )

            dragon_pred, dragon_conf, dragon_metrics, dragon_overlay, dragon_probs = _run_dragon(
                img=img,
                weights_path=dragon_weights_path,
                device=dragon_device,
            )

            pred_label, confidence, fused_probs = _fuse_product_of_experts(
                fused_classes=FUSED_CLASSES,
                probs_clip=clip_probs,
                probs_dragon=dragon_probs,
                alpha=fuse_alpha,
                beta=fuse_beta,
            )

            p_clip_for_label = float(clip_probs.get(pred_label, 0.0))
            p_dragon_for_label = float(dragon_probs.get(pred_label, 0.0))

            if p_clip_for_label >= p_dragon_for_label:
                chosen_overlay = clip_overlay
                heatmap_type = "clip_attention"
            else:
                chosen_overlay = dragon_overlay
                heatmap_type = "gradcam"

            if score_only:
                return MethodResult(
                    name=self.name,
                    task="attribution",
                    pred_label=pred_label,
                    confidence=confidence,
                    metrics={},
                    visuals_b64={},
                )

            metrics: Dict[str, Any] = {
                "fuse_alpha": float(fuse_alpha),
                "fuse_beta": float(fuse_beta),
                "fuse_floor_lambda": float(FUSE_FLOOR_LAMBDA),
                "fuse_floor_value": float(FUSE_FLOOR_LAMBDA / max(len(FUSED_CLASSES), 1)),
                "fused_confidence": float(confidence),

                "heatmap_type": heatmap_type,

                "clip_pred_label": str(clip_pred),
                "clip_pred_confidence": float(clip_conf),
                "dragon_pred_label": str(dragon_pred),
                "dragon_pred_confidence_merged": float(dragon_conf),

                "viz_p_clip_for_fused_label": float(p_clip_for_label),
                "viz_p_dragon_for_fused_label": float(p_dragon_for_label),
            }

            for k, v in fused_probs.items():
                metrics[f"fused_p_{k}"] = float(v)

            for k, v in clip_metrics.items():
                metrics[f"clip_{k}"] = float(v)

            for k, v in dragon_metrics.items():
                metrics[f"dragon_{k}"] = float(v)

            visuals: Dict[str, str] = {}
            if chosen_overlay is not None:
                visuals["heatmap"] = _img_to_b64_png(chosen_overlay)

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
