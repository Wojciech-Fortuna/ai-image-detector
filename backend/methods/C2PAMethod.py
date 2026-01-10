from __future__ import annotations

import json
from typing import Dict, Any, Optional, Tuple

from PIL import Image
from c2pa import Reader

from methods.base import BaseMethod, MethodResult, register


def load_manifest(image_path: str) -> Tuple[Dict[str, Any], Optional[str], Optional[str]]:
    try:
        with Reader(image_path) as reader:
            try:
                validation_state = reader.get_validation_state()
            except Exception:
                validation_state = None

            raw_json = reader.json()
    except Exception as e:
        return {}, str(e), None

    if not raw_json:
        return {}, None, validation_state

    try:
        data = json.loads(raw_json)
    except json.JSONDecodeError as e:
        return {}, f"Invalid C2PA manifest JSON: {e}", validation_state

    return data, None, validation_state


def iter_assertions(manifest_dict: Dict[str, Any]):
    for _mid, m in manifest_dict.get("manifests", {}).items():
        for a in m.get("assertions", []):
            yield a


def has_digital_source(manifest_dict: Dict[str, Any], suffix: str) -> bool:
    target = suffix.lower()

    for assertion in iter_assertions(manifest_dict):
        data = assertion.get("data", {})

        dst = (data.get("digitalSourceType") or "").lower()
        if dst.endswith(target):
            return True

        if assertion.get("label", "").startswith("c2pa.actions"):
            for act in data.get("actions", []):
                dst2 = (act.get("digitalSourceType") or "").lower()
                if dst2.endswith(target):
                    return True

    return False


def classify_from_c2pa(manifest: Dict[str, Any], load_error: Optional[str]):
    if load_error is not None:
        return (
            "unreadable_manifest",
            f"Failed to read C2PA manifest: {load_error}",
        )

    if not manifest or "manifests" not in manifest:
        return (
            "unknown",
            "No C2PA manifest found or manifest contains no 'manifests' section.",
        )

    if has_digital_source(manifest, "trainedalgorithmicmedia"):
        return (
            "ai_origin",
            "digitalSourceType = trainedAlgorithmicMedia (created using Generative AI).",
        )

    if has_digital_source(manifest, "compositewithtrainedalgorithmicmedia"):
        return (
            "ai_edited",
            "digitalSourceType = compositeWithTrainedAlgorithmicMedia (edited using Generative AI).",
        )

    for _mid, m in manifest.get("manifests", {}).items():
        if m.get("description") == "AI Generated Image":
            sw = m.get("softwareAgent")
            if sw in {"Azure OpenAI DALL-E", "Azure OpenAI ImageGen"}:
                return (
                    "ai_origin",
                    (
                        "Azure OpenAI C2PA pattern detected: "
                        f"description='AI Generated Image', softwareAgent='{sw}'."
                    ),
                )

    for assertion in iter_assertions(manifest):
        if assertion.get("label", "").startswith("c2pa.actions"):
            for act in assertion.get("data", {}).get("actions", []):
                if act.get("action") == "c2pa.created":
                    dst = (act.get("digitalSourceType") or "").lower()
                    if "trainedalgorithmicmedia" in dst:
                        return (
                            "ai_origin",
                            "c2pa.created action indicates trainedAlgorithmicMedia.",
                        )
                    if "compositewithtrainedalgorithmicmedia" in dst:
                        return (
                            "ai_edited",
                            "c2pa.created action indicates compositeWithTrainedAlgorithmicMedia.",
                        )

    return (
        "unknown",
        "The C2PA manifest contains no known indicators of Generative AI usage.",
    )


@register
class C2PAMethod(BaseMethod):
    name = "c2pa"
    description = "Checks Content Credentials (C2PA) metadata for declared AI generation or AI edits."

    how_title = "C2PA (Content Credentials)"
    how_text = (
        "Some tools and platforms embed a Content Credentials (C2PA) manifest into the file, "
        "which can explicitly declare that an image was AI-generated or AI-edited. "
        "If such a verified declaration is present, it is a strong signal. "
        "If the manifest is missing or contains no AI indicators, it does NOT prove the image is real — "
        "it only means C2PA provides no conclusion."
    )

    def analyze(self, img: Image.Image, **kwargs) -> MethodResult:
        image_path: Optional[str] = kwargs.get("image_path")
        validation_state: Optional[str] = None

        if not image_path:
            manifest, load_error = {}, "Missing original file path (image_path)."
            label, explanation = classify_from_c2pa(manifest, load_error)
        else:
            manifest, load_error, validation_state = load_manifest(image_path)
            label, explanation = classify_from_c2pa(manifest, load_error)

        if validation_state is None:
            validation_ok = None
        else:
            validation_ok = str(validation_state).lower() in {"valid", "ok"}

        if label in ("ai_origin", "ai_edited") and validation_ok:
            score = 99.9
        else:
            score = float("nan")

        metrics: Dict[str, Any] = {
            "label": label,
            "explanation": explanation,
            "validation_state": validation_state or "unknown",
            "validation_ok": validation_ok,
        }

        return MethodResult(
            name=self.name,
            task="detection",
            score=score,
            metrics=metrics,
            visuals_b64={},
        )
