from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Any, Literal, Optional
from PIL import Image

REGISTRY: Dict[str, "BaseMethod"] = {}


@dataclass
class MethodResult:
    name: str
    task: Literal["detection", "attribution"] = "detection"
    score: float = 0.0  # p(AI)
    pred_label: Optional[str] = None
    confidence: Optional[float] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    visuals_b64: Dict[str, str] = field(default_factory=dict)


class BaseMethod:
    name: str = "base"
    description: str = "Base interface"

    def analyze(self, img: Image.Image, **kwargs) -> MethodResult:
        raise NotImplementedError


def register(cls):
    if not hasattr(cls, "name"):
        raise ValueError("The method must define a 'name' attribute.")

    name = getattr(cls, "name")

    if name in REGISTRY:
        raise ValueError(f"A method named '{name}' is already registered.")

    REGISTRY[name] = cls()
    return cls
