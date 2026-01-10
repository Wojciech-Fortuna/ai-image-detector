from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any
from PIL import Image

REGISTRY: Dict[str, "BaseMethod"] = {}

@dataclass
class MethodResult:
    name: str
    score: float
    metrics: Dict[str, float]
    visuals_b64: Dict[str, str]


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
