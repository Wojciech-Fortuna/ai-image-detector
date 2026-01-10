import math
from types import SimpleNamespace

import numpy as np
import pytest

import core.analyze as az


class DummyClf:
    def __init__(self, classes=(0, 1), proba=(0.2, 0.8)):
        self.classes_ = np.array(classes)
        self._proba = np.array([proba], dtype=float)

    def predict_proba(self, X):
        assert len(X.shape) == 2 and X.shape[0] == 1
        return self._proba


def mr(name, score, metrics=None, task="detection", visuals_b64=None):
    return SimpleNamespace(
        name=name,
        score=score,
        metrics=metrics or {},
        task=task,
        visuals_b64=visuals_b64 or {},
    )


def test_mode_from_selected_priorities():
    assert az._mode_from_selected([az.META_METHOD_NAME]) == "combined"
    assert az._mode_from_selected([az.META_METHOD_FAST_NAME]) == "combined_fast"
    assert az._mode_from_selected([az.META_MODEL_ONLY_NAME]) == "meta_only"

    assert az._mode_from_selected([az.META_METHOD_NAME, az.META_METHOD_FAST_NAME]) == "combined_fast"
    assert az._mode_from_selected([az.META_METHOD_NAME, az.META_MODEL_ONLY_NAME]) == "meta_only"


def test_score_to_component_value_handles_non_finite():
    assert az._score_to_component_value(0.123) == pytest.approx(0.123)
    assert az._score_to_component_value("0.5") == pytest.approx(0.5)
    assert az._score_to_component_value("nope") is None
    assert az._score_to_component_value(float("nan")) is None
    assert az._score_to_component_value(float("inf")) is None
    assert az._score_to_component_value(-float("inf")) is None


def test_normalize_label():
    assert az._normalize_label(" Diffusion! ") == "diffusion"
    assert az._normalize_label("SD-XL_1.0") == "sd-xl_10"


def test_extract_attrib_prediction_prefers_attrs_then_metrics():
    r1 = SimpleNamespace(pred_label="diffusion", confidence=0.9, metrics={})
    assert az._extract_attrib_prediction(r1) == ("diffusion", 0.9)

    r2 = SimpleNamespace(pred_label=None, confidence=None, metrics={"generator": "diffusion", "conf": "0.7"})
    assert az._extract_attrib_prediction(r2) == ("diffusion", 0.7)

    r3 = SimpleNamespace(pred_label=None, confidence=None, metrics={"label": 123, "confidence": "bad"})
    assert az._extract_attrib_prediction(r3) == ("123", None)


def test_compute_meta_features_logit_p_ai_and_face_only_neutral():
    meta_package = {
        "clf": DummyClf(classes=(0, 1), proba=(0.7, 0.3)),
        "model_names": ["m1", "m2_face"],
        "use_logit": True,
        "feature_semantics": "logit_p_ai",
        "label_map": {"real": 0, "ai": 1},
        "face_only_model_names": ["m2_face"],
        "neutral_p_ai_for_skipped_face_models": 0.5,
        "add_is_face_feature": False,
    }

    results = [
        mr("m1", 0.2),
    ]

    out = az.compute_meta_features_from_results(results, meta_package, has_face=0)
    assert out["label"] in {"AI", "REAL"}
    assert out["score_ai"] == pytest.approx(0.3)
    assert out["has_face"] == 0
    assert out["P_ai"].shape == (1, 2)
    assert out["P_ai"][0, 0] == pytest.approx(0.2)
    assert out["P_ai"][0, 1] == pytest.approx(0.5)


def test_compute_meta_features_raises_when_required_missing_non_face_only():
    meta_package = {
        "clf": DummyClf(classes=(0, 1), proba=(0.5, 0.5)),
        "model_names": ["need_me"],
        "use_logit": False,
        "feature_semantics": "p_ai",
        "label_map": {"real": 0, "ai": 1},
        "face_only_model_names": [],
        "neutral_p_ai_for_skipped_face_models": 0.5,
    }
    with pytest.raises(RuntimeError):
        az.compute_meta_features_from_results([], meta_package, has_face=0)


def test_build_methods_to_run_for_meta_enforces_rules(monkeypatch):
    class M:
        def __init__(self, name):
            self.name = name
        def analyze(self, *a, **k):
            raise AssertionError("should not be called")

    fake_registry = {
        "k_c2pa": M("c2pa"),
        "k_prnu": M("prnu"),
        "k_a": M("a"),
        "k_face": M("face_model"),
    }
    monkeypatch.setattr(az, "REGISTRY", fake_registry, raising=False)

    meta_package = {
        "model_names": ["c2pa", "prnu", "a", "face_model"],
        "face_only_model_names": ["face_model"],
    }

    m = az._build_methods_to_run_for_meta(
        meta_package=meta_package,
        mode="combined_fast",
        has_face=1,
        c2pa_registry_key=None,
        prnu_registry_key=None,
    )
    names = {v.name for v in m.values()}
    assert "prnu" not in names
    assert "c2pa" in names and "a" in names and "face_model" in names

    m2 = az._build_methods_to_run_for_meta(
        meta_package=meta_package,
        mode="meta_only",
        has_face=1,
        c2pa_registry_key=None,
        prnu_registry_key=None,
    )
    names2 = {v.name for v in m2.values()}
    assert "prnu" not in names2
    assert "c2pa" not in names2
    assert "a" in names2

    m3 = az._build_methods_to_run_for_meta(
        meta_package=meta_package,
        mode="combined",
        has_face=0,
        c2pa_registry_key=None,
        prnu_registry_key=None,
    )
    names3 = {v.name for v in m3.values()}
    assert "face_model" not in names3
