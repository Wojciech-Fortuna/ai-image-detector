import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

import core.analyze as az


def MR(name, score, metrics=None, task="detection", visuals_b64=None):
    return SimpleNamespace(
        name=name,
        score=score,
        metrics=metrics or {},
        task=task,
        visuals_b64=visuals_b64 or {},
    )


class FakeMethod:
    def __init__(self, name, score, metrics=None, task="detection"):
        self.name = name
        self._score = score
        self._metrics = metrics or {}
        self._task = task

    def analyze(self, img, image_path=None):
        return MR(self.name, self._score, metrics=dict(self._metrics), task=self._task, visuals_b64={})


class FakeAttribMethod:
    def __init__(self, name, pred_label=None, confidence=None, metrics=None):
        self.name = name
        self._pred_label = pred_label
        self._confidence = confidence
        self._metrics = metrics or {}

    def analyze(self, img, image_path=None):
        return SimpleNamespace(
            name=self.name,
            pred_label=self._pred_label,
            confidence=self._confidence,
            metrics=dict(self._metrics),
            task="attribution",
            visuals_b64={},
        )


class DummyClf:
    def __init__(self, classes=(0, 1), proba=(0.2, 0.8)):
        self.classes_ = np.array(classes)
        self._proba = np.array([proba], dtype=float)

    def predict_proba(self, X):
        return self._proba


def test_analyze_image_default_all_methods_mean(monkeypatch, tiny_img):
    fake_registry = {
        "k1": FakeMethod("m1", 0.2),
        "k2": FakeMethod("m2", 0.6),
    }
    monkeypatch.setattr(az, "REGISTRY", fake_registry, raising=False)

    out = az.analyze_image(tiny_img, threshold=0.5, selected_methods=[], exif={}, image_path=None)

    assert out["result"]["score_ai"] == pytest.approx(0.4, rel=1e-6)
    assert out["result"]["label"] == "REAL"


def test_analyze_image_skips_errored_and_nan_in_mean(monkeypatch, tiny_img):
    fake_registry = {
        "k1": FakeMethod("m1", float("nan")),
        "k2": FakeMethod("m2", 0.8),
        "k3": FakeMethod("m3", 0.1, metrics={"error": 1.0}),
    }
    monkeypatch.setattr(az, "REGISTRY", fake_registry, raising=False)

    out = az.analyze_image(tiny_img, threshold=0.5, selected_methods=[], exif={}, image_path=None)
    assert out["result"]["score_ai"] == pytest.approx(0.8)
    assert out["result"]["label"] == "AI"


def test_analyze_image_c2pa_gate_fastpath_ai(monkeypatch, tiny_img):
    fake_registry = {
        "k_c2pa": FakeMethod("c2pa", 99.9),
        "k_other": FakeMethod("m1", 0.1),
    }
    monkeypatch.setattr(az, "REGISTRY", fake_registry, raising=False)

    out = az.analyze_image(tiny_img, selected_methods=[az.META_METHOD_NAME], exif={}, image_path=None)
    assert out["result"]["label"] == "AI"
    assert out["result"]["score_ai"] == pytest.approx(99.9)
    assert "attribution" in out


def test_analyze_image_prnu_gate_fastpath_real(monkeypatch, tiny_img):
    fake_registry = {
        "k_c2pa": FakeMethod("c2pa", float("nan")),
        "k_prnu": FakeMethod("prnu", 0.1),
        "k_other": FakeMethod("m1", 0.9),
    }
    monkeypatch.setattr(az, "REGISTRY", fake_registry, raising=False)

    out = az.analyze_image(tiny_img, selected_methods=[az.META_METHOD_NAME], exif={}, image_path=None)
    assert out["result"]["label"] == "REAL"
    assert out["result"]["score_ai"] == pytest.approx(0.1)


def test_analyze_image_meta_path_sets_meta_metrics_and_component(monkeypatch, tiny_img):
    fake_registry = {
        "k_c2pa": FakeMethod("c2pa", float("nan")),
        "k_a": FakeMethod("a", 0.2),
        "k_b": FakeMethod("b", 0.9),
    }
    monkeypatch.setattr(az, "REGISTRY", fake_registry, raising=False)

    monkeypatch.setattr(az, "_get_has_face", lambda img, image_path, meta_package=None: 0)

    meta_package = {
        "clf": DummyClf(classes=(0, 1), proba=(0.3, 0.7)),  # p(ai)=0.7
        "model_names": ["a", "b"],
        "use_logit": False,
        "feature_semantics": "p_ai",
        "label_map": {"real": 0, "ai": 1},
        "face_only_model_names": [],
        "neutral_p_ai_for_skipped_face_models": 0.5,
        "add_is_face_feature": False,
    }
    monkeypatch.setattr(az, "load_meta_package", lambda path=None: meta_package)

    out = az.analyze_image(tiny_img, selected_methods=[az.META_METHOD_FAST_NAME], exif={}, image_path=None)

    assert out["metrics"]["meta_variant"] == "fast"
    assert out["metrics"]["meta_has_face"] == 0
    assert "meta_request_id" in out["metrics"]
    assert out["result"]["label"] == "AI"
    assert out["result"]["components"][az.META_METHOD_NAME] == pytest.approx(0.7)


def test_analyze_image_attribution_runs_when_selected_and_detection_present(monkeypatch, tiny_img):
    fake_registry = {
        "k_det": FakeMethod("m1", 0.9),
        "k_attr": FakeAttribMethod("attrib_generator", pred_label="diffusion", confidence=0.8),
        "k_sdv": FakeAttribMethod("attrib_sd_variant", pred_label="sdxl", confidence=0.6),
    }
    monkeypatch.setattr(az, "REGISTRY", fake_registry, raising=False)

    out = az.analyze_image(
        tiny_img,
        selected_methods=[
            "m1",
            az.ATTRIB_GENERATOR_METHOD_NAME,
            az.ATTRIB_SD_VARIANT_METHOD_NAME,
        ],
        exif={},
        image_path=None,
        threshold=0.5,
    )

    assert out["result"]["label"] == "AI"
    assert "generator" in out["attribution"]
    assert out["attribution"]["generator"]["label"] == "diffusion"
    assert "sd_variant" in out["attribution"]


def test_run_batch_writes_reports_and_artifacts(monkeypatch, tmp_path):
    input_dir = tmp_path / "in"
    input_dir.mkdir()
    out_dir = tmp_path / "out"
    (input_dir / "a.png").write_bytes(b"fake")
    (input_dir / "b.jpg").write_bytes(b"fake")

    monkeypatch.setattr(az, "load_image", lambda fh: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(az, "extract_exif", lambda img: {"x": 1})

    def fake_analyze_image(img, threshold=0.5, selected_methods=None, exif=None, image_path=None, request_id=None):
        return {
            "result": {"label": "REAL", "score_ai": 0.1, "components": {"m": 0.1}},
            "metrics": {},
            "visuals": {"x": "y"},
            "exif": exif or {},
            "attribution": {},
        }

    monkeypatch.setattr(az, "analyze_image", fake_analyze_image)

    saved = []
    monkeypatch.setattr(az, "save_report_json", lambda report, path: saved.append(Path(path)))

    monkeypatch.setattr(az, "replace_non_finite", lambda x: x)
    monkeypatch.setattr(az, "flatten_report_to_row", lambda report, rel_id: {"path": rel_id, "result_score_ai": 0.1})

    monkeypatch.setattr(az, "build_summary_from_rows", lambda rows, **k: {"n_total": len(rows)})
    monkeypatch.setattr(az, "render_pie_png", lambda counts: b"PNG")

    def fake_write_batch_artifacts(*, out_dir, rows, threshold, summary, pie_png_bytes):
        p = Path(out_dir) / "results.csv"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("ok", encoding="utf-8")
        return {"results_csv": p}

    monkeypatch.setattr(az, "write_batch_artifacts", fake_write_batch_artifacts)

    res_csv = az.run_batch(input_dir, out_dir, threshold=0.5, selected_methods=[az.META_METHOD_NAME])

    assert res_csv.exists()
    assert len(saved) == 2
    assert all(p.parent.name == "reports" for p in saved)
