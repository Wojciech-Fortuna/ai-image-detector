from __future__ import annotations

import importlib
import pkgutil
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from joblib import load
from PIL import Image

from methods.base import REGISTRY, MethodResult
from utils.io import extract_exif, load_image, save_report_json

from utils.batch_pipeline import replace_non_finite, flatten_report_to_row, write_batch_artifacts, compute_counts_from_rows
from utils.batch_summary import build_summary_from_rows
from utils.charts import render_pie_png


META_METHOD_NAME = "combined_methods"
META_METHOD_FAST_NAME = "combined_methods_fast"
META_MODEL_ONLY_NAME = "meta_model_only"

DEFAULT_META_MODEL_PATH = Path("models") / "meta_model_v2.joblib"

_META_MODEL_PATH: str = DEFAULT_META_MODEL_PATH
_META_PACKAGE_CACHE: Optional[Dict[str, Any]] = None

ATTRIB_GENERATOR_METHOD_NAME = "attrib_generator"
ATTRIB_SD_VARIANT_METHOD_NAME = "attrib_sd_variant"


def set_meta_model_path(path: str) -> None:
    global _META_MODEL_PATH, _META_PACKAGE_CACHE
    _META_MODEL_PATH = path
    _META_PACKAGE_CACHE = None


def auto_import_methods():
    import methods as methods_pkg

    package_path = methods_pkg.__path__
    for _, module_name, is_pkg in pkgutil.iter_modules(package_path):
        if is_pkg:
            continue
        if module_name in {"base", "__init__"}:
            continue
        importlib.import_module(f"methods.{module_name}")


def logit(p: float, eps: float = 1e-6) -> float:
    p = min(max(p, eps), 1.0 - eps)
    return float(np.log(p / (1.0 - p)))


def load_meta_package(path: Optional[str] = None) -> Dict[str, Any]:
    global _META_MODEL_PATH, _META_PACKAGE_CACHE
    if path is None:
        path = _META_MODEL_PATH

    if path != _META_MODEL_PATH:
        _META_MODEL_PATH = path
        _META_PACKAGE_CACHE = None

    if _META_PACKAGE_CACHE is None:
        if not Path(path).is_file():
            raise FileNotFoundError(f"Meta-model file not found: {path}")
        _META_PACKAGE_CACHE = load(path)

    return _META_PACKAGE_CACHE


def _result_task(r: MethodResult) -> str:
    return str(getattr(r, "task", "detection") or "detection")


def _is_detection(r: MethodResult) -> bool:
    return _result_task(r) == "detection"


def _find_method_by_name(method_name: str) -> Optional[Any]:
    for key, method in REGISTRY.items():
        name = getattr(method, "name", key)
        if name == method_name:
            return method
    return None


def _normalize_label(s: str) -> str:
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum() or ch in {"_", "-"})


def _extract_attrib_prediction(r: MethodResult) -> Tuple[Optional[str], Optional[float]]:
    pred_label = getattr(r, "pred_label", None)
    confidence = getattr(r, "confidence", None)

    if pred_label is None:
        m = getattr(r, "metrics", {}) or {}
        pred_label = (
            m.get("pred_label")
            or m.get("label")
            or m.get("generator")
            or m.get("model_name")
            or m.get("predicted")
        )

    if confidence is None:
        m = getattr(r, "metrics", {}) or {}
        confidence = m.get("confidence") or m.get("conf") or m.get("prob") or m.get("p")

    try:
        conf_f = float(confidence) if confidence is not None else None
    except Exception:
        conf_f = None

    return (str(pred_label) if pred_label is not None else None, conf_f)


def _safe_float_score(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def _score_to_component_value(score: Any) -> Optional[float]:
    s = _safe_float_score(score)
    return float(s) if np.isfinite(s) else None


def _mode_from_selected(selected_methods: List[str]) -> str:
    s = set(selected_methods or [])
    if META_MODEL_ONLY_NAME in s:
        return "meta_only"
    if META_METHOD_FAST_NAME in s:
        return "combined_fast"
    if META_METHOD_NAME in s:
        return "combined"
    return "none"


def _detect_face_runtime(image_path: Optional[str], img: Image.Image) -> int:
    try:
        import face_recognition
    except Exception:
        return 0

    try:
        if image_path and Path(image_path).is_file():
            arr = face_recognition.load_image_file(image_path)
        else:
            arr = np.array(img)
        locs = face_recognition.face_locations(arr, model="hog")
        return 1 if len(locs) > 0 else 0
    except Exception:
        return 0


def _get_has_face(
    img: Image.Image,
    image_path: Optional[str],
    meta_package: Optional[Dict[str, Any]] = None,
) -> int:
    return _detect_face_runtime(image_path, img)


def compute_meta_features_from_results(
    results: List[MethodResult],
    meta_package: Dict[str, Any],
    has_face: int = 0,
) -> Dict[str, Any]:
    clf = meta_package["clf"]
    model_names: List[str] = meta_package["model_names"]
    use_logit: bool = meta_package.get("use_logit", True)

    label_map = meta_package.get("label_map", {"real": 0, "ai": 1})
    feature_semantics = meta_package.get(
        "feature_semantics", "logit_p_ai" if use_logit else "p_ai"
    )

    face_only_names: Set[str] = set(meta_package.get("face_only_model_names", []) or [])
    neutral_p: float = float(meta_package.get("neutral_p_ai_for_skipped_face_models", 0.5))
    add_is_face_feature: bool = bool(meta_package.get("add_is_face_feature", False))

    det_results = [r for r in results if _is_detection(r)]
    by_name: Dict[str, MethodResult] = {r.name: r for r in det_results}

    m = len(model_names)
    X = np.zeros((1, m), dtype=np.float32)
    P_ai = np.zeros((1, m), dtype=np.float32)

    for j, name in enumerate(model_names):
        if name not in by_name:
            if (int(has_face) == 0) and (name in face_only_names):
                p_ai = neutral_p
            else:
                raise RuntimeError(
                    f"Base method '{name}' required by meta-model "
                    f"not found among executed detection methods: {list(by_name.keys())}"
                )
        else:
            r = by_name[name]
            p_ai = _safe_float_score(r.score)

        if not np.isfinite(p_ai):
            p_ai = neutral_p

        if p_ai < 0.0 or p_ai > 1.0:
            p_ai = min(max(p_ai, 0.0), 1.0)

        P_ai[0, j] = p_ai

        if feature_semantics in ("logit_p_ai", "p_ai"):
            X[0, j] = logit(p_ai) if (use_logit and feature_semantics == "logit_p_ai") else p_ai
        elif feature_semantics in ("logit_p_real", "p_real"):
            p_real = 1.0 - p_ai
            X[0, j] = logit(p_real) if (use_logit and feature_semantics == "logit_p_real") else p_real
        else:
            raise RuntimeError(f"Unknown meta feature_semantics='{feature_semantics}'")

    if add_is_face_feature:
        X = np.concatenate([X, np.array([[float(int(has_face))]], dtype=np.float32)], axis=1)

    if not np.isfinite(X).all():
        X[~np.isfinite(X)] = 0.0

    proba = clf.predict_proba(X)[0]
    classes = list(getattr(clf, "classes_", []))

    ai_label = int(label_map.get("ai", 1))
    real_label = int(label_map.get("real", 0))

    if ai_label not in classes or real_label not in classes:
        raise RuntimeError(f"Meta-model classes_ mismatch: classes_={classes}, label_map={label_map}")

    p_ai_meta = float(proba[classes.index(ai_label)])
    p_real_meta = float(proba[classes.index(real_label)])

    return {
        "score_ai": p_ai_meta,
        "score_real": p_real_meta,
        "label": "AI" if p_ai_meta >= 0.5 else "REAL",
        "P_ai": P_ai,
        "has_face": int(has_face),
    }


def _run_attribution_method(
    method_name: str,
    img: Image.Image,
    image_path: Optional[str],
) -> Optional[Dict[str, Any]]:
    method = _find_method_by_name(method_name)
    if method is None:
        return None

    try:
        r: MethodResult = method.analyze(img, image_path=image_path)
    except Exception as e:
        return {
            "name": method_name,
            "label": None,
            "confidence": None,
            "metrics": {"error": 1.0, "error_msg": str(e)},
            "visuals": {},
        }

    pred_label, conf = _extract_attrib_prediction(r)
    return {
        "name": getattr(r, "name", method_name),
        "label": pred_label,
        "confidence": conf,
        "metrics": getattr(r, "metrics", {}) or {},
        "visuals": getattr(r, "visuals_b64", {}) or {},
    }


def _build_methods_to_run_for_meta(
    *,
    meta_package: Dict[str, Any],
    mode: str,
    has_face: int,
    c2pa_registry_key: Optional[str],
    prnu_registry_key: Optional[str],
) -> Dict[str, Any]:
    meta_model_names: List[str] = meta_package["model_names"]
    face_only_names: Set[str] = set(meta_package.get("face_only_model_names", []) or [])

    name_to_key: Dict[str, str] = {}
    for key, method in REGISTRY.items():
        name_to_key[getattr(method, "name", key)] = key

    methods_to_run: Dict[str, Any] = {}

    for base_name in meta_model_names:
        if mode == "combined_fast" and base_name == "prnu":
            continue
        if mode == "meta_only" and base_name in {"prnu", "c2pa"}:
            continue

        if base_name not in name_to_key:
            raise RuntimeError(
                f"Base method '{base_name}' required by meta-model not found in REGISTRY. "
                f"Available: {list(name_to_key.keys())}"
            )
        reg_key = name_to_key[base_name]
        methods_to_run[reg_key] = REGISTRY[reg_key]

    if (has_face == 0) and face_only_names:
        drop_keys: List[str] = []
        for key, method in methods_to_run.items():
            mname = getattr(method, "name", key)
            if mname in face_only_names:
                drop_keys.append(key)
        for k in drop_keys:
            methods_to_run.pop(k, None)

    if c2pa_registry_key is not None:
        methods_to_run.pop(c2pa_registry_key, None)
    if prnu_registry_key is not None:
        methods_to_run.pop(prnu_registry_key, None)

    if mode in {"combined_fast", "meta_only"}:
        drop_prnu = [k for k, m in methods_to_run.items() if getattr(m, "name", k) == "prnu"]
        for k in drop_prnu:
            methods_to_run.pop(k, None)

    if mode == "meta_only":
        drop_c2pa = [k for k, m in methods_to_run.items() if getattr(m, "name", k) == "c2pa"]
        for k in drop_c2pa:
            methods_to_run.pop(k, None)

    return methods_to_run


def analyze_image(
    img: Image.Image,
    threshold: float = 0.5,
    selected_methods: Optional[List[str]] = None,
    exif: Optional[Dict[str, Any]] = None,
    image_path: Optional[str] = None,
    request_id: Optional[str] = None,
) -> Dict[str, Any]:
    rid = request_id or str(uuid.uuid4())
    selected_methods = selected_methods or []

    mode = _mode_from_selected(selected_methods)

    if not REGISTRY:
        raise RuntimeError("No registered methods. Make sure methods/*.py are importable.")

    prnu_result: Optional[MethodResult] = None
    prnu_registry_key: Optional[str] = None

    c2pa_result: Optional[MethodResult] = None
    c2pa_registry_key: Optional[str] = None

    ATTRIB_METHOD_NAMES = {ATTRIB_GENERATOR_METHOD_NAME, ATTRIB_SD_VARIANT_METHOD_NAME}
    explicit_attrib_selected: Set[str] = {
        m for m in (selected_methods or []) if m in ATTRIB_METHOD_NAMES
    }

    use_gates = mode in {"combined", "combined_fast"}

    if use_gates:
        c2pa_method = None
        for key, method in REGISTRY.items():
            method_name = getattr(method, "name", key)
            if method_name == "c2pa":
                c2pa_registry_key = key
                c2pa_method = method
                break

        if c2pa_method is not None:
            try:
                c2pa_result = c2pa_method.analyze(img, image_path=image_path)
            except Exception as e:
                c2pa_result = MethodResult(
                    name="c2pa",
                    score=float("nan"),
                    metrics={"error": 1.0, "error_msg": str(e)},
                    visuals_b64={},
                )

            c2pa_score = _safe_float_score(c2pa_result.score)

            if np.isfinite(c2pa_score) and abs(c2pa_score - 99.9) < 1e-6:
                det_metrics: Dict[str, Any] = {
                    f"{c2pa_result.name}_{k}": v for k, v in (c2pa_result.metrics or {}).items()
                }
                det_components = {c2pa_result.name: _score_to_component_value(c2pa_score)}

                detection_report = {
                    "result": {
                        "label": "AI",
                        "score_ai": c2pa_score,
                        "components": det_components,
                    },
                    "metrics": det_metrics,
                    "visuals": {},
                    "exif": exif or {},
                }

                attribution: Dict[str, Any] = {}
                gen = _run_attribution_method(ATTRIB_GENERATOR_METHOD_NAME, img, image_path)
                if gen is not None:
                    attribution["generator"] = gen
                    if _normalize_label(str(gen.get("label") or "")) == "diffusion":
                        sdv = _run_attribution_method(ATTRIB_SD_VARIANT_METHOD_NAME, img, image_path)
                        if sdv is not None:
                            attribution["sd_variant"] = sdv

                detection_report["attribution"] = attribution
                return detection_report

    if mode == "combined":
        prnu_method = None
        for key, method in REGISTRY.items():
            method_name = getattr(method, "name", key)
            if method_name == "prnu":
                prnu_registry_key = key
                prnu_method = method
                break

        if prnu_method is not None:
            try:
                prnu_result = prnu_method.analyze(img, image_path=image_path)
            except Exception as e:
                prnu_result = MethodResult(
                    name="prnu",
                    score=float("nan"),
                    metrics={"error": 1.0, "error_msg": str(e)},
                    visuals_b64={},
                )

            prnu_score = _safe_float_score(prnu_result.score)

            if np.isfinite(prnu_score) and abs(prnu_score - 0.1) < 1e-6:
                label = "AI" if prnu_score >= threshold else "REAL"

                det_metrics: Dict[str, Any] = {
                    f"{prnu_result.name}_{k}": v for k, v in (prnu_result.metrics or {}).items()
                }
                det_visuals: Dict[str, str] = {
                    f"{prnu_result.name}_{k}": v for k, v in (prnu_result.visuals_b64 or {}).items()
                }
                det_components: Dict[str, Any] = {prnu_result.name: _score_to_component_value(prnu_score)}

                if c2pa_result is not None:
                    c2pa_s = _safe_float_score(c2pa_result.score)
                    det_components[c2pa_result.name] = _score_to_component_value(c2pa_s)
                    det_metrics.update(
                        {f"{c2pa_result.name}_{k}": v for k, v in (c2pa_result.metrics or {}).items()}
                    )
                    det_visuals.update(
                        {f"{c2pa_result.name}_{k}": v for k, v in (c2pa_result.visuals_b64 or {}).items()}
                    )

                report = {
                    "result": {
                        "label": label,
                        "score_ai": prnu_score,
                        "components": det_components,
                    },
                    "metrics": det_metrics,
                    "visuals": det_visuals,
                    "exif": exif or {},
                }

                if label == "AI":
                    attribution: Dict[str, Any] = {}
                    gen = _run_attribution_method(ATTRIB_GENERATOR_METHOD_NAME, img, image_path)
                    if gen is not None:
                        attribution["generator"] = gen
                        if _normalize_label(str(gen.get("label") or "")) == "diffusion":
                            sdv = _run_attribution_method(ATTRIB_SD_VARIANT_METHOD_NAME, img, image_path)
                            if sdv is not None:
                                attribution["sd_variant"] = sdv
                    report["attribution"] = attribution
                else:
                    report["attribution"] = {}

                return report

    meta_package: Optional[Dict[str, Any]] = None
    has_face: int = 0
    methods_to_run: Dict[str, Any] = {}

    if mode in {"combined", "combined_fast", "meta_only"}:
        meta_package = load_meta_package()

        try:
            has_face = _get_has_face(img, image_path, meta_package=meta_package)
        except Exception:
            has_face = 0

        methods_to_run = _build_methods_to_run_for_meta(
            meta_package=meta_package,
            mode=mode,
            has_face=has_face,
            c2pa_registry_key=c2pa_registry_key,
            prnu_registry_key=prnu_registry_key,
        )

    else:
        selected_without_meta = [
            m
            for m in selected_methods
            if m not in {META_METHOD_NAME, META_METHOD_FAST_NAME, META_MODEL_ONLY_NAME}
        ]
        if selected_without_meta:
            methods_to_run = {
                k: v
                for k, v in REGISTRY.items()
                if getattr(v, "name", k) in selected_without_meta or k in selected_without_meta
            }
        else:
            methods_to_run = dict(REGISTRY)

    if not methods_to_run and c2pa_result is None and prnu_result is None:
        raise RuntimeError("No method selected to run.")

    results: List[MethodResult] = []
    if c2pa_result is not None:
        results.append(c2pa_result)
    if prnu_result is not None:
        results.append(prnu_result)

    for key in sorted(methods_to_run.keys()):
        method = methods_to_run[key]
        mname = getattr(method, "name", key)

        try:
            res = method.analyze(img, image_path=image_path)
            results.append(res)
        except Exception:
            results.append(
                MethodResult(
                    name=mname,
                    score=float("nan"),
                    metrics={"error": 1.0, "error_msg": "exception"},
                    visuals_b64={},
                )
            )

    attrib_block: Dict[str, Any] = {}
    if mode not in {"combined", "combined_fast", "meta_only"} and explicit_attrib_selected:
        KEY_MAP = {
            ATTRIB_GENERATOR_METHOD_NAME: "generator",
            ATTRIB_SD_VARIANT_METHOD_NAME: "sd_variant",
        }

        attrib_results = [r for r in results if not _is_detection(r)]
        for r in attrib_results:
            pred_label, conf = _extract_attrib_prediction(r)
            key = KEY_MAP.get(r.name)
            if not key:
                continue

            attrib_block[key] = {
                "name": getattr(r, "name", None) or "",
                "label": pred_label,
                "confidence": conf,
                "metrics": getattr(r, "metrics", {}) or {},
                "visuals": getattr(r, "visuals_b64", {}) or {},
            }

    det_results = [r for r in results if _is_detection(r)]

    finite_scores: List[float] = []
    for r in det_results:
        if "error" in (r.metrics or {}):
            continue
        s = _safe_float_score(r.score)
        if np.isfinite(s):
            finite_scores.append(float(s))

    if not finite_scores:
        score_ai = float("nan")
        label = "UNKNOWN"
    else:
        if mode in {"combined", "combined_fast", "meta_only"} and meta_package is not None:
            meta_out = compute_meta_features_from_results(det_results, meta_package, has_face=has_face)
            score_ai = float(meta_out["score_ai"])
            label = "AI" if score_ai >= threshold else "REAL"
        else:
            score_ai = float(np.mean(finite_scores))
            label = "AI" if score_ai >= threshold else "REAL"

    metrics: Dict[str, Any] = {}
    visuals: Dict[str, str] = {}
    for r in det_results:
        metrics.update({f"{r.name}_{k}": v for k, v in (r.metrics or {}).items()})
        visuals.update({f"{r.name}_{k}": v for k, v in (r.visuals_b64 or {}).items()})

    if mode in {"combined", "combined_fast", "meta_only"}:
        metrics["meta_has_face"] = int(has_face)
        metrics["meta_request_id"] = rid
        metrics["meta_variant"] = (
            "meta_only" if mode == "meta_only" else ("fast" if mode == "combined_fast" else "full")
        )
    else:
        metrics["request_id"] = rid

    components: Dict[str, Any] = {}
    for r in det_results:
        components[r.name] = _score_to_component_value(r.score)

    if mode in {"combined", "combined_fast", "meta_only"}:
        if np.isfinite(_safe_float_score(score_ai)):
            components[META_METHOD_NAME] = float(score_ai)
        else:
            components[META_METHOD_NAME] = None

    report: Dict[str, Any] = {
        "result": {
            "label": label,
            "score_ai": round(score_ai, 4) if np.isfinite(_safe_float_score(score_ai)) else score_ai,
            "components": components,
        },
        "metrics": metrics,
        "visuals": visuals,
        "exif": exif or {},
    }

    if mode in {"combined", "combined_fast", "meta_only"}:
        attribution: Dict[str, Any] = {}
        if label == "AI":
            gen = _run_attribution_method(ATTRIB_GENERATOR_METHOD_NAME, img, image_path)
            if gen is not None:
                attribution["generator"] = gen
                if _normalize_label(str(gen.get("label") or "")) == "diffusion":
                    sdv = _run_attribution_method(ATTRIB_SD_VARIANT_METHOD_NAME, img, image_path)
                    if sdv is not None:
                        attribution["sd_variant"] = sdv
        report["attribution"] = attribution
    else:
        report["attribution"] = attrib_block if explicit_attrib_selected else {}

    return report


def run_batch(
    input_dir: Path,
    out_dir: Path,
    threshold: float = 0.5,
    selected_methods: Optional[List[str]] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reports_dir = out_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    img_exts = {".jpg", ".jpeg", ".png", ".bmp"}

    for p in sorted(input_dir.rglob("*")):
        if not p.is_file():
            continue
        if p.suffix.lower() not in img_exts:
            continue

        rel_id = str(p.relative_to(input_dir))
        try:
            with p.open("rb") as fh:
                img = load_image(fh)
            exif = extract_exif(img)

            report = analyze_image(
                img,
                threshold=threshold,
                selected_methods=selected_methods,
                exif=exif,
                image_path=str(p),
            )

            report = dict(report)
            report["visuals"] = {}

            attr = report.get("attribution", {})
            if isinstance(attr, dict):
                attr2: Dict[str, Any] = {}
                for stage_name, stage in attr.items():
                    if isinstance(stage, dict):
                        st = dict(stage)
                        st["visuals"] = {}
                        attr2[stage_name] = st
                    else:
                        attr2[stage_name] = stage
                report["attribution"] = attr2

            report = replace_non_finite(report)

            rep_path = reports_dir / f"{p.stem}_report.json"
            save_report_json(report, rep_path)

            rows.append(flatten_report_to_row(report, rel_id))

        except Exception as e:
            rows.append(
                {
                    "path": rel_id,
                    "result_label": "",
                    "result_score_ai": "",
                    "error": 1,
                    "error_msg": f"{type(e).__name__}: {e}",
                }
            )

    summary = build_summary_from_rows(
        rows,
        error_column="error",
        score_column="result_score_ai",
        id_column="path",
    )

    counts = compute_counts_from_rows(rows, float(threshold))
    pie_png = render_pie_png(counts)

    artifacts = write_batch_artifacts(
        out_dir=out_dir,
        rows=rows,
        threshold=float(threshold),
        summary=summary,
        pie_png_bytes=pie_png,
    )

    return artifacts["results_csv"]
