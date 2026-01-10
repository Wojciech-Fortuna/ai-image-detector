import csv
import io
import json
import math
from pathlib import Path

import pytest

from utils import batch_pipeline as bp


def test_replace_non_finite_recurses_and_converts_tuple_to_list():
    x = {
        "a": 1.0,
        "b": float("nan"),
        "c": [float("inf"), 2.0, {"d": -float("inf")}],
        "t": (1.0, float("nan")),
        "s": "ok",
    }
    out = bp.replace_non_finite(x)
    assert out["a"] == 1.0
    assert out["b"] is None
    assert out["c"][0] is None
    assert out["c"][1] == 2.0
    assert out["c"][2]["d"] is None
    assert isinstance(out["t"], list)
    assert out["t"] == [1.0, None]
    assert out["s"] == "ok"


def test_flatten_report_to_row_basic_fields_and_components_metrics():
    report = {
        "result": {
            "label": "AI",
            "score_ai": 0.9,
            "components": {"m1": 0.1, "combined_methods": 0.9},
        },
        "metrics": {"k": 7, "flag": True},
        "attribution": {},
    }
    row = bp.flatten_report_to_row(report, "img1.png")
    assert row["path"] == "img1.png"
    assert row["result_label"] == "AI"
    assert row["result_score_ai"] == 0.9
    assert row["component__m1"] == 0.1
    assert row["component__combined_methods"] == 0.9
    assert row["metric__k"] == 7
    assert row["metric__flag"] is True


def test_flatten_report_to_row_attribution_and_stage_metrics_filtering():
    report = {
        "result": {"label": "AI", "score_ai": 0.8, "components": {}},
        "metrics": {},
        "attribution": {
            "generator": {
                "label": "diffusion",
                "confidence": 0.77,
                "metrics": {
                    "model": "sdxl",
                    "steps": 30,
                    "ok": True,
                    "p": 0.2,
                    "nested": {"no": "include"},
                    "list": [1, 2, 3],
                },
            },
            "sd_variant": "not_a_dict",
        },
    }
    row = bp.flatten_report_to_row(report, "x")
    assert row["attrib__generator__label"] == "diffusion"
    assert row["attrib__generator__confidence"] == 0.77
    assert row["attrib__generator__metric__model"] == "sdxl"
    assert row["attrib__generator__metric__steps"] == 30
    assert row["attrib__generator__metric__ok"] is True
    assert row["attrib__generator__metric__p"] == 0.2

    assert "attrib__generator__metric__nested" not in row
    assert "attrib__generator__metric__list" not in row
    assert "attrib__sd_variant__label" not in row


def test_flatten_report_to_row_propagates_error_fields():
    report = {"error": 1, "error_msg": "boom", "result": {}, "metrics": {}, "attribution": {}}
    row = bp.flatten_report_to_row(report, "img")
    assert row["error"] == 1
    assert row["error_msg"] == "boom"


def _parse_csv_bytes(b: bytes):
    text = b.decode("utf-8")
    reader = csv.DictReader(io.StringIO(text))
    rows = list(reader)
    return reader.fieldnames, rows


def test_rows_to_csv_bytes_has_core_fields_first_and_sorted_rest():
    rows = [
        {
            "path": "a",
            "result_label": "AI",
            "result_score_ai": 0.9,
            "z_extra": 1,
            "a_extra": 2,
            "error": "",
            "error_msg": "",
        },
        {
            "path": "b",
            "result_label": "REAL",
            "result_score_ai": 0.1,
            "a_extra": 3,
        },
    ]
    b = bp.rows_to_csv_bytes(rows)
    header, parsed = _parse_csv_bytes(b)

    assert header[:5] == ["path", "result_label", "result_score_ai", "error", "error_msg"]

    assert header[5:] == ["a_extra", "z_extra"]

    assert parsed[0]["path"] == "a"
    assert parsed[1]["path"] == "b"
    assert parsed[0]["a_extra"] == "2"
    assert parsed[0]["z_extra"] == "1"


def test_parse_finite_float():
    assert bp.parse_finite_float("0.5") == pytest.approx(0.5)
    assert bp.parse_finite_float(1) == pytest.approx(1.0)
    assert bp.parse_finite_float(float("nan")) is None
    assert bp.parse_finite_float(float("inf")) is None
    assert bp.parse_finite_float("nope") is None


def test_compute_counts_from_rows_threshold_and_errors():
    rows = [
        {"result_score_ai": 0.9, "error": ""},          # AI
        {"result_score_ai": "0.1", "error": "0"},       # REAL
        {"result_score_ai": "", "error": ""},           # unknown
        {"result_score_ai": None, "error": "1"},        # unknown + err
        {"result_score_ai": "nan", "error": ""},        # unknown
        {"result_score_ai": 0.5, "error": ""},          # AI if >= threshold
    ]
    counts = bp.compute_counts_from_rows(rows, threshold=0.5)
    assert counts["total"] == 6
    assert counts["ai"] == 2
    assert counts["real"] == 1
    assert counts["unknown"] == 3
    assert counts["error_count"] == 1
    assert counts["threshold"] == pytest.approx(0.5)


def test_write_batch_artifacts_writes_all_files_and_content(tmp_path: Path):
    out_dir = tmp_path / "out"
    rows = [
        {"path": "a", "result_label": "AI", "result_score_ai": 0.9},
        {"path": "b", "result_label": "REAL", "result_score_ai": 0.1, "error": 1, "error_msg": "x"},
    ]
    summary = {"n_total": 2, "foo": "bar"}
    pie = b"PNGDATA"

    paths = bp.write_batch_artifacts(
        out_dir=out_dir,
        rows=rows,
        threshold=0.5,
        summary=summary,
        pie_png_bytes=pie,
    )

    for k, p in paths.items():
        assert Path(p).exists(), f"{k} missing: {p}"

    csv_bytes = paths["results_csv"].read_bytes()
    header, parsed = _parse_csv_bytes(csv_bytes)
    assert len(parsed) == 2
    assert "path" in header and "result_score_ai" in header

    counts = json.loads(paths["counts_json"].read_text(encoding="utf-8"))
    assert counts["total"] == 2
    assert counts["ai"] == 1
    assert counts["real"] == 1
    assert counts["unknown"] == 0
    assert counts["error_count"] == 1

    summ2 = json.loads(paths["summary_json"].read_text(encoding="utf-8"))
    assert summ2 == summary

    assert paths["pie_png"].read_bytes() == pie

    readme = paths["readme"].read_text(encoding="utf-8")
    assert "Batch results folder" in readme
    assert "results.csv" in readme
    assert "counts.json" in readme
