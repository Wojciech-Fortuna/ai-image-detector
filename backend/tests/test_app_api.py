import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import app as ap


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(ap, "REGISTRY", {"dummy": object()}, raising=False)

    monkeypatch.setattr(ap, "load_image", lambda fh: Image.new("RGB", (1, 1)))
    monkeypatch.setattr(ap, "extract_exif", lambda img: {"ok": 1})

    calls = {"selected": None}

    def fake_analyze_image(img, threshold=0.5, selected_methods=None, exif=None, image_path=None, request_id=None):
        calls["selected"] = list(selected_methods or [])
        return {
            "result": {"label": "REAL", "score_ai": 0.1, "components": {"x": 0.1}},
            "metrics": {"m": 1},
            "visuals": {},
            "exif": exif or {},
            "attribution": {},
        }

    monkeypatch.setattr(ap, "analyze_image", fake_analyze_image)

    monkeypatch.setattr(ap, "replace_non_finite", lambda x: x)
    monkeypatch.setattr(ap, "flatten_report_to_row", lambda report, path_in_zip: {"path": path_in_zip, "result_score_ai": 0.1})
    monkeypatch.setattr(ap, "rows_to_csv_bytes", lambda rows: b"CSV")
    monkeypatch.setattr(ap, "compute_counts_from_rows", lambda rows, thr: {"total": len(rows), "real": len(rows), "ai": 0, "unknown": 0, "error_count": 0, "threshold": float(thr)})
    monkeypatch.setattr(ap, "build_summary_from_rows", lambda rows, **k: {"n_total": len(rows)})
    monkeypatch.setattr(ap, "render_pie_png", lambda counts: b"PNG")

    test_app = ap.create_app()
    test_app.state._calls = calls
    return TestClient(test_app)


def _make_png_bytes():
    img = Image.new("RGB", (1, 1))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def test_analyze_defaults_to_combined_methods_when_methods_empty(client):
    png = _make_png_bytes()
    resp = client.post(
        "/analyze",
        files={"file": ("x.png", png, "image/png")},
        data={"threshold": "0.5", "methods_json": "[]"},
    )
    assert resp.status_code == 200
    calls = client.app.state._calls
    assert calls["selected"] == [ap.META_METHOD_NAME]
    body = resp.json()
    assert body["result"]["label"] == "REAL"


def test_analyze_rejects_bad_threshold(client):
    png = _make_png_bytes()
    resp = client.post(
        "/analyze",
        files={"file": ("x.png", png, "image/png")},
        data={"threshold": "1.5", "methods_json": "[]"},
    )
    assert resp.status_code == 400
    assert "threshold must be in [0,1]" in resp.text


def test_analyze_rejects_too_large_file_413(client, monkeypatch):
    monkeypatch.setattr(ap, "MAX_SINGLE_IMAGE_BYTES", 5)

    big = b"x" * 6
    resp = client.post(
        "/analyze",
        files={"file": ("big.png", big, "image/png")},
        data={"threshold": "0.5", "methods_json": "[]"},
    )

    assert resp.status_code == 413
    assert "too large" in resp.text.lower()


def _zip_with_images():
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("a.png", _make_png_bytes())
        z.writestr("b.jpg", b"fakejpg")
        z.writestr("note.txt", b"ignore")
    return buf.getvalue()


def test_analyze_zip_returns_zip_and_defaults_to_combined_methods(client, tmp_path):
    zbytes = _zip_with_images()
    resp = client.post(
        "/analyze_zip",
        files={"file": ("imgs.zip", zbytes, "application/zip")},
        data={"threshold": "0.5", "methods_json": "[]"},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/zip")

    calls = client.app.state._calls
    assert calls["selected"] == [ap.META_METHOD_NAME]

    z = zipfile.ZipFile(io.BytesIO(resp.content))
    names = set(z.namelist())
    assert "results.csv" in names
    assert "counts.json" in names
    assert "summary.json" in names
    assert "pie.png" in names
    assert "README.txt" in names

    assert z.read("results.csv") == b"CSV"
    counts = json.loads(z.read("counts.json").decode("utf-8"))
    assert counts["total"] == 2
