import io
import json
import zipfile
from pathlib import Path

import pytest
from fastapi import HTTPException

import app as ap


def test_parse_methods_json_ok():
    assert ap._parse_methods_json('["a","b"]') == ["a", "b"]
    assert ap._parse_methods_json("[]") == []


def test_parse_methods_json_bad():
    with pytest.raises(HTTPException):
        ap._parse_methods_json('{"a":1}')
    with pytest.raises(HTTPException):
        ap._parse_methods_json('["a", 1]')


def test_validate_threshold():
    ap._validate_threshold(0.0)
    ap._validate_threshold(1.0)
    with pytest.raises(HTTPException):
        ap._validate_threshold(-0.1)
    with pytest.raises(HTTPException):
        ap._validate_threshold(1.1)


def _make_zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_safe_extract_zip_to_tempdir_extracts_images_and_ignores_others(tmp_path):
    zbytes = _make_zip_bytes(
        {
            "a.jpg": b"fake",
            "b.png": b"fake",
            "c.txt": b"nope",
            "sub/inner.jpeg": b"fake",
        }
    )
    zpath = tmp_path / "in.zip"
    zpath.write_bytes(zbytes)

    outdir = tmp_path / "out"
    outdir.mkdir()

    extracted = ap._safe_extract_zip_to_tempdir(str(zpath), str(outdir))
    names_in_zip = [x[0] for x in extracted]
    assert "a.jpg" in names_in_zip
    assert "b.png" in names_in_zip
    assert "sub/inner.jpeg" in names_in_zip
    for _, extracted_path in extracted:
        assert Path(extracted_path).exists()


def test_safe_extract_zip_to_tempdir_blocks_traversal(tmp_path):
    zbytes = _make_zip_bytes(
        {
            "../evil.jpg": b"x",
            "ok.png": b"x",
        }
    )
    zpath = tmp_path / "in.zip"
    zpath.write_bytes(zbytes)
    outdir = tmp_path / "out"
    outdir.mkdir()

    extracted = ap._safe_extract_zip_to_tempdir(str(zpath), str(outdir))
    names_in_zip = [x[0] for x in extracted]
    assert "ok.png" in names_in_zip
    assert "../evil.jpg" not in names_in_zip


def test_safe_extract_zip_to_tempdir_raises_if_no_images(tmp_path):
    zbytes = _make_zip_bytes({"a.txt": b"x"})
    zpath = tmp_path / "in.zip"
    zpath.write_bytes(zbytes)
    outdir = tmp_path / "out"
    outdir.mkdir()

    with pytest.raises(HTTPException) as e:
        ap._safe_extract_zip_to_tempdir(str(zpath), str(outdir))
    assert "No supported images" in str(e.value.detail)


def test_safe_extract_zip_to_tempdir_skips_too_large_member_and_raises_if_nothing_left(tmp_path, monkeypatch):
    monkeypatch.setattr(ap, "MAX_MEMBER_BYTES", 5)

    zbytes = _make_zip_bytes({"big.png": b"x" * 6})
    zpath = tmp_path / "in.zip"
    zpath.write_bytes(zbytes)

    outdir = tmp_path / "out"
    outdir.mkdir()

    with pytest.raises(HTTPException) as e:
        ap._safe_extract_zip_to_tempdir(str(zpath), str(outdir))

    assert "No supported images" in str(e.value.detail)
