import pytest

from utils import charts


def test_get_int_parses_ints_and_falls_back():
    d = {"a": "5", "b": None, "c": "nope", "d": 3.2}
    assert charts._get_int(d, "a", 0) == 5
    assert charts._get_int(d, "b", 7) == 7
    assert charts._get_int(d, "c", 9) == 9
    assert charts._get_int(d, "d", 0) == 3
    assert charts._get_int(d, "missing", 11) == 11


def test_autopct_factory_hides_tiny_slices_and_formats_counts():
    f = charts._autopct_factory([60, 40])

    assert f(0.4) == ""
    s = f(12.3)
    assert "12.3%" in s
    assert "(" in s and ")" in s


def test_autopct_factory_total_zero_returns_empty():
    f = charts._autopct_factory([])
    assert f(50.0) == ""


def _assert_png_bytes(png: bytes):
    assert isinstance(png, (bytes, bytearray))
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert len(png) > 1000  # should not be tiny


def test_render_pie_png_no_data():
    png = charts.render_pie_png({"real": 0, "ai": 0, "unknown": 0})
    _assert_png_bytes(png)


def test_render_pie_png_omits_zero_categories_and_still_renders():
    png = charts.render_pie_png({"real": 0, "ai": 7, "unknown": 0})
    _assert_png_bytes(png)

    png2 = charts.render_pie_png({"real": 3, "ai": 0, "unknown": 2})
    _assert_png_bytes(png2)


def test_render_pie_png_handles_string_values():
    png = charts.render_pie_png({"real": "2", "ai": "1", "unknown": "0"})
    _assert_png_bytes(png)
