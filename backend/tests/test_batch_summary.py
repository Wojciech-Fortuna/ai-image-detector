import math

import numpy as np
import pytest

from utils import batch_summary as bs


def test_as_str_and_na_strings():
    assert bs._as_str(None) == ""
    assert bs._as_str(True) == "true"
    assert bs._as_str(False) == "false"
    assert bs._is_na_string("NA")
    assert bs._is_na_string(" n/a ")
    assert not bs._is_na_string("something")


def test_to_float_or_nan_handles_numbers_and_na_strings():
    assert bs._to_float_or_nan(1) == pytest.approx(1.0)
    assert bs._to_float_or_nan(1.5) == pytest.approx(1.5)
    assert math.isnan(bs._to_float_or_nan(True))
    assert math.isnan(bs._to_float_or_nan("NA"))
    assert math.isnan(bs._to_float_or_nan("n/a"))
    assert math.isnan(bs._to_float_or_nan(""))
    assert bs._to_float_or_nan("0.25") == pytest.approx(0.25)
    assert math.isnan(bs._to_float_or_nan("not-a-number"))
    assert math.isnan(bs._to_float_or_nan(float("inf")))
    assert math.isnan(bs._to_float_or_nan(float("nan")))


def test_is_mostly_numeric_thresholds():
    vals = [str(i / 10) for i in range(20)]
    assert bs._is_mostly_numeric(vals) is True

    vals2 = [str(i) for i in range(19)] + ["NA"]
    assert bs._is_mostly_numeric(vals2) is False

    assert bs._is_mostly_numeric(vals2, min_numeric_count=10, min_numeric_frac=0.5) is True

    vals3 = [str(i) for i in range(10)] + ["NA"] * 20
    assert bs._is_mostly_numeric(vals3, min_numeric_count=10, min_numeric_frac=0.7) is False


def test_summarize_numeric_all_missing():
    out = bs._summarize_numeric(["NA", "", None, "nan"])
    assert out["present_count"] == 0
    assert out["status"] == "all_missing"
    assert out["missing_count"] == 4
    assert out["missing_rate"] == pytest.approx(1.0)


def test_summarize_numeric_basic_stats_and_percentiles():
    vals = [0, 1, 2, 3, 4, 5]
    out = bs._summarize_numeric(vals, percentiles=(0.5, 0.9))
    assert out["present_count"] == 6
    assert out["missing_count"] == 0
    assert out["min"] == 0.0
    assert out["max"] == 5.0
    assert out["mean"] == pytest.approx(2.5)
    assert "p50" in out and "p90" in out
    assert out["p50"] == pytest.approx(np.quantile(np.array(vals, dtype=float), 0.5))
    assert out["p90"] == pytest.approx(np.quantile(np.array(vals, dtype=float), 0.9))


def test_summarize_categorical_counts_and_na_cleanup():
    vals = ["cat", "dog", "dog", "", "NA", "n/a", "cat"]
    out = bs._summarize_categorical(vals, top_k=10)
    assert out["unique_count"] >= 2
    assert out["empty_count"] == 3
    assert out["empty_rate"] == pytest.approx(3 / 7)

    top = out["top_values"]
    assert any(item["value"] == "" for item in top)
    d = {i["value"]: i["count"] for i in top}
    assert d["dog"] == 2
    assert d["cat"] == 2


def test_detect_error_rules():
    vals = ["", "0", "1", "err", 0, 2, None]
    mask = bs._detect_error(vals)
    assert mask.tolist() == [False, False, True, True, False, True, False]


def test_pick_top_bottom_examples_basic():
    rows = [{"path": f"p{i}", "result_score_ai": i / 10} for i in range(10)]
    out = bs._pick_top_bottom_examples(rows, score_column="result_score_ai", id_column="path", top_k=3)
    assert len(out["top"]) == 3
    assert len(out["bottom"]) == 3
    assert out["top"][0]["score"] == pytest.approx(0.9)
    assert out["top"][0]["id"] == "p9"
    assert out["bottom"][0]["score"] == pytest.approx(0.0)
    assert out["bottom"][0]["id"] == "p0"


def test_pick_top_bottom_examples_all_missing_or_non_numeric():
    rows = [{"path": "a", "result_score_ai": "NA"}, {"path": "b", "result_score_ai": ""}]
    out = bs._pick_top_bottom_examples(rows, score_column="result_score_ai", id_column="path", top_k=5)
    assert out["status"] == "all_missing_or_non_numeric"
    assert out["top"] == []
    assert out["bottom"] == []


def test_build_summary_from_rows_meta_and_columns_and_examples():
    rows = []
    for i in range(25):
        rows.append(
            {
                "path": f"img{i}.png",
                "result_score_ai": i / 24,
                "result_label": "AI" if i > 12 else "REAL",
                "error": "0",
                "metric__x": i,
                "component__m": "NA" if i < 2 else i / 10,
            }
        )

    rows.append({"path": "bad1", "result_score_ai": 0.99, "error": "1", "result_label": "AI"})
    rows.append({"path": "bad2", "result_score_ai": 0.01, "error": "boom", "result_label": "REAL"})

    summ = bs.build_summary_from_rows(
        rows,
        error_column="error",
        score_column="result_score_ai",
        id_column="path",
        top_k_examples=5,
    )

    meta = summ["meta"]
    assert meta["rows_total"] == 27
    assert meta["rows_error"] == 2
    assert meta["rows_ok"] == 25
    assert meta["error_rate"] == pytest.approx(2 / 27)
    assert meta["error_column"] == "error"
    assert meta["score_column"] == "result_score_ai"
    assert meta["id_column"] == "path"

    cols = summ["columns"]
    assert "result_score_ai" in cols
    assert cols["result_score_ai"]["type"] == "numeric"
    assert "result_label" in cols
    assert cols["result_label"]["type"] == "categorical"

    ex = summ["examples"]
    assert "top" in ex and "bottom" in ex
    assert len(ex["top"]) == 5
    assert len(ex["bottom"]) == 5
    assert ex["top"][0]["id"] == "img24.png"
    assert ex["top"][0]["score"] == pytest.approx(1.0)
    assert ex["bottom"][0]["id"] == "img0.png"
    assert ex["bottom"][0]["score"] == pytest.approx(0.0)


def test_build_summary_no_error_column_no_examples_when_missing_score():
    rows = [{"a": 1}, {"a": 2}]
    summ = bs.build_summary_from_rows(rows, error_column="error", score_column="result_score_ai", id_column="path")
    assert summ["meta"]["error_column"] is None
    assert summ["examples"] == {}
    assert summ["columns"]["a"]["type"] == "categorical"
