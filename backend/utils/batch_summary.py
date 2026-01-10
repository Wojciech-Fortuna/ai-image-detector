from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

_NA_STRINGS = {
    "",
    "nan",
    "NaN",
    "none",
    "None",
    "null",
    "NULL",
    "na",
    "NA",
    "N/A",
    "n/a",
    "-",
}


def _as_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, bool):
        return "true" if x else "false"
    return str(x)


def _to_str_list(values: Iterable[Any]) -> List[str]:
    return [_as_str(v) for v in values]


def _is_na_string(s: str) -> bool:
    return s.strip() in _NA_STRINGS


def _to_float_or_nan(x: Any) -> float:
    if x is None:
        return float("nan")
    if isinstance(x, (int, float)) and not isinstance(x, bool):
        try:
            v = float(x)
        except Exception:
            return float("nan")
        return v if math.isfinite(v) else float("nan")

    s = _as_str(x).strip()
    if _is_na_string(s):
        return float("nan")
    try:
        v = float(s)
    except Exception:
        return float("nan")
    return v if math.isfinite(v) else float("nan")


def _finite_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


def _is_mostly_numeric(
    values: List[Any],
    *,
    min_numeric_frac: float = 0.7,
    min_numeric_count: int = 20,
) -> bool:
    if not values:
        return False
    arr = np.array([_to_float_or_nan(v) for v in values], dtype=float)
    ok = _finite_mask(arr)
    count = int(ok.sum())
    total = int(arr.size)
    frac = float(count / max(total, 1))
    return (count >= int(min_numeric_count)) and (frac >= float(min_numeric_frac))


def _summarize_numeric(
    values: List[Any],
    *,
    percentiles: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99),
) -> Dict[str, Any]:
    arr = np.array([_to_float_or_nan(v) for v in values], dtype=float)
    ok = _finite_mask(arr)
    present = arr[ok]

    n_total = int(arr.size)
    n = int(present.size)

    out: Dict[str, Any] = {
        "present_count": n,
        "missing_count": n_total - n,
        "missing_rate": float((n_total - n) / max(n_total, 1)),
    }
    if n == 0:
        out["status"] = "all_missing"
        return out

    out.update(
        {
            "mean": float(present.mean()),
            "std": float(present.std(ddof=0)),
            "min": float(present.min()),
            "max": float(present.max()),
        }
    )

    qs = np.quantile(present, list(percentiles))
    for p, v in zip(percentiles, qs):
        out[f"p{int(round(p * 100)):02d}"] = float(v)

    return out


def _summarize_categorical(values: List[Any], *, top_k: int = 30) -> Dict[str, Any]:
    s = _to_str_list(values)
    s2 = [v.strip() for v in s]

    cleaned: List[str] = []
    for v in s2:
        if v == "":
            cleaned.append("")
            continue
        cleaned.append("" if _is_na_string(v) else v)

    total = int(len(cleaned))
    empty_count = int(sum(1 for v in cleaned if v == ""))
    unique_count = int(len(set(cleaned)))

    counts: Dict[str, int] = {}
    for v in cleaned:
        counts[v] = counts.get(v, 0) + 1

    top_items = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[: int(top_k)]
    top_values = [
        {"value": str(k), "count": int(v), "rate": float(int(v) / max(total, 1))}
        for k, v in top_items
    ]

    return {
        "unique_count": unique_count,
        "empty_count": empty_count,
        "empty_rate": float(empty_count / max(total, 1)),
        "top_values": top_values,
    }


def _detect_error(values: List[Any]) -> np.ndarray:
    """
    Treat rows as error when:
      - value is non-empty AND value != '0'
    This matches your API CSV convention.
    """
    s = _to_str_list(values)
    s = [v.strip() for v in s]
    return np.array([(v != "" and v != "0") for v in s], dtype=bool)


def _pick_top_bottom_examples(
    rows_ok: List[Dict[str, Any]],
    *,
    score_column: str,
    id_column: Optional[str],
    top_k: int,
) -> Dict[str, Any]:
    scores: List[Tuple[float, int]] = []
    for i, r in enumerate(rows_ok):
        s = _to_float_or_nan(r.get(score_column))
        if math.isfinite(s):
            scores.append((float(s), i))

    if not scores:
        return {"status": "all_missing_or_non_numeric", "top": [], "bottom": []}

    scores_sorted = sorted(scores, key=lambda t: t[0])
    bottom = scores_sorted[:top_k]
    top = list(reversed(scores_sorted[-top_k:]))

    def _mk(items: List[Tuple[float, int]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for sc, idx in items:
            item: Dict[str, Any] = {"score": float(sc)}
            if id_column and id_column in rows_ok[idx]:
                item["id"] = _as_str(rows_ok[idx].get(id_column))
            out.append(item)
        return out

    return {"top": _mk(top), "bottom": _mk(bottom)}


def build_summary_from_rows(
    rows: List[Dict[str, Any]],
    *,
    error_column: Optional[str] = None,
    score_column: Optional[str] = None,
    id_column: Optional[str] = None,
    top_k_examples: int = 10,
    min_numeric_frac: float = 0.7,
    min_numeric_count: int = 20,
) -> Dict[str, Any]:
    n_total = int(len(rows))
    cols: List[str] = sorted({k for r in rows for k in r.keys()})

    if error_column and error_column in cols:
        is_error = _detect_error([r.get(error_column, "") for r in rows])
    else:
        is_error = np.zeros((n_total,), dtype=bool)

    rows_ok = [r for r, bad in zip(rows, is_error) if not bool(bad)]
    n_err = int(is_error.sum())
    n_ok = int(len(rows_ok))

    summary: Dict[str, Any] = {
        "meta": {
            "rows_total": n_total,
            "rows_ok": n_ok,
            "rows_error": n_err,
            "error_rate": float(n_err / max(n_total, 1)),
            "error_column": error_column if (error_column and error_column in cols) else None,
            "score_column": score_column if (score_column and score_column in cols) else None,
            "id_column": id_column if (id_column and id_column in cols) else None,
        },
        "columns": {},
        "examples": {},
    }

    for col in cols:
        src_rows = rows_ok if rows_ok else rows
        values = [r.get(col, "") for r in src_rows]

        if _is_mostly_numeric(
            values,
            min_numeric_frac=float(min_numeric_frac),
            min_numeric_count=int(min_numeric_count),
        ):
            summary["columns"][col] = {"type": "numeric", "stats": _summarize_numeric(values)}
        else:
            summary["columns"][col] = {"type": "categorical", "stats": _summarize_categorical(values)}

    if score_column and score_column in cols and rows_ok:
        summary["examples"] = _pick_top_bottom_examples(
            rows_ok,
            score_column=score_column,
            id_column=id_column if (id_column and id_column in cols) else None,
            top_k=int(top_k_examples),
        )
        if summary["examples"].get("status") == "all_missing_or_non_numeric":
            summary["meta"]["score_status"] = "all_missing_or_non_numeric"
    else:
        summary["examples"] = {}

    return summary
