from __future__ import annotations

import csv
import io
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional


def replace_non_finite(x: Any) -> Any:
    if isinstance(x, float):
        return x if math.isfinite(x) else None
    if isinstance(x, dict):
        return {k: replace_non_finite(v) for k, v in x.items()}
    if isinstance(x, list):
        return [replace_non_finite(v) for v in x]
    if isinstance(x, tuple):
        return [replace_non_finite(v) for v in x]
    return x


def flatten_report_to_row(report: Dict[str, Any], image_id: str) -> Dict[str, Any]:
    row: Dict[str, Any] = {}
    row["path"] = image_id

    result = (report or {}).get("result", {}) or {}
    row["result_label"] = result.get("label", "")
    row["result_score_ai"] = result.get("score_ai", "")

    comps = result.get("components", {}) or {}
    if isinstance(comps, dict):
        for k, v in comps.items():
            row[f"component__{k}"] = v

    metrics = (report or {}).get("metrics", {}) or {}
    if isinstance(metrics, dict):
        for k, v in metrics.items():
            row[f"metric__{k}"] = v

    attribution = (report or {}).get("attribution", {}) or {}
    if isinstance(attribution, dict):
        for stage_name, stage in attribution.items():
            if not isinstance(stage, dict):
                continue
            row[f"attrib__{stage_name}__label"] = stage.get("label", "")
            row[f"attrib__{stage_name}__confidence"] = stage.get("confidence", "")
            stage_metrics = stage.get("metrics", {}) or {}
            if isinstance(stage_metrics, dict):
                for mk, mv in stage_metrics.items():
                    if isinstance(mv, (str, int, float, bool)) or mv is None:
                        row[f"attrib__{stage_name}__metric__{mk}"] = mv

    if "error" in report:
        row["error"] = report.get("error", "")
    if "error_msg" in report:
        row["error_msg"] = report.get("error_msg", "")

    return row


def rows_to_csv_bytes(rows: List[Dict[str, Any]]) -> bytes:
    core = ["path", "result_label", "result_score_ai", "error", "error_msg"]
    keys = set()
    for r in rows:
        keys.update(r.keys())

    rest = sorted([k for k in keys if k not in core])
    fieldnames = [k for k in core if k in keys] + rest

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k, "") for k in fieldnames})

    return buf.getvalue().encode("utf-8")


def parse_finite_float(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except Exception:
        return None
    if not math.isfinite(v):
        return None
    return v


def compute_counts_from_rows(rows: List[Dict[str, Any]], threshold: float) -> Dict[str, Any]:
    real = 0
    ai = 0
    unknown = 0
    err = 0

    for r in rows:
        e = str(r.get("error", "")).strip()
        if e and e != "0":
            err += 1

        s = parse_finite_float(r.get("result_score_ai", None))
        if s is None:
            unknown += 1
        else:
            if s >= float(threshold):
                ai += 1
            else:
                real += 1

    total = len(rows)
    return {
        "total": total,
        "real": real,
        "ai": ai,
        "unknown": unknown,
        "error_count": err,
        "threshold": float(threshold),
    }


def write_batch_artifacts(
    *,
    out_dir: Path,
    rows: List[Dict[str, Any]],
    threshold: float,
    summary: Dict[str, Any],
    pie_png_bytes: bytes,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_bytes = rows_to_csv_bytes(rows)
    counts = compute_counts_from_rows(rows, float(threshold))

    results_csv = out_dir / "results.csv"
    counts_json = out_dir / "counts.json"
    summary_json = out_dir / "summary.json"
    pie_png = out_dir / "pie.png"
    readme = out_dir / "README.txt"

    results_csv.write_bytes(csv_bytes)
    counts_json.write_text(json.dumps(counts, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    pie_png.write_bytes(pie_png_bytes)
    readme.write_text(
        "Batch results folder\n"
        "- results.csv: per-image flat metrics\n"
        "- counts.json: REAL/AI/UNKNOWN counts\n"
        "- summary.json: aggregated column statistics\n"
        "- pie.png: pie chart of counts\n",
        encoding="utf-8",
    )

    return {
        "results_csv": results_csv,
        "counts_json": counts_json,
        "summary_json": summary_json,
        "pie_png": pie_png,
        "readme": readme,
    }
