from __future__ import annotations

import argparse
import json
import os
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response

from methods.base import REGISTRY
from utils.io import extract_exif, load_image

from core.analyze import (
    META_METHOD_NAME,
    META_METHOD_FAST_NAME,
    META_MODEL_ONLY_NAME,
    analyze_image,
    auto_import_methods,
    run_batch,
    set_meta_model_path,
)

from utils.batch_summary import build_summary_from_rows
from utils.charts import render_pie_png

from utils.batch_pipeline import (
    replace_non_finite,
    flatten_report_to_row,
    rows_to_csv_bytes,
    compute_counts_from_rows,
)

MAX_ZIP_BYTES = 200 * 1024 * 1024  # 200MB
MAX_ZIP_FILES = 5000
MAX_MEMBER_BYTES = 60 * 1024 * 1024  # 60MB per image (uncompressed)
MAX_SINGLE_IMAGE_BYTES = MAX_MEMBER_BYTES  # 60MB

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}


def _parse_methods_json(methods_json: str) -> List[str]:
    try:
        selected = json.loads(methods_json)
        if not isinstance(selected, list) or not all(isinstance(x, str) for x in selected):
            raise ValueError("methods_json must be a JSON list of strings")
        return selected
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid methods_json: {e}")


def _validate_threshold(threshold: float) -> None:
    if threshold < 0.0 or threshold > 1.0:
        raise HTTPException(status_code=400, detail="threshold must be in [0,1]")


async def _read_upload_limited(upload: UploadFile, max_bytes: int) -> bytes:
    total = 0
    chunks: list[bytes] = []
    while True:
        chunk = await upload.read(1024 * 1024)  # 1MB
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=f"Image too large. Limit is {max_bytes} bytes (~{max_bytes/1024/1024:.0f}MB).",
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _safe_extract_zip_to_tempdir(zip_path: str, temp_dir: str) -> List[Tuple[str, str]]:
    extracted: List[Tuple[str, str]] = []

    with zipfile.ZipFile(zip_path) as zf:
        infos = zf.infolist()

        if len(infos) > MAX_ZIP_FILES:
            raise HTTPException(
                status_code=400,
                detail=f"ZIP contains too many files ({len(infos)}). Limit is {MAX_ZIP_FILES}.",
            )

        for info in infos:
            if info.is_dir():
                continue

            name = info.filename
            p = Path(name)
            if p.is_absolute() or ".." in p.parts:
                continue

            ext = p.suffix.lower()
            if ext not in IMG_EXTS:
                continue

            if info.file_size and info.file_size > MAX_MEMBER_BYTES:
                continue

            out_path = Path(temp_dir) / p.name  # flatten
            base = out_path.stem
            suffix = out_path.suffix
            n = 1
            while out_path.exists():
                out_path = Path(temp_dir) / f"{base}__{n}{suffix}"
                n += 1

            with zf.open(info) as src, open(out_path, "wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    dst.write(chunk)

            extracted.append((name, str(out_path)))

    if not extracted:
        raise HTTPException(status_code=400, detail="No supported images found in ZIP (jpg/jpeg/png/bmp).")

    return extracted


def _build_batch_zip_bytes(
    *,
    csv_bytes: bytes,
    counts: Dict[str, Any],
    summary: Dict[str, Any],
    pie_png_bytes: bytes,
) -> bytes:
    import io

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, mode="w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as zf:
        zf.writestr("results.csv", csv_bytes)
        zf.writestr("counts.json", json.dumps(counts, ensure_ascii=False, indent=2).encode("utf-8"))
        zf.writestr("summary.json", json.dumps(summary, ensure_ascii=False, indent=2).encode("utf-8"))
        zf.writestr("pie.png", pie_png_bytes)
        zf.writestr(
            "README.txt",
            (
                "Batch results package\n"
                "- results.csv: per-image flat metrics\n"
                "- counts.json: REAL/AI/UNKNOWN counts\n"
                "- summary.json: aggregated column statistics\n"
                "- pie.png: pie chart of counts\n"
            ).encode("utf-8"),
        )
    return buf.getvalue()


def create_app() -> FastAPI:
    auto_import_methods()
    if not REGISTRY:
        raise RuntimeError("No registered methods. Check methods/*.py and decorators.")

    app = FastAPI(title="AI-Image Detector API", version="1.0")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            "http://localhost:5173",
            "http://127.0.0.1:5173",
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.get("/")
    def root():
        return {"ok": True, "endpoints": ["/health", "/methods", "/analyze", "/analyze_zip"]}

    @app.get("/health")
    def health():
        return {"ok": True}

    @app.get("/methods")
    def methods():
        out: List[Dict[str, Any]] = []

        for reg_key, method in REGISTRY.items():
            key = getattr(method, "name", reg_key)
            out.append(
                {
                    "key": key,
                    "description": getattr(method, "description", ""),
                    "how_text": getattr(method, "how_text", ""),
                    "how_title": getattr(method, "how_title", ""),
                }
            )

        out.append({"key": META_METHOD_NAME, "description": "CatBoost stacker over base detectors."})
        out.append({"key": META_METHOD_FAST_NAME, "description": "CatBoost stacker (fast): skips PRNU."})
        out.append(
            {
                "key": META_MODEL_ONLY_NAME,
                "description": "Meta-model only: runs only base detectors used by meta-model; skips C2PA/PRNU gates.",
            }
        )

        uniq: Dict[str, Dict[str, Any]] = {}
        for m in out:
            uniq[str(m.get("key", ""))] = m

        return {"methods": sorted(uniq.values(), key=lambda x: str(x.get("key", "")))}

    @app.post("/analyze")
    async def analyze(
        file: UploadFile = File(...),
        threshold: float = Form(0.5),
        methods_json: str = Form("[]"),
    ):
        selected = _parse_methods_json(methods_json)
        if not selected:
            selected = [META_METHOD_NAME]
        _validate_threshold(float(threshold))

        suffix = Path(file.filename or "upload.png").suffix or ".png"
        tmp_path: str | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp_path = tmp.name
                content = await _read_upload_limited(file, MAX_SINGLE_IMAGE_BYTES)
                tmp.write(content)

            try:
                with open(tmp_path, "rb") as fh:
                    img = load_image(fh)
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"Uploaded file is not a supported image: {e}")

            exif = extract_exif(img)

            report = analyze_image(
                img,
                threshold=float(threshold),
                selected_methods=selected,
                exif=exif,
                image_path=tmp_path,
            )

            safe_report = replace_non_finite(report)
            return JSONResponse(content=safe_report)

        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Analysis failed: {type(e).__name__}: {e}")
        finally:
            if tmp_path and os.path.isfile(tmp_path):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass

    @app.post("/analyze_zip")
    async def analyze_zip(
        file: UploadFile = File(...),
        threshold: float = Form(0.5),
        methods_json: str = Form("[]"),
    ):
        selected = _parse_methods_json(methods_json)
        if not selected:
            selected = [META_METHOD_NAME]
        _validate_threshold(float(threshold))

        fname = (file.filename or "").lower()
        if not (
            fname.endswith(".zip")
            or (file.content_type or "") in {"application/zip", "application/x-zip-compressed"}
        ):
            if not fname.endswith(".zip"):
                raise HTTPException(status_code=400, detail="Expected a .zip file.")

        try:
            content_length = file.headers.get("content-length")
        except Exception:
            content_length = None

        if content_length:
            try:
                if int(content_length) > MAX_ZIP_BYTES:
                    raise HTTPException(status_code=413, detail=f"ZIP too large. Limit is {MAX_ZIP_BYTES} bytes.")
            except ValueError:
                pass

        zip_tmp_path: str | None = None
        tmp_dir: str | None = None

        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
                zip_tmp_path = tmp.name
                total = 0
                while True:
                    chunk = await file.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > MAX_ZIP_BYTES:
                        raise HTTPException(status_code=413, detail=f"ZIP too large. Limit is {MAX_ZIP_BYTES} bytes.")
                    tmp.write(chunk)

            tmp_dir = tempfile.mkdtemp(prefix="batch_zip_")
            extracted = _safe_extract_zip_to_tempdir(zip_tmp_path, tmp_dir)

            rows: List[Dict[str, Any]] = []
            for path_in_zip, extracted_path in extracted:
                try:
                    with open(extracted_path, "rb") as fh:
                        img = load_image(fh)
                    exif = extract_exif(img)

                    report = analyze_image(
                        img,
                        threshold=float(threshold),
                        selected_methods=selected,
                        exif=exif,
                        image_path=extracted_path,
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
                    rows.append(flatten_report_to_row(report, path_in_zip))

                except Exception as e:
                    err_msg = f"{type(e).__name__}: {e}"
                    rows.append(
                        {
                            "path": path_in_zip,
                            "result_label": "",
                            "result_score_ai": "",
                            "error": 1,
                            "error_msg": err_msg,
                        }
                    )

            csv_bytes = rows_to_csv_bytes(rows)

            counts = compute_counts_from_rows(rows, float(threshold))
            summary = build_summary_from_rows(
                rows,
                error_column="error",
                score_column="result_score_ai",
                id_column="path",
            )
            pie_png = render_pie_png(counts)

            zip_bytes = _build_batch_zip_bytes(
                csv_bytes=csv_bytes,
                counts=counts,
                summary=summary,
                pie_png_bytes=pie_png,
            )

            return Response(
                content=zip_bytes,
                media_type="application/zip",
                headers={"Content-Disposition": 'attachment; filename="batch_results.zip"'},
            )

        except HTTPException:
            raise
        except zipfile.BadZipFile:
            raise HTTPException(status_code=400, detail="Invalid ZIP file.")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Batch analysis failed: {type(e).__name__}: {e}")
        finally:
            if zip_tmp_path and os.path.isfile(zip_tmp_path):
                try:
                    os.unlink(zip_tmp_path)
                except OSError:
                    pass
            if tmp_dir and os.path.isdir(tmp_dir):
                try:
                    for p in Path(tmp_dir).glob("*"):
                        try:
                            if p.is_file():
                                p.unlink()
                        except OSError:
                            pass
                    os.rmdir(tmp_dir)
                except OSError:
                    pass

    return app


app = create_app()


def main():
    parser = argparse.ArgumentParser(
        description="Batch CLI for AI-Image Detector (API is started via uvicorn)."
    )

    parser.add_argument(
        "input_dir",
        type=str,
        help="Folder with images for batch analysis",
    )
    parser.add_argument("--out", type=str, default="./out", help="Output folder for reports and CSV")
    parser.add_argument("--threshold", type=float, default=0.5, help="AI/REAL decision threshold")
    parser.add_argument(
        "--methods",
        type=str,
        default=None,
        help="Comma-separated list of methods (e.g., ela,fft,attrib_generator)",
    )

    args = parser.parse_args()

    selected_methods: List[str] | None
    if args.methods:
        selected_methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    else:
        selected_methods = [META_METHOD_NAME]

    csv_path = run_batch(
        Path(args.input_dir),
        Path(args.out),
        threshold=args.threshold,
        selected_methods=selected_methods,
    )
    print(f"Batch summary saved: {csv_path}")


if __name__ == "__main__":
    main()
