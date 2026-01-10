import os
import argparse
import sqlite3
from glob import glob
from typing import Iterable, Tuple

import face_recognition


DEFAULT_EXTS = (".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff")


def iter_image_files(root_dir: str, exts=DEFAULT_EXTS) -> Iterable[str]:
    for dirpath, _, filenames in os.walk(root_dir):
        for fn in filenames:
            if fn.lower().endswith(exts):
                yield os.path.join(dirpath, fn)


def connect_db(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS faces_cache (
            path TEXT PRIMARY KEY,
            contains_face INTEGER NOT NULL,
            model TEXT NOT NULL,
            n_faces INTEGER NOT NULL,
            error TEXT,
            updated_at DATETIME DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_faces_cache_contains_face ON faces_cache(contains_face);")
    return conn


def is_cached(conn: sqlite3.Connection, rel_path: str) -> bool:
    cur = conn.execute("SELECT 1 FROM faces_cache WHERE path = ? LIMIT 1;", (rel_path,))
    return cur.fetchone() is not None


def upsert_result(
    conn: sqlite3.Connection,
    rel_path: str,
    contains_face: int,
    model: str,
    n_faces: int,
    error: str | None,
):
    conn.execute(
        """
        INSERT INTO faces_cache(path, contains_face, model, n_faces, error)
        VALUES(?, ?, ?, ?, ?)
        ON CONFLICT(path) DO UPDATE SET
            contains_face=excluded.contains_face,
            model=excluded.model,
            n_faces=excluded.n_faces,
            error=excluded.error,
            updated_at=CURRENT_TIMESTAMP;
        """,
        (rel_path, contains_face, model, n_faces, error),
    )


def detect_faces(image_path: str, model: str = "hog") -> Tuple[int, int]:
    image = face_recognition.load_image_file(image_path)
    locations = face_recognition.face_locations(image, model=model)
    n = len(locations)
    return (1 if n > 0 else 0), n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data", help="Katalog z danymi (np. data)")
    ap.add_argument("--db", default="faces_cache.sqlite", help="Ścieżka do pliku SQLite")
    ap.add_argument("--model", choices=["hog", "cnn"], default="hog", help="Model detekcji twarzy")
    ap.add_argument("--commit-every", type=int, default=50, help="Commit co N obrazów")
    ap.add_argument("--force", action="store_true", help="Przelicz nawet jeśli jest w cache")
    ap.add_argument("--limit", type=int, default=0, help="Opcjonalny limit obrazów (0=bez limitu)")
    args = ap.parse_args()

    root_dir = os.path.abspath(args.root)
    db_path = os.path.abspath(args.db)

    print(f"[INFO] root_dir = {root_dir}")
    print(f"[INFO] db_path  = {db_path}")
    print(f"[INFO] model    = {args.model}")

    conn = connect_db(db_path)

    processed = 0
    skipped = 0
    errors = 0

    try:
        conn.execute("BEGIN;")
        for idx, abs_path in enumerate(iter_image_files(root_dir)):
            if args.limit and processed >= args.limit:
                break

            rel_path = os.path.relpath(abs_path, root_dir).replace("\\", "/")

            if (not args.force) and is_cached(conn, rel_path):
                skipped += 1
                continue

            if (processed + 1) % 100 == 0:
                print(f"[INFO] Processing {processed+1}: {rel_path}")

            try:
                contains, n_faces = detect_faces(abs_path, model=args.model)
                upsert_result(conn, rel_path, contains, args.model, n_faces, None)
            except Exception as e:
                errors += 1
                upsert_result(conn, rel_path, 0, args.model, 0, repr(e))

            processed += 1

            if processed % args.commit_every == 0:
                conn.commit()
                conn.execute("BEGIN;")
                print(f"[INFO] Committed. processed={processed} skipped={skipped} errors={errors}")

        conn.commit()
    finally:
        conn.close()

    print(f"[DONE] processed={processed} skipped={skipped} errors={errors}")
    print(f"[DONE] Cache saved in: {db_path}")


if __name__ == "__main__":
    main()
