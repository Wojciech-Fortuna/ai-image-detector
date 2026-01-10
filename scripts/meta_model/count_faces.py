import sqlite3
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="faces_cache.sqlite", help="Ścieżka do bazy SQLite")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    try:
        total = conn.execute(
            "SELECT COUNT(*) FROM faces_cache;"
        ).fetchone()[0]

        with_face = conn.execute(
            "SELECT COUNT(*) FROM faces_cache WHERE contains_face = 1;"
        ).fetchone()[0]

        without_face = conn.execute(
            "SELECT COUNT(*) FROM faces_cache WHERE contains_face = 0;"
        ).fetchone()[0]

        errors = conn.execute(
            "SELECT COUNT(*) FROM faces_cache WHERE error IS NOT NULL;"
        ).fetchone()[0]

    finally:
        conn.close()

    print("=== FACE CACHE STATS ===")
    print(f"Total images     : {total}")
    print(f"With face        : {with_face}")
    print(f"Without face     : {without_face}")
    print(f"Errors           : {errors}")

    if total > 0:
        print(f"Face ratio       : {with_face / total:.3%}")


if __name__ == "__main__":
    main()
