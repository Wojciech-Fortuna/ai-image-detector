import argparse
import pandas as pd
import numpy as np
from pathlib import Path


def read_csv_auto(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        if df.shape[1] == 1:
            df = pd.read_csv(path, sep=";")
        return df
    except Exception as e:
        raise RuntimeError(f"Cannot read CSV: {path} ({e})")


def normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    return df


def require_cols(df: pd.DataFrame, path: str, cols: list[str]):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"CSV '{path}' is missing required columns: {missing}\n"
            f"It has columns: {list(df.columns)}\n"
            f"You should have at least: img_path, pce, rank"
        )


def top1_per_image(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["pce"] = pd.to_numeric(df["pce"], errors="coerce")
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    df = df.dropna(subset=["img_path", "pce"])

    df_r1 = df[df["rank"] == 1].copy()

    if df_r1["img_path"].nunique() < df["img_path"].nunique():
        df_max = (
            df.sort_values("pce", ascending=False)
            .groupby("img_path", as_index=False)
            .head(1)
        )
        have = set(df_r1["img_path"].unique())
        missing_imgs = [p for p in df_max["img_path"].unique() if p not in have]
        df_fill = df_max[df_max["img_path"].isin(missing_imgs)]
        out = pd.concat([df_r1, df_fill], ignore_index=True)
    else:
        out = df_r1

    out = (
        out.sort_values(["img_path", "pce"], ascending=[True, False])
        .groupby("img_path", as_index=False)
        .head(1)
    )
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Set a threshold as max(AI_top1)+margin and count how many REAL_top1 exceed it."
    )
    ap.add_argument("--ai-csv", required=True, help="AI CSV: img_path,fingerprint,pce,rank,...")
    ap.add_argument("--real-csv", required=True, help="REAL CSV: img_path,fingerprint,pce,rank,...")
    ap.add_argument("--margin", type=float, default=1.0, help="Margin added to max AI (default +1.0)")
    ap.add_argument(
        "--eps-zero",
        type=float,
        default=1e-12,
        help="Threshold for treating PCE as zero (default 1e-12)",
    )
    ap.add_argument("--show-passing", action="store_true", help="Print REAL images that passed the threshold")
    args = ap.parse_args()

    ai_path = args.ai_csv
    real_path = args.real_csv

    df_ai = normalize_cols(read_csv_auto(ai_path))
    df_real = normalize_cols(read_csv_auto(real_path))

    require_cols(df_ai, ai_path, ["img_path", "pce", "rank"])
    require_cols(df_real, real_path, ["img_path", "pce", "rank"])

    ai_top1 = top1_per_image(df_ai)
    real_top1 = top1_per_image(df_real)

    n_ai = ai_top1.shape[0]
    n_real = real_top1.shape[0]

    ai_top1["is_pos"] = ai_top1["pce"] > args.eps_zero
    real_top1["is_pos"] = real_top1["pce"] > args.eps_zero

    idx_max = ai_top1["pce"].idxmax()
    max_ai = float(ai_top1.loc[idx_max, "pce"])
    max_ai_img = str(ai_top1.loc[idx_max, "img_path"])

    T = max_ai + float(args.margin)

    real_pass = real_top1[real_top1["pce"] > T].copy()
    tpr = (real_pass.shape[0] / n_real) if n_real else float("nan")

    pct_nonzero_ai = (ai_top1["is_pos"].mean() * 100.0) if n_ai else float("nan")

    print("\n=== TOP1 (rank==1) statistics ===")
    print(f"AI:   n={n_ai} | nonzero(top1)={ai_top1['is_pos'].sum()} | P(PCE>0)={pct_nonzero_ai:.2f}%")
    print(f"REAL: n={n_real} | nonzero(top1)={real_top1['is_pos'].sum()}")

    print("\n=== Max AI (top1) ===")
    print(f"max_ai_top1 = {max_ai:.6f}")
    print(f"img_path    = {max_ai_img}")

    print("\n=== Threshold ===")
    print(f"margin      = {args.margin}")
    print(f"T = max_ai_top1 + margin = {T:.6f}")

    print("\n=== REAL exceeding threshold (top1) ===")
    print(f"TPR = {tpr:.3f} ({real_pass.shape[0]} / {n_real})")

    if args.show_passing and real_pass.shape[0] > 0:
        cols = [c for c in ["img_path", "fingerprint", "pce", "rank"] if c in real_pass.columns]
        real_pass = real_pass.sort_values("pce", ascending=False)
        print("\n[PASSING REAL]")
        for _, r in real_pass.iterrows():
            fp = r["fingerprint"] if "fingerprint" in real_pass.columns else "(missing fingerprint column)"
            print(f"  pce={float(r['pce']):.6f} | fp={fp} | {r['img_path']}")


if __name__ == "__main__":
    main()
