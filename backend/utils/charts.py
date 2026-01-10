from __future__ import annotations

import io
from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt


def _get_int(d: Dict[str, Any], key: str, default: int = 0) -> int:
    try:
        v = d.get(key, default)
        if v is None:
            return int(default)
        return int(v)
    except Exception:
        return int(default)


def _autopct_factory(values: List[int]):
    total = int(sum(values))

    def _autopct(pct: float) -> str:
        if total <= 0:
            return ""
        count = int(round(pct * total / 100.0))
        if pct < 0.5:
            return ""
        return f"{pct:.1f}%\n({count})"

    return _autopct


def render_pie_png(counts: Dict[str, Any]) -> bytes:
    real = _get_int(counts, "real", 0)
    ai = _get_int(counts, "ai", 0)
    unknown = _get_int(counts, "unknown", 0)

    pairs: List[Tuple[str, int]] = []
    if real > 0:
        pairs.append(("REAL", real))
    if ai > 0:
        pairs.append(("AI", ai))
    if unknown > 0:
        pairs.append(("UNKNOWN", unknown))

    labels = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    total = int(real + ai + unknown)

    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=160)

    if total <= 0:
        ax.text(
            0.5,
            0.5,
            "No data",
            ha="center",
            va="center",
            fontsize=14,
            transform=ax.transAxes,
        )
        ax.set_axis_off()
    else:
        if not values:
            ax.text(
                0.5,
                0.5,
                "No data",
                ha="center",
                va="center",
                fontsize=14,
                transform=ax.transAxes,
            )
            ax.set_axis_off()
        else:
            ax.pie(
                values,
                labels=[f"{lab}" for lab in labels],
                autopct=_autopct_factory(values),
                startangle=90,
                counterclock=False,
                wedgeprops={"linewidth": 1.0, "edgecolor": "white"},
                textprops={"fontsize": 10},
            )
            ax.axis("equal")
            ax.set_title(f"Batch verdicts (n={total})", fontsize=12)

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)

    return buf.getvalue()
