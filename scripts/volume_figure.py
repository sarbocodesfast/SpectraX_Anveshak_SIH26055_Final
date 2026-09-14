#!/usr/bin/env python
"""The architecture x volume interaction, drawn so the reversal is visible.

Two lines that cross is the whole finding, and a table of four numbers does
not show a crossing. The figure exists to make one thing obvious: the
architecture ranking depends on the training volume, so neither architecture
is "better" without naming the volume.

Every point is scored on the **same** held-out set, built from seeds no arm
trains on. That is what makes the two lines comparable at all; the previous
corpus comparison failed precisely because its arms validated on different
sets.

The one-seed caveat is drawn into the figure rather than left to the caption.
A reader who takes only the image away should still know that this is an
effect size and not a significance claim.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPORTS = REPO_ROOT / "reports"

#: Consistent with reports/arch_latency.md: the production architecture, the
#: timing-disqualified one, and the GPU-only alternative.
COLOUR = {"transformer": "#1f77b4", "gru": "#d62728", "tcn": "#2ca02c"}


def plot(arms: dict, base: float, out: Path) -> None:
    """Left: the crossing. Right: the volume gain per architecture."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12.5, 5.2),
                                   gridspec_kw={"width_ratios": [1.35, 1]})

    series: dict[str, list[tuple[int, float]]] = {}
    for v in arms.values():
        series.setdefault(v["arch"], []).append((v["vol"], v["ap"]))
    for arch in series:
        series[arch].sort()

    # ---- left: the reversal --------------------------------------------
    ax1.axhline(base, color="0.6", ls=":", lw=1.4, zorder=1)
    # Anchored right: the gru's 40-episode point sits just above the base rate
    # and its value label would overprint this note on the left.
    ax1.annotate(f"base rate {base:.4f} — AP here is no skill", (200, base),
                 textcoords="offset points", xytext=(0, 7), fontsize=8,
                 color="0.35", ha="right")
    for arch, pts in series.items():
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        style = "-o" if len(pts) > 1 else "D"
        ax1.plot(xs, ys, style, color=COLOUR.get(arch, "0.4"), lw=2.4, ms=9,
                 zorder=3, label=arch)
        # gru and tcn land within 0.005 AP at 200 episodes, so a shared
        # offset would overprint them. Push each architecture's labels to a
        # different side rather than shrinking the font until both are unread.
        dx, dy, ha = {"transformer": (0, -17, "center"),
                      "gru": (10, 4, "left"),
                      "tcn": (10, -14, "left")}.get(arch, (0, -16, "center"))
        for x, y in pts:
            ax1.annotate(f"{y:.4f}", (x, y), textcoords="offset points",
                         xytext=(dx, dy), ha=ha, fontsize=8.5,
                         color=COLOUR.get(arch, "0.4"))

    t = dict(series.get("transformer", []))
    g = dict(series.get("gru", []))
    if 40 in t and 40 in g:
        ax1.annotate("", xy=(40, t[40]), xytext=(40, g[40]),
                     arrowprops={"arrowstyle": "<->", "color": "0.4", "lw": 1.3})
        ax1.text(43, (t[40] + g[40]) / 2,
                 f"transformer ahead\n{t[40] - g[40]:+.3f} AP",
                 fontsize=8.5, color="0.3", va="center")
    if 200 in t and 200 in g:
        # Pulled below the lines and leadered: at 200 episodes the gru and tcn
        # value labels already occupy the space beside the points.
        ax1.annotate(f"gru ahead {t[200] - g[200]:+.3f} AP",
                     xy=(200, (t[200] + g[200]) / 2), xytext=(-18, -40),
                     textcoords="offset points", fontsize=8.5, color="0.3",
                     ha="right",
                     arrowprops={"arrowstyle": "-", "color": "0.6", "lw": 1})

    ax1.set_xscale("log")
    ax1.set_xticks([40, 200])
    ax1.set_xticklabels(["40", "200"])
    ax1.minorticks_off()
    ax1.set_xlim(30, 300)
    ax1.set_xlabel("training episodes (log scale)")
    ax1.set_ylabel("average precision, shared held-out set")
    ax1.set_title("The ranking reverses with training volume")
    # upper left is the only quadrant no series or annotation occupies
    ax1.legend(fontsize=9, loc="upper left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    # ---- right: the gain -----------------------------------------------
    gains, labels, cols = [], [], []
    for arch in ("transformer", "gru"):
        pts = dict(series.get(arch, []))
        if 40 in pts and 200 in pts:
            gains.append(pts[200] - pts[40])
            labels.append(arch)
            cols.append(COLOUR[arch])
    bars = ax2.bar(labels, gains, color=cols, width=0.55)
    for b, v in zip(bars, gains, strict=True):
        ax2.annotate(f"{v:+.4f}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=10, color="0.25")
    if len(gains) == 2:
        ratio = gains[1] / gains[0] if gains[0] else float("inf")
        ax2.set_title(f"Volume gain, 40 → 200\ngru gains {ratio:.1f}× the transformer")
        ax2.annotate(f"interaction\nI = {gains[0] - gains[1]:+.4f} AP",
                     (0.5, max(gains) * 0.55), ha="center", fontsize=10,
                     color="0.25",
                     bbox={"boxstyle": "round,pad=0.4", "fc": "#fff3cd",
                           "ec": "#d9b45b"})
    ax2.set_ylabel("Δ average precision")
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("Architecture and training volume interact — "
                 "ONE SEED PER CELL, effect size only, no significance claimed",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(REPORTS / "volume_control.json"))
    ap.add_argument("--out", default=str(REPORTS / "volume_interaction.png"))
    args = ap.parse_args()

    src = Path(args.json)
    if not src.is_file():
        raise SystemExit(f"no results at {src}")
    d = json.loads(src.read_text(encoding="utf-8"))
    plot(d["arms"], float(d["base_rate"]), Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
