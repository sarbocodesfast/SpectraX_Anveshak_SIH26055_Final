#!/usr/bin/env python
"""Why the more accurate predictor does not ship.

The GRU has the highest average precision of any architecture measured. It is
not the production model. A table states that; a figure should make it
obvious without being read closely.

So the 3 ms budget is drawn as a hard line, and everything is measured
against it. Bars that cross it are not "worse" -- they are excluded, and the
figure says so in those words rather than leaving a reader to infer severity
from a length.

Two details the drawing has to get right or it misleads:

* **The whole decision, not the forward pass.** The budget covers belief
  update, scoring and argmax as well. Plotting the forward pass alone credits
  every architecture with ~0.6 ms it does not have, which is enough to move
  the GRU from infeasible to apparently fine on GPU.
* **TCN is not generally deployable.** It passes on GPU and fails on CPU by
  over three times. A single "TCN passes" bar would be a false claim, so CPU
  and GPU are separate bars and the label says GPU-only.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPORTS = REPO_ROOT / "reports"

#: Measured average precision on the shared held-out set at 200 episodes,
#: from reports/volume_control.json. Carried here so the figure can show
#: accuracy and feasibility together -- the whole point is that the ordering
#: differs between them.
AP_200 = {"transformer": 0.5282, "gru": 0.5714, "tcn": 0.5671}

DECISION_LABEL = {"transformer": "Transformer", "gru": "GRU", "tcn": "TCN"}


def plot(rep: dict, overhead: float, out: Path) -> None:
    """Left: latency against the budget. Right: accuracy, for contrast."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    budget = rep["budget_ms"]
    archs = ["transformer", "gru", "tcn"]
    by = {(r["arch"], r["device"]): r for r in rep["archs"]}

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8),
                                   gridspec_kw={"width_ratios": [1.5, 1]})

    # ---- left: the gate --------------------------------------------------
    x = np.arange(len(archs))
    w = 0.36
    for off, dev, hatch in ((-w / 2, "cpu", ""), (w / 2, "cuda", "//")):
        vals, cols = [], []
        for a in archs:
            r = by.get((a, dev))
            v = (r["p99_ms"] + overhead) if r else 0.0
            vals.append(v)
            cols.append("#2a9d5c" if v <= budget else "#c0392b")
        bars = ax1.bar(x + off, vals, w, color=cols, hatch=hatch,
                       edgecolor="white", linewidth=0.6)
        for b, v in zip(bars, vals, strict=True):
            ok = v <= budget
            ax1.annotate(f"{v:.2f} ms\n{'PASS' if ok else 'FAIL'}",
                         (b.get_x() + b.get_width() / 2, v), ha="center",
                         va="bottom", fontsize=9,
                         color="#2a9d5c" if ok else "#c0392b",
                         fontweight="bold")

    ax1.axhline(budget, color="0.15", lw=2.6, ls="--", zorder=4)
    ax1.set_yscale("log")
    ax1.set_ylim(top=max(v for a in archs for d in ("cpu", "cuda")
                         if (r := by.get((a, d))) for v in [r["p99_ms"]]) * 3.2)
    ax1.set_xticks(x)
    ax1.set_xticklabels([DECISION_LABEL[a] for a in archs], fontsize=11)
    ax1.set_ylabel("end-to-end p99 decision latency (ms, log scale)\n"
                   f"forward pass + {overhead:.1f} ms belief update, scoring, argmax")
    # The budget lives in the title. Every placement inside the axes covered
    # some bar's PASS/FAIL label, and those labels are the panel's content.
    ax1.set_title("Feasibility is decided first\n"
                  f"dashed line = {budget:.2f} ms per-dwell budget, a HARD GATE",
                  fontsize=12, fontweight="bold")
    # Bars are coloured green/red by pass/fail, so a colour-keyed device
    # legend would claim a meaning the colours do not carry. Key the devices
    # on hatch, in neutral grey, and let colour mean only the verdict.
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(facecolor="0.75", edgecolor="white", label="CPU"),
                        Patch(facecolor="0.75", edgecolor="white", hatch="//",
                              label="GPU"),
                        Patch(facecolor="#2a9d5c", label="within budget"),
                        Patch(facecolor="#c0392b", label="over budget")],
               fontsize=9, loc="upper left", ncol=2)
    ax1.grid(alpha=0.25, axis="y", which="both")

    # ---- right: accuracy, for contrast -----------------------------------
    aps = [AP_200[a] for a in archs]
    # Green only where the architecture is deployable on BOTH devices; the
    # point of the panel is that the highest bar is not the one that ships.
    cols = []
    for a in archs:
        cpu = by.get((a, "cpu"))
        gpu = by.get((a, "cuda"))
        both = (cpu and cpu["p99_ms"] + overhead <= budget
                and gpu and gpu["p99_ms"] + overhead <= budget)
        cols.append("#2a9d5c" if both else "#c0392b")
    bars = ax2.bar(x, aps, 0.55, color=cols)
    for b, a, v in zip(bars, archs, aps, strict=True):
        note = {"transformer": "SHIPS", "gru": "excluded\nby timing",
                "tcn": "GPU-only"}[a]
        ax2.annotate(f"{v:.4f}\n{note}", (b.get_x() + b.get_width() / 2, v),
                     ha="center", va="bottom", fontsize=9.5, color="0.2")
    ax2.set_xticks(x)
    ax2.set_xticklabels([DECISION_LABEL[a] for a in archs], fontsize=11)
    ax2.set_ylim(0, max(aps) * 1.35)
    ax2.set_ylabel("average precision at 200 episodes\n(shared held-out set)")
    ax2.set_title("Accuracy does not decide", fontsize=12, fontweight="bold")
    ax2.grid(alpha=0.25, axis="y")

    fig.suptitle("The most accurate predictor is not the one that ships — "
                 "the GRU leads on AP and misses the dwell budget on both devices\n"
                 "TCN passes on GPU only; latency measured on this hardware, so the "
                 "ratio between architectures transfers, not the absolute ms",
                 fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(REPORTS / "arch_latency.json"))
    ap.add_argument("--overhead", type=float, default=0.6)
    ap.add_argument("--out", default=str(REPORTS / "latency_gate.png"))
    args = ap.parse_args()

    src = Path(args.json)
    if not src.is_file():
        raise SystemExit(f"no latency results at {src}; run scripts/arch_latency.py")
    plot(json.loads(src.read_text(encoding="utf-8")), args.overhead, Path(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
