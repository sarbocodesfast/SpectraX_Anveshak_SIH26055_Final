#!/usr/bin/env python
"""When is prediction-led exploitation safe? Two controlled interventions.

Across the three tiers, a sharper predictor sometimes helped the scheduler and
sometimes destroyed it, and the tiers differ in five ways at once -- emitter
count, emitter class mix, interferers, pop-ups and the decoy penalty -- so no
tier comparison could say which mattered. The obvious reading, that emitter
density drives it, turned out to be a confound.

Two interventions separate the candidates, 30 paired seeds each:

**Persistence intervention.** Fifteen emitters throughout, live-channel count
fixed, composition swung from all-persistent to all-scanning. Detectable-cell
duty moves with it, but the count does not.

**Density intervention.** Emitter count 15 -> 30 with the mix held
proportional, which leaves the persistence profile identical at 57.8% of
emitters below 0.5 individual persistence.

What separates the regimes is **persistence**, measured per emitter rather
than pooled. Pooled persistence is useless for a mixture: at half persistent
emitters it reads 0.99, because the always-on emitters contribute nearly every
detectable cell while the scanners hiding behind them are exactly the ones
being missed.

The result splits into two claims that the interventions can support
separately:

* **Persistence sets the sign.** Between 0% and 46.7% low-persistence
  emitters, the predictor's intercept-time advantage over ``whittle`` crosses
  from +18.0% to -30.2%, with density untouched.
* **Density scales the magnitude.** At an identical 57.8% persistence profile,
  going from 15 to 30 emitters leaves the sign alone and takes the penalty
  from -297.5% to -1660.2%, and the excess emitters missed from 97 to 178.

The all-scanning anchor is **not** a third point on that trend. Its detectable
duty is 0.0013 and ``whittle``'s median time to first intercept is infinite --
over half the emitters are never intercepted by anything -- so both policies
are unreliable and no comparison there is significant. It is plotted
separately as a saturation regime, because folding a degenerate ceiling into a
trend line would hide the very distinction that makes the finding useful.
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

#: condition key -> (fraction of emitters below 0.5 persistence, label, saturated)
#: The fractions come from scripts' per-emitter persistence measurement, not
#: from the pooled statistic.
CONDITIONS = {
    "pairHI-B p=1.0 persist1.0": (0.000, "15 emitters\nall persistent", False),
    "pairMD-B p=0.5 persist.99": (0.467, "15 emitters\nhalf persistent", False),
    "pairMD-A n=15 persist.81": (0.578, "15 emitters\nmixed", False),
    "pairHI-A n=30 persist.78": (0.578, "30 emitters\nmixed", False),
    "anchor   p=0.0 persist.00": (1.000, "15 emitters\nall scanning", True),
}


def deltas(d: dict) -> list[dict]:
    """Predictor minus whittle, per condition."""
    out = []
    for key, (frac, label, sat) in CONDITIONS.items():
        if key not in d:
            continue
        p = d[key]["policies"]
        x = np.asarray(p["predictor"]["ttfi"], float)
        y = np.asarray(p["whittle"]["ttfi"], float)
        ok = np.isfinite(x) & np.isfinite(y)
        # Positive = predictor is FASTER. Infinite medians occur in the
        # saturated regime, where the difference is not meaningful anyway.
        dt = ((np.median(y[ok]) - np.median(x[ok])) / abs(np.median(y[ok])) * 100
              if ok.any() else float("nan"))
        out.append({
            "key": key, "frac": frac, "label": label, "saturated": sat,
            "d_ttfi": float(dt),
            "d_missed": int(sum(p["predictor"]["never"]) - sum(p["whittle"]["never"])),
            "duty": float(d[key]["duty"]),
        })
    return sorted(out, key=lambda r: (r["frac"], r["d_ttfi"]))


def plot(rows: list[dict], out: Path) -> None:
    """Two panels, one per intervention.

    The persistence series and the density series are drawn as separate
    objects on purpose. Joining them would put a line through two points that
    differ in emitter count rather than persistence, which is exactly the
    confound the experiment was built to break.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.4))
    by = {r["key"]: r for r in rows}
    # Intervention 1: 15 emitters throughout, composition swung.
    pers = [by[k] for k in ("pairHI-B p=1.0 persist1.0", "pairMD-B p=0.5 persist.99")
            if k in by]
    satu = [r for r in rows if r["saturated"]]
    # Intervention 2: persistence profile identical, count doubled.
    dens = sorted((r for r in rows if abs(r["frac"] - 0.578) < 1e-6),
                  key=lambda r: r["label"])

    ax1.axhline(0, color="0.55", lw=1, zorder=1)
    ax1.plot([r["frac"] * 100 for r in pers], [r["d_ttfi"] for r in pers],
             "-o", color="#1f77b4", lw=2, zorder=3,
             label="persistence swept\n(15 emitters throughout)")
    if len(dens) == 2:
        ax1.plot([r["frac"] * 100 for r in dens], [r["d_ttfi"] for r in dens],
                 "s--", color="#d62728", lw=1.5, ms=7, zorder=3,
                 label="density swept\n(same persistence profile)")
    for r in satu:
        ax1.plot(r["frac"] * 100, r["d_ttfi"], "x", ms=13, mew=3,
                 color="#7f7f7f", zorder=3,
                 label="saturated: no significant\ncomparison possible")
    for r in rows:
        dx, dy = (-72, 6) if r["saturated"] else (9, -3)
        ax1.annotate(r["label"], (r["frac"] * 100, r["d_ttfi"]),
                     textcoords="offset points", xytext=(dx, dy), fontsize=8,
                     color="0.3")
    ax1.set_xlim(-8, 116)
    ax1.set_xlabel("emitters below 0.5 individual persistence (%)")
    ax1.set_ylabel("predictor vs whittle, time to first intercept (%)\n"
                   "positive = predictor faster")
    ax1.set_title("Persistence sets the sign")
    ax1.set_yscale("symlog", linthresh=50)
    ax1.legend(fontsize=7.5, loc="lower left")
    ax1.grid(alpha=0.25)

    # Panel B ordered by persistence then emitter count, so the density pair
    # sits adjacent and the comparison is visible without reading labels.
    order = sorted(rows, key=lambda r: (r["frac"], int(r["label"].split()[0])))
    xs = np.arange(len(order))
    cols = ["#7f7f7f" if r["saturated"] else "#d62728" for r in order]
    bars = ax2.bar(xs, [r["d_missed"] for r in order], color=cols)
    for b, r in zip(bars, order, strict=True):
        ax2.annotate(f"{r['d_missed']:+d}", (b.get_x() + b.get_width() / 2,
                                             r["d_missed"]),
                     ha="center", va="bottom", fontsize=9, color="0.25")
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f"{r['frac'] * 100:.0f}%\n{r['label'].splitlines()[0]}"
                         for r in order], fontsize=8)
    ax2.set_ylabel("extra emitters never intercepted\n(predictor minus whittle)")
    ax2.set_title("Density scales the cost")
    ax2.grid(alpha=0.25, axis="y")
    same = [i for i, r in enumerate(order) if abs(r["frac"] - 0.578) < 1e-6]
    if len(same) == 2:
        lo, hi = same
        top = max(order[lo]["d_missed"], order[hi]["d_missed"])
        ax2.annotate("", xy=(hi, order[hi]["d_missed"] * 1.02),
                     xytext=(lo, order[lo]["d_missed"] * 1.02),
                     arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.5})
        ax2.text((lo + hi) / 2, top * 1.12, "same persistence,\n2x the emitters",
                 ha="center", fontsize=8, color="0.25")
    ax2.set_ylim(0, max(r["d_missed"] for r in order) * 1.32)

    fig.suptitle("Prediction helps when targets persist; density decides what a "
                 "wrong call costs  (30 paired seeds per condition)", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def render(rows: list[dict]) -> str:
    L: list[str] = []
    A = L.append
    A("# When is prediction-led exploitation safe?")
    A("")
    A("_Generated by `scripts/persistence_figure.py`, 30 paired seeds per condition._")
    A("")
    A("![persistence](persistence_figure.png)")
    A("")
    A("| emitters below 0.5 persistence | scenario | det. duty | Δ intercept time | "
      "Δ emitters missed |")
    A("|---|---|---|---|---|")
    for r in rows:
        note = " _(saturated)_" if r["saturated"] else ""
        A(f"| {r['frac'] * 100:.1f}% | {r['label'].replace(chr(10), ', ')}{note} | "
          f"{r['duty']:.4f} | {r['d_ttfi']:+.1f}% | {r['d_missed']:+d} |")
    A("")
    A("Δ is `predictor` against `whittle`; positive intercept time means the predictor")
    A("is faster, positive missed means it loses more emitters.")
    A("")
    A("**Persistence sets the sign.** Holding the emitter count at 15 and the")
    A("live-channel count fixed, swinging composition from all-persistent to half")
    A("takes the predictor's intercept-time advantage from **+18.0% to −30.2%**. No")
    A("change in density is involved.")
    A("")
    A("**Density scales the magnitude.** The two 57.8% rows have *identical*")
    A("persistence profiles and differ only in emitter count. The sign does not move;")
    A("the penalty goes from **−297.5% to −1660.2%** and the excess emitters missed")
    A("from **97 to 178**.")
    A("")
    A("**The all-scanning anchor is a different regime, not a third trend point.**")
    A("Its detectable duty is 0.0013 and `whittle`'s median time to first intercept is")
    A("infinite -- more than half the emitters are never intercepted by any policy --")
    A("so neither comparison there reaches significance. Folding it into the trend")
    A("would disguise a degenerate ceiling as a continuation of the effect.")
    A("")
    A("**Why per-emitter persistence and not the pooled statistic.** At half")
    A("persistent emitters the pooled figure reads 0.99, because the always-on")
    A("emitters contribute nearly every detectable cell while the scanners hiding")
    A("behind them are precisely the ones being missed. Any online estimator faces the")
    A("same trap: it has to estimate the persistence *distribution*, not its average.")
    A("")
    return "\n".join(L)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", default=str(REPORTS / "persistence_matrix.json"))
    ap.add_argument("--out", default=str(REPORTS / "persistence_figure"))
    args = ap.parse_args()

    src = Path(args.json)
    if not src.is_file():
        raise SystemExit(f"no matrix at {src}")
    rows = deltas(json.loads(src.read_text(encoding="utf-8")))
    stem = Path(args.out)
    plot(rows, stem.with_suffix(".png"))
    stem.with_suffix(".md").write_text(render(rows), encoding="utf-8")
    print(render(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
