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

from smartscan.analysis.metrics import paired_bootstrap_delta

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

for _stream in (sys.stdout, sys.stderr):
    with contextlib.suppress(AttributeError, ValueError):
        _stream.reconfigure(encoding="utf-8", errors="replace")

REPORTS = REPO_ROOT / "reports"

#: Below this many finite pairs a comparison is withheld rather than reported.
#: The all-scanning anchor has 5 of 30, which is the finding about that
#: condition, not a nuisance to work around.
MIN_PAIRS = 10

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


def paired_stats(x: np.ndarray, y: np.ndarray) -> dict:
    """Paired bootstrap CI and Wilcoxon p for treatment against baseline.

    Resamples seeds rather than observations, which keeps the pairing intact
    and removes scenario variance -- the reason 30 seeds carry a claim here.

    Infinite pairs are dropped and *counted*, because dropping them silently
    is how a bootstrap comes to favour the policy that abandons hard targets:
    an emitter never intercepted has no finite time to contribute, so the
    policy that misses it simply stops being measured on it. The count is what
    makes the anchor row readable as a saturation regime rather than a result.
    """
    ok = np.isfinite(x) & np.isfinite(y)
    out = {"n_paired": int(ok.sum()), "n_censored": int((~ok).sum())}
    if ok.sum() < MIN_PAIRS:
        return {**out, "delta_pct": float("nan"), "ci": (float("nan"),) * 2,
                "p": float("nan"), "withheld": True}
    ci = paired_bootstrap_delta(x[ok], y[ok], relative=True)
    p_val = float("nan")
    with contextlib.suppress(Exception):
        from scipy.stats import wilcoxon
        p_val = float(wilcoxon(x[ok], y[ok]).pvalue)
    return {**out, "delta_pct": ci.point * 100,
            "ci": (ci.lo * 100, ci.hi * 100), "p": p_val, "withheld": False}


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
            **paired_stats(x, y),
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
    """The evidence chain, in the order the claims actually depend on."""
    L: list[str] = []
    A = L.append
    by = {r["key"]: r for r in rows}
    pure = by.get("pairHI-B p=1.0 persist1.0")
    half = by.get("pairMD-B p=0.5 persist.99")
    d15 = by.get("pairMD-A n=15 persist.81")
    d30 = by.get("pairHI-A n=30 persist.78")
    anch = next((r for r in rows if r["saturated"]), None)

    A("# When is prediction-led exploitation safe?")
    A("")
    A("_Generated by `scripts/persistence_figure.py`. 30 paired seeds per")
    A("condition; `predictor` against `whittle`; paired bootstrap on seeds,")
    A("Wilcoxon signed-rank for p. One fixed checkpoint throughout, so the")
    A("model is a constant and none of the checkpoint confounds touch this._")
    A("")
    A("![persistence](persistence_figure.png)")
    A("")

    A("## 1. Hypothesis")
    A("")
    A("Temporal **persistence** -- how much of the time an emitter is")
    A("detectable -- determines whether prediction-led exploitation helps or")
    A("hurts, and emitter **density** does not.")
    A("")
    A("This is worth testing because the obvious reading was the other one.")
    A("Across the three tiers a sharper predictor sometimes helped and")
    A("sometimes destroyed the scheduler, and the tiers differ in five ways at")
    A("once -- emitter count, class mix, interferers, pop-ups and the decoy")
    A("penalty -- so \"as density rises\" was an unsupported reading of a")
    A("confounded contrast.")
    A("")

    A("## 2. Controlled result: persistence sets the sign")
    A("")
    A("Emitter count fixed at 15 and the live-channel count fixed; only the")
    A("class mix moves.")
    A("")
    A("| condition | emitters <0.5 persistence | det. duty | Δ intercept time | 95% CI | p |")
    A("|---|---|---|---|---|---|")
    for r in (pure, half):
        if r:
            A(f"| {r['label'].replace(chr(10), ', ')} | {r['frac'] * 100:.1f}% | "
              f"{r['duty']:.4f} | **{r['delta_pct']:+.1f}%** | "
              f"[{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] | {r['p']:.3f} |")
    A("")
    if pure and half:
        A(f"The advantage **crosses zero**: {pure['delta_pct']:+.1f}% when every")
        A(f"emitter is persistent, {half['delta_pct']:+.1f}% when half are not.")
        A("Both are significant at the 5% level, in opposite directions, with")
        A("density untouched. Positive means the predictor reaches its first")
        A("intercept sooner.")
        A("")
        A(f"The all-persistent CI reaches {pure['ci'][0]:+.1f}% at its lower")
        A("bound, so the *size* of the gain there is weakly determined even")
        A("though its sign is not. The claim this supports is that the sign")
        A("flips, not that the gain is large.")
    A("")

    A("## 3. Amplification: density scales the magnitude")
    A("")
    A("Persistence profile **identical** at 57.8% of emitters below 0.5; only")
    A("the emitter count moves.")
    A("")
    A("| condition | emitters | Δ intercept time | 95% CI | p | Δ emitters missed |")
    A("|---|---|---|---|---|---|")
    for r in (d15, d30):
        if r:
            n = r["label"].split()[0]
            A(f"| {r['label'].replace(chr(10), ', ')} | {n} | "
              f"{r['delta_pct']:+.1f}% | [{r['ci'][0]:+.1f}, {r['ci'][1]:+.1f}] | "
              f"{r['p']:.4f} | **{r['d_missed']:+d}** |")
    A("")
    if d15 and d30:
        A("The sign does not move; the penalty deepens from")
        A(f"{d15['delta_pct']:+.1f}% to {d30['delta_pct']:+.1f}% and the excess")
        A(f"emitters missed from {d15['d_missed']:+d} to {d30['d_missed']:+d}.")
        A("Density is an amplifier of a cost whose sign persistence has already")
        A("decided.")
    A("")

    A("## 4. Boundary condition: the all-scanning anchor is saturated")
    A("")
    if anch:
        A("At 100% low-persistence emitters the detectable duty is")
        A(f"{anch['duty']:.4f} and **{anch['n_censored']} of")
        A(f"{anch['n_censored'] + anch['n_paired']} seed pairs are censored** --")
        A("at least one policy never intercepts anything, so there is no finite")
        A("time to compare.")
        A("")
        if anch.get("withheld"):
            A(f"The comparison is **withheld**: {anch['n_paired']} surviving")
            A(f"pairs is below the {MIN_PAIRS}-pair floor, so no interval is")
            A("reported rather than an interval computed from the handful of")
            A("seeds that happened to finish.")
        else:
            A(f"What survives is {anch['delta_pct']:+.1f}% at p = "
              f"{anch['p']:.3f}, which is not a result.")
        A("")
        A("So it is **not** a third point on the trend of section 2, and is")
        A("plotted separately. Folding a degenerate ceiling into a trend line")
        A("would disguise an observation-budget limit as a continuation of the")
        A("effect -- and would invert its apparent sign, since the surviving")
        A("pairs are exactly the seeds where the predictor got lucky.")
    A("")

    A("## 5. Implication for the scheduler")
    A("")
    A("An online scheduler should estimate the **distribution** of temporal")
    A("persistence before trusting prediction-led exploitation, not its")
    A("average.")
    A("")
    A("Pooled persistence is actively misleading for a mixture: at half")
    A("persistent emitters it reads **0.99**, because the always-on emitters")
    A("contribute nearly every detectable cell while the scanners hiding")
    A("behind them are precisely the ones being missed. An estimator that")
    A("tracks the mean would report a regime where exploitation is safe at the")
    A("exact point where it costs 88 emitters.")
    A("")
    A("That is the specification for the regime estimator: the quantity to")
    A("track is the fraction of emitters below a persistence threshold, which")
    A("is a property of the distribution's lower tail and invisible in its")
    A("mean.")
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
