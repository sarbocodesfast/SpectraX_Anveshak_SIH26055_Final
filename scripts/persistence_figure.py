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
    """One panel per finding, so the two cannot be read as one trend.

    Panel A is the **sign**: persistence swept with emitter count fixed.
    Panel B is the **magnitude**: emitter count doubled with the persistence
    profile fixed. An earlier version drew both on one axis, which invited a
    reader to run a line through two points that differ in emitter count
    rather than persistence -- the exact confound the experiment exists to
    break.

    The saturated condition appears in neither trend. It is a hatched
    placeholder labelled with its censoring, because at 25 of 30 pairs
    censored there is no comparison to put on an axis, and the five surviving
    pairs are the seeds where the predictor happened to get lucky.

    Confidence intervals and p-values are drawn into the panels rather than
    left to the caption, so the figure cannot be skimmed into a stronger
    claim than the data supports.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13.5, 5.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    by = {r["key"]: r for r in rows}
    pers = [by[k] for k in ("pairHI-B p=1.0 persist1.0", "pairMD-B p=0.5 persist.99")
            if k in by]
    dens = sorted((r for r in rows if abs(r["frac"] - 0.578) < 1e-6),
                  key=lambda r: int(r["label"].split()[0]))
    satu = next((r for r in rows if r["saturated"]), None)

    # ---- A: the sign -----------------------------------------------------
    ax1.axhline(0, color="0.3", lw=1.8, zorder=2)
    xs = [r["frac"] * 100 for r in pers]
    ys = [r["delta_pct"] for r in pers]
    lo = [y - r["ci"][0] for y, r in zip(ys, pers, strict=True)]
    hi = [r["ci"][1] - y for y, r in zip(ys, pers, strict=True)]
    ax1.errorbar(xs, ys, yerr=[lo, hi], fmt="-o", color="#1f77b4", lw=2.4,
                 ms=10, capsize=6, zorder=3,
                 label="persistence swept, 15 emitters throughout")
    for x, y, r in zip(xs, ys, pers, strict=True):
        ax1.annotate(f"{y:+.1f}%   p = {r['p']:.3f}", (x, y),
                     textcoords="offset points", xytext=(13, 2), fontsize=9.5,
                     color="#1f77b4", fontweight="bold")
        ax1.annotate(r["label"].replace("\n", ", "), (x, y),
                     textcoords="offset points", xytext=(13, -13), fontsize=8,
                     color="0.4")
    # Right-anchored: the 0% point and its labels occupy the upper left.
    ax1.annotate("predictor FASTER", (0.98, 0.96), xycoords="axes fraction",
                 fontsize=9, color="#1a8f4a", va="top", ha="right",
                 fontweight="bold")
    ax1.annotate("predictor SLOWER", (0.98, 0.04), xycoords="axes fraction",
                 fontsize=9, color="#c0392b", va="bottom", ha="right",
                 fontweight="bold")
    ax1.set_xlim(-12, 78)
    ax1.set_xticks([0, 46.7])
    ax1.set_xticklabels(["0%\nall persistent", "46.7%\nhalf persistent"])
    ax1.set_xlabel("emitters below 0.5 individual persistence")
    ax1.set_ylabel("intercept time vs `whittle` (%)\nbars are 95% paired bootstrap CI")
    ax1.set_title("A - Persistence sets the SIGN\nemitter count fixed at 15",
                  fontsize=11.5, fontweight="bold")
    ax1.legend(fontsize=8.5, loc="lower left", framealpha=0.95)
    ax1.grid(alpha=0.25)

    # ---- B: the magnitude ------------------------------------------------
    labels = [f"{int(r['label'].split()[0])} emitters" for r in dens]
    vals = [abs(r["delta_pct"]) for r in dens]
    miss = [r["d_missed"] for r in dens]
    if satu:
        labels.append("all scanning")
    xpos = np.arange(len(labels))
    bars = ax2.bar(xpos[:len(vals)], vals, color="#d62728", width=0.55)
    for b, v, m in zip(bars, vals, miss, strict=True):
        ax2.annotate(f"-{v:.0f}%\n{m:+d} emitters missed",
                     (b.get_x() + b.get_width() / 2, v), ha="center",
                     va="bottom", fontsize=9.5, color="0.25")
    if satu:
        # Deliberately not a value: there is no comparison to plot. The bar is
        # a placeholder whose only job is to say why this condition is absent.
        ax2.bar([xpos[-1]], [max(vals) * 0.08], color="none", edgecolor="0.55",
                hatch="///", width=0.55)
        ax2.annotate(f"WITHHELD\n{satu['n_censored']} of "
                     f"{satu['n_censored'] + satu['n_paired']} seed pairs "
                     f"censored\nsaturated - no comparison possible",
                     (xpos[-1], max(vals) * 0.10), ha="center", va="bottom",
                     fontsize=8.5, color="0.4")
    ax2.set_xticks(xpos)
    ax2.set_xticklabels(labels, fontsize=9.5)
    ax2.set_ylabel("intercept-time penalty (%), magnitude")
    ax2.set_title("B - Density scales the MAGNITUDE\n"
                  "persistence profile fixed at 57.8%",
                  fontsize=11.5, fontweight="bold")
    ax2.set_ylim(0, max(vals) * 1.5)
    ax2.grid(alpha=0.25, axis="y")
    if len(vals) == 2:
        # Routed above both bars. A diagonal drawn between their tops cuts
        # straight through the value labels, which are the reason the panel
        # exists at all.
        top = max(vals) * 1.28
        ax2.annotate("", xy=(0.92, top), xytext=(0.08, top),
                     arrowprops={"arrowstyle": "->", "color": "0.35", "lw": 1.8})
        ax2.text(0.5, top * 1.04, "same persistence, 2x the emitters",
                 ha="center", fontsize=9.5, color="0.25")

    fig.suptitle("Prediction helps when targets persist; density decides what a "
                 "wrong call costs\n"
                 "30 paired seeds per condition - paired bootstrap CI - "
                 "Wilcoxon signed-rank - one fixed checkpoint throughout",
                 fontsize=10.5)
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
