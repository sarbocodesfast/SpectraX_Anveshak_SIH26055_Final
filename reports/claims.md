# What this project claims, and what it does not

Frozen register. Every headline number in this repository should be traceable
to a row here, with its qualifier attached. If a claim is not on the first
list, it is not this project's claim — including claims that would be
flattering.

## Safe to claim

| claim | evidence | qualifier |
|---|---|---|
| The **transformer** is the production/deployable predictor under the current 3 ms decision budget | [arch_latency.md](arch_latency.md) | measured on this hardware; the *ratio* between architectures transfers, the absolute milliseconds do not |
| The **GRU is excluded by latency** despite higher high-volume AP | [arch_latency.md](arch_latency.md) | fails on CPU (60.5 ms) and GPU (3.2 ms) against a 3.00 ms budget, forward pass plus measured 0.6 ms overhead |
| **TCN is a viable GPU-specific alternative**, not a general replacement | [arch_control.md](arch_control.md), [arch_latency.md](arch_latency.md) | passes the gate on GPU only; 9.8 ms on CPU. Low-volume behaviour unmeasured |
| **Prediction value depends strongly on temporal persistence** | [persistence_figure.md](persistence_figure.md) | 30 paired seeds/condition, bootstrap CI, Wilcoxon; sign crosses +18.0% (p=0.023) → −30.2% (p=0.023) |
| **Density amplifies the cost of poor exploitation decisions** | [persistence_figure.md](persistence_figure.md) | at an identical persistence profile: −297.5% → −1660.2%, missed +97 → +178 |
| **Architecture and training volume interact strongly** | [volume_control.md](volume_control.md) | **in the tested one-seed experiment**; I = −0.4334 AP, descriptive only |
| The previous **corpus-improvement result was confounded** by architecture/training regime | [evidence_status.md](evidence_status.md), [checkpoints.md](checkpoints.md) | mechanism identified: no `--arch`, unrecorded batch ladder, `--steps` and teacher-epoch overrides |
| `P(detection \| revisit gap)` is **flat** | [staleness_value.md](staleness_value.md) | 8.26 M dwells, `random` policy alone; pooling policies gives a Simpson's-paradox artefact |
| Uniform i.i.d. emitter channels make **coverage optimal for first contact** | analytic | holds before first contact and nowhere else |

## Do not claim

| not claimed | why not |
|---|---|
| The transformer is the **intrinsically most accurate** architecture | It loses to both GRU and TCN at 200 episodes on the common held-out set. It ships on feasibility, not accuracy |
| The **GRU is intrinsically inferior** | It has the highest AP of any architecture measured (0.5714 at 200 episodes). It is excluded on timing alone |
| **Full-corpus training intrinsically improves prediction** | Never demonstrated. The result that suggested it was measured on the volume-sensitive architecture at one corner of an interaction |
| The architecture × volume interaction is **statistically significant** | One seed per cell. Direction and effect size only. No paired test is possible |
| The observed **persistence thresholds generalise** beyond the tested simulation conditions | Measured on MEDIUM-class scenarios in this simulator. The mechanism should transfer; the numeric thresholds are not claimed to |
| Any architecture comparison is **recipe-independent** | All arms share one learning rate and schedule. Transformers are the most schedule-sensitive of the three, and this one's validation AP is 5× worse than the others' while its final score is only 9% worse — it may be undertrained under the shared recipe |
| The 400-episode condition **failed to show an effect** | It was never measured. Infeasible under the training memory budget (~21 GB), recorded as a feasibility limit rather than a result |

## The one-line summary

A feasibility-constrained closed-loop receiver scheduler that learns online,
and shows experimentally that the value and risk of prediction-led
exploitation depend on temporal persistence — while training volume and
predictor architecture can interact strongly enough that naive
model-comparison conclusions become misleading.

Not "we built an ML scheduler". The contribution is the conditions under which
prediction helps, and the discipline that established them.

## Why this file exists

Four confounds reached reported numbers in this project before being caught,
each by an anomaly rather than by a check: density versus emitter composition,
checkpoint identity, architecture versus corpus, and batch size. Every one
produced a claim that looked reasonable. A register of what is and is not
claimed is cheaper than rediscovering that a headline number carried an
unstated condition.
