# What this project has proved, and what it has not

Each claim below carries the table that decides whether it is evidence: what
varied, and what was held still. A claim whose "frozen" column is incomplete
is quarantined regardless of how large its effect looked.

---

## Proved

### 1. Uniform i.i.d. emitter channels make coverage optimal for first contact

Emitter home channels are drawn i.i.d. uniform over the band, so an emitter
that has never been intercepted has produced no observations and its channel
posterior equals the prior. No amount of training beats uniform coverage at
finding it.

This is an analytic result about the generative model, not a measurement, so
it has no confounds to control. It holds **before first contact and nowhere
else** -- once an emitter has been seen its channel is known exactly.

### 2. `P(detection | revisit gap)` is flat

| varied | frozen |
|---|---|
| revisit gap (binned) | policy (`random` alone), tier, receiver, emitter population |

8.26 M replayed dwells against an exogenous instrument: 0.0343 on EASY,
0.0538 on MEDIUM, 0.0750 on HARD, homogeneous across gap bins at p = 0.81,
0.55 and 0.31. The `coverage_weight x staleness` term therefore multiplies a
quantity that predicts nothing.

Pooling `sequential` with `random` produced a spurious p = 1.1e-12 on HARD,
driven by 810 of 644,544 looks in a bin a saw-tooth should never reach --
Simpson's paradox. The frozen-policy column is what makes this claim safe.

### 3. Persistence sets the sign of prediction's value; density scales it

| intervention | varied | frozen |
|---|---|---|
| persistence | emitter class mix | emitter count (15), live-channel count, seeds (30 paired) |
| density | emitter count (15 -> 30) | persistence profile (57.8% below 0.5), class proportions, seeds |

Between 0% and 46.7% low-persistence emitters the predictor's intercept-time
advantage over `whittle` crosses +18.0% -> -30.2% with density untouched. At
an identical persistence profile, doubling the emitter count leaves the sign
alone and moves the penalty -297.5% -> -1660.2%.

**This result is unaffected by the checkpoint confounds below**: it used one
fixed checkpoint throughout, so the model is a constant, not a variable. It
stays frozen and does not need rerunning.

### 4. The all-scanning anchor is a saturation regime, not a trend point

Detectable duty 0.0013; `whittle`'s median time to first intercept is
infinite because over half the emitters are never intercepted by anything.
No comparison there reaches significance, so it is plotted separately.

---

## Quarantined

### 5. "Training on the full corpus improved prediction"

Reported as easy +45.5% ttfi, medium +24.6% TWIR, hard +89.7% ttfi / +40
coverage.

| varied | frozen |
|---|---|
| training corpus | seeds, tier, evaluation config |
| **architecture** (transformer -> gru) | -- *not frozen* |
| **batch size** (64 -> ladder 32/16/8, rung unrecorded) | -- *not frozen* |
| **epochs** (`--steps 6`) | -- *not frozen* |
| **teacher epochs** (default -> 3) | -- *not frozen* |

Five factors moved together, so the effect is attributable to all five
jointly and to none of them separately. Reading
`notebooks/deepnote_train_cell.py` shows how: it passes no `--arch`, tries
`batch_size` in a `(32, 16, 8)` ladder without recording which rung
succeeded, and overrides `--steps` and `teacher_epochs` besides.

The batch rung being *unrecorded* is the sharpest part. Even knowing the
confound exists, the run cannot be reconstructed from its own output.

The mechanism is now known: `configs/base.yaml` declares `arch: gru` while
the shipped EASY and MEDIUM weights are transformers. `build_predictor`
honours the checkpoint's own architecture, so runtime stayed correct and the
disagreement never surfaced. A retraining run that omitted `--arch` took the
declared default and silently changed architecture.

Batch size belongs on that list because of a measurement, not a worry:
rerunning the EASY transformer arm at batch 8 instead of 64 moved student AP
**0.6915 -> 0.7457**, larger than several effects this project had credited
to the corpus.

**Status:** awaiting the corpus x architecture control at a frozen recipe.

### 6. The retracted medium claim

"adaptive wins on medium +52.6% TWIR" compared medians-of-medians. Paired
properly it is +28.3%, p = 0.160 -- not significant. Deleted rather than
quarantined.

---

## How these are now enforced

`assert_comparable(a, b, target_factor=...)` refuses any comparison whose
manifests differ in more than the named factor, and refuses two files with
identical bytes. Run against the current checkpoint set, **no comparison is
licensed** -- see [checkpoints.md](checkpoints.md). That is the correct
answer for a history in which no run recorded its recipe.
