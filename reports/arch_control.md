# Architecture, isolated: the GRU wins by ~9%

_Two independent runs on Deepnote (L4). Student = the deployed,
observation-only model; the teacher sees privileged state and is not a
deployable candidate._

## Result

| run | transformer AP | gru AP | Δ AP | Δ AUC | gru training cost |
|---|---|---|---|---|---|
| 1 (6 student + 3 teacher epochs) | 0.5132 | **0.5591** | **+8.9%** | +3.9% | 7.9× |
| 2 (3 student + 2 teacher epochs) | 0.5103 | **0.5573** | **+9.2%** | +3.5% | 5.7× |

On the internal validation metric the gap is far wider: best val AP **0.322
for the GRU against 0.063 for the transformer**, over 5×.

## What varied, and what was frozen

| | |
|---|---|
| **varied** | architecture (`transformer` vs `gru`) |
| **frozen** | 200 episodes, 200 windows/episode, seed 0, batch 32, epochs, teacher epochs, tier (MEDIUM), torch 2.5.1+cu121, one machine, one run |

Both arms take the same `--episodes` path, so window construction, the 80/20
split and the held-out set are identical. The evaluation base rate is
**0.085967 in all four arms, to six digits** — that is the check that the two
architectures were scored against one set rather than two. The corpus
comparison this replaced failed exactly there: its arms validated on
different held-out sets, so their average precisions were never one
measurement.

Both runs record `torch: 2.5.1+cu121`, so the replication does not cross a
torch build. An earlier attempt silently moved from 2.5.1+cu121 to
2.14.0+cu130 because a pip index resolved differently, which is why the
version is now a manifest field rather than an assumption.

## What this settles

**The quarantined "full corpus improved prediction" claim was substantially
an architecture effect.** The retraining run passed no `--arch`, so it took
the config default and switched transformer → GRU while also changing the
corpus. The improvement was real; the attribution was wrong. A ~9% AP gap
from architecture alone accounts for a large part of what was credited to the
corpus.

**The shipped EASY and MEDIUM checkpoints are the weaker architecture.** They
are transformers, while `configs/base.yaml` declares `gru`. The config was
right and the checkpoints were wrong — the opposite of what the filenames
implied.

**Halving the epoch budget did not change the answer.** Run 2 used 3 student
and 2 teacher epochs against run 1's 6 and 3, and moved AP by 0.003. The cut
was made because the transformer's best epoch was 2 of 6 and validation AP
fell monotonically after it (0.0693, 0.0605, 0.0592, 0.0573, 0.0572), so the
later epochs bought overfitting and wall time.

## What it does not settle

**One seed, one tier.** Both runs use seed 0 on MEDIUM, so the ±9% has no
confidence interval. Two runs agreeing to 0.3 points is consistency under a
changed epoch budget, not a paired test across seeds. A seed sweep would be
needed before quoting this as ±9% ± anything.

**Training cost is not inference cost.** The GRU costs 5.7–7.9× the
transformer to train, and the receiver's constraint is the 1 ms dwell budget,
not training time. Nothing here measures per-dwell latency, and that is the
number a deployment decision needs. The GRU is also memory-hungry: it asked
for 4.79 GiB on a 6 GB card at batch 32 and OOM'd, and 6.52 GiB at batch 64.

**Prediction quality is not mission performance.** This measures average
precision on next-slot occupancy. The project's own evidence is that a
sharper predictor can *lose* emitters by parking — on HARD, `predictor` posts
the second-best interception ratio and the worst hard-target record, missing
126 of 146. A better AP is a reason to prefer an architecture, not a reason
to expect a better scheduler.

## Shipping recommendation

**Ship the GRU for MEDIUM, and make `configs/base.yaml` the authority.**

The evidence licenses this for MEDIUM specifically: same data, same split,
same recipe, replicated. It also removes the standing inconsistency where the
declared default and the shipped weights disagree.

**Do not extend it to EASY or HARD without running the same control there.**
Each tier is a different regime, and this project has already been caught
generalising across tiers that differ in five ways at once.

**Before shipping, measure per-dwell inference latency for both
architectures.** If the GRU cannot meet the dwell budget on the target
hardware, a 9% AP advantage is irrelevant and the transformer stays. That
measurement does not exist yet and it is the one that can overturn this
recommendation.

See [`evidence_status.md`](evidence_status.md) for what else is proved and
what remains quarantined, and [`checkpoints.md`](checkpoints.md) for what
each checkpoint file actually contains.
