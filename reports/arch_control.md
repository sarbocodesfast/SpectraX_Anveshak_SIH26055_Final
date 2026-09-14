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

## Shipping recommendation: RETRACTED — keep the transformer

The first version of this section recommended shipping the GRU for MEDIUM,
conditional on measuring inference latency. That measurement has been taken
and it **overturns the recommendation**. See
[arch_latency.md](arch_latency.md).

| device | transformer p99 | gru p99 | budget | gru share of budget |
|---|---|---|---|---|
| cpu | 1.381 ms | 58.475 ms | 3.00 ms | **1949%** |
| cuda | 0.838 ms | 3.007 ms | 3.00 ms | **100%** |

The GRU's forward pass **alone** exceeds the entire per-dwell budget on CPU by
a factor of nineteen, and exactly consumes it on GPU — leaving nothing for the
belief update and the argmax that must also happen inside the dwell. The whole
decision already measures p99 2.031 ms with the transformer
(`latency_budget.json`), so the overhead outside the forward pass is around
0.6 ms; adding that to the GRU's 3.007 ms puts it over budget on GPU too.

The cause is structural, not a tuning problem. A GRU is recurrent and must
step through all 128 window slots in sequence; a transformer attends over the
window in one shot. The GRU is the **smaller** model — 232,705 parameters
against 323,905 — and is still 42x slower per forward pass, because parameter
count is not what a real-time budget constrains.

**Ship the transformer.** A 9% average-precision advantage is worth nothing
from a model that cannot answer within the dwell.

This also resolves the config/checkpoint mismatch in the opposite direction
from the one the architecture result suggested: `configs/base.yaml` should
declare **`transformer`**, matching the shipped EASY and MEDIUM weights. The
GRU default is what a retraining run silently picked up, and it was never a
deployable choice.

### The lead worth following

`tcn` is the fastest architecture measured on GPU and the cheapest by far in
parameters — 95,553, under a third of the transformer — at p99 0.849 ms
against the transformer's 0.838 ms, and it was never entered in the accuracy
control. If it predicts anywhere near the GRU it would dominate both: the
dilated convolutions are parallel over time, so it has none of the GRU's
sequential penalty. Running the same two-arm control with `tcn` against
`transformer` is the obvious next experiment, and it is cheap.

Note that `tcn` is 294% over budget on **CPU**, so this lead only matters if
the receiver has a GPU. That is a hardware question, not a model one.

See [`evidence_status.md`](evidence_status.md) for what else is proved and
what remains quarantined, and [`checkpoints.md`](checkpoints.md) for what
each checkpoint file actually contains.
