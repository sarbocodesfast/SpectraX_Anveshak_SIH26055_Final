"""A hard coverage constraint that wraps any scheduler.

Every value-based policy here expresses coverage as a *score*::

    value = P(occupied) + coverage_weight * staleness

and that construction fails twice over.

It fails empirically. Measured on 8.26 M replayed dwells with an exogenous
instrument, ``P(detection | revisit gap)`` is flat -- 0.0343 on EASY, 0.0538 on
MEDIUM, 0.0750 on HARD, homogeneous across gap bins at p = 0.81, 0.55 and 0.31.
Staleness carries no information about whether the next look lands, so the term
is multiplying a quantity that predicts nothing.

It fails structurally. In an additive score the two terms compete on one
scalar, so whichever is larger wins *globally* and the other is switched off.
A sharper predictor widens ``P`` until staleness cannot compete and the policy
parks: on HARD, ``predictor`` reaches the best interception rate of any policy
while missing 39 emitters against the sweep's 26, and its intercept time
collapses by 1398%. Re-weighting only moves which term loses -- sweeping
``coverage_weight`` from 1 to 16 never recovered it.

So coverage is not priced here, it is **enforced**. This wrapper is a
lexicographic gate: if any channel is at its revisit deadline the receiver goes
there, and only if none is does the wrapped policy get its choice. The policy
keeps its whole score function and its full strength; what it loses is the
ability to abandon the band, which is the one behaviour the mission cannot
tolerate.

The deadline is a bound on the largest revisit gap, which is the quantity the
scan-on-scan analysis says matters: an emitter is intercepted when the receiver
is present during an illumination, and a bounded gap bounds how long an
illumination window can hide. It is not a bound on the *mean*, which a policy
can satisfy while starving a few channels indefinitely.

The gate selects by ``window_max``, not ``window_value``. Summing staleness
over a window dilutes one desperately overdue channel among fresh neighbours,
so the sum would send the receiver to a window of moderately stale channels
while the starved one stays unvisited -- exactly the failure the gate exists to
prevent.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import BeliefState
from smartscan.config import Config

__all__ = ["CoverageGate"]


class CoverageGate(Scheduler):
    """Veto a policy's choice when a channel is at its revisit deadline.

    Args:
        inner: The scheduler being wrapped. Keeps its own configuration.
        config: Resolved configuration.
        max_staleness: Deadline in slots. A channel unvisited for this long
            takes priority over whatever the policy wanted.
    """

    key = "coverage_gate"

    def __init__(self, inner: Scheduler, config: Config, max_staleness: int) -> None:
        super().__init__(config, getattr(inner, "seed", 0), f"gated:{inner.name}")
        self.inner = inner
        self.max_staleness = int(max_staleness)
        #: Dwells the gate took away from the policy. Reported so the cost of
        #: the constraint is visible rather than hidden inside the score.
        self.n_vetoes = 0
        # Inherit the wrapped policy's retune economics. The base Scheduler
        # does not define retune_penalty -- policies that charge for a retune
        # set it themselves -- so without this the gate raises AttributeError
        # on its first veto, and gating must not change what a retune costs.
        self.retune_penalty = float(getattr(inner, "retune_penalty", 0.0))
        # Inherit the delegate's period requirement: gating must not silently
        # stop the belief estimating periods the wrapped policy depends on.
        self.needs_periods = bool(getattr(inner, "needs_periods", False))

    def reset(self) -> None:
        """Reset the wrapped policy and the veto counter."""
        super().reset()
        self.inner.reset()
        self.n_vetoes = 0

    def observe(self, obs: Any) -> None:
        """Forward the observation to the wrapped policy."""
        self.inner.observe(obs)

    def act(self, belief: BeliefState, t: int) -> int:
        """Serve the deadline if one is due, otherwise defer to the policy."""
        stale = np.asarray(belief.time_since_visit, dtype=np.float64)
        if self.max_staleness > 0 and float(stale.max()) >= self.max_staleness:
            # window_max, not window_value: see the module docstring.
            action = self.argmax_legal(self.window_max(stale), self.retune_penalty)
            self.n_vetoes += 1
        else:
            action = int(self.inner.act(belief, t))
        self.last_action = action
        return action
