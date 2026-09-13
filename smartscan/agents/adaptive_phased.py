"""Hand over when harvesting beats discovering, judged online.

``PhasedScheduler`` hands over after ``patience`` slots without a newly
detecting channel. Sweeping that parameter on HARD produced a clean phase
transition -- interception rate peaks at ``patience = 200``, about 12% of the
episode spent sweeping, and falls threefold as exploration grows to 96% -- but
it also showed the parameter is the wrong object. The optimum moves with the
policy and with emitter density, and a slot count cannot transfer between
them.

The same sweep showed what the trigger should actually compare. At the
optimum, **25 of roughly 52 channels are still discovered after the hand-over**:
the exploit policy is not a pure exploiter, it discovers too. So the question
is not "has discovery stopped?" -- it has not -- but "does a policy that
discovers *and* harvests now beat one that only discovers?"

That comparison is observable without a counterfactual, which is what makes it
implementable. Every dwell the sweep takes produces hits, and each hit is
either on a channel never seen before or on one already known:

* new channels per look estimates the **discovery** rate, and it decays as the
  band is covered;
* hits on known channels per look estimates the **harvest** available, and it
  grows as more emitters are known.

The sweep collects the second only incidentally; a value-directed policy would
target it. So the hand-over fires when

    harvest_weight * harvest_rate > discovery_rate + margin

Both sides are EWMAs of quantities read from ``Observation.hits`` alone. No
ground truth, and no estimate of what the exploit policy *would* have done.

``harvest_weight`` is still a constant, and that is deliberate rather than
hidden: it is the mission's exchange rate between finding a new emitter and
collecting from a known one. Unlike a slot count it carries meaning, and it
should transfer across tiers where a timer cannot -- which is the claim the
held-out evaluation has to test rather than assume.

The minimum exploration period is one sweep revisit period, ``(B/K)`` dwells.
It is not a tuned number: before the receiver has crossed the band once, the
discovery-rate estimate has not seen most of the channels and comparing it to
anything is meaningless.

The switch is one-way. Hysteresis and a return path are the obvious next step,
but a single hand-over keeps the experiment clean: any difference against
fixed ``patience`` is attributable to *when* the policy switched and nothing
else.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from smartscan.agents.phased import PhasedScheduler
from smartscan.config import Config

__all__ = ["AdaptivePhasedScheduler"]


class AdaptivePhasedScheduler(PhasedScheduler):
    """Phased hand-over driven by a marginal-value comparison, not a timer.

    Args:
        config: Resolved configuration.
        seed: Seed forwarded to both delegates.
        name: Optional display name.
    """

    key = "adaptive_phased"
    needs_periods = True

    def __init__(self, config: Config, seed: int = 0, name: str | None = None) -> None:
        super().__init__(config, seed, name)
        cfg = config.agents
        #: Worth of one harvest look relative to one discovery look.
        self.harvest_weight = float(getattr(cfg, "adaptive_harvest_weight", 1.0))
        #: Dead band on the comparison, to stop the trigger firing on noise.
        self.margin = float(getattr(cfg, "adaptive_margin", 0.0))
        #: EWMA horizon in dwells.
        self.ewma_span = int(getattr(cfg, "adaptive_ewma_span", 200))

        # One full revisit period: the receiver cannot have a meaningful
        # discovery rate before it has crossed the band once.
        k = max(int(config.receiver.ibw_channels), 1)
        self.min_explore_dwells = int(np.ceil(config.spectrum.n_channels / k))

        self._alpha = 2.0 / (self.ewma_span + 1.0)
        self._discovery = 0.0
        self._harvest = 0.0
        self._dwells = 0
        #: Rates at the moment of hand-over, for the report.
        self.discovery_at_switch = float("nan")
        self.harvest_at_switch = float("nan")

    def reset(self) -> None:
        """Clear the rate estimates along with the phase state."""
        super().reset()
        self._discovery = 0.0
        self._harvest = 0.0
        self._dwells = 0
        self.discovery_at_switch = float("nan")
        self.harvest_at_switch = float("nan")

    def observe(self, obs: Any) -> None:
        """Split this dwell's hits into discovery and harvest, then smooth."""
        # Read membership BEFORE the parent updates `_seen`, or every hit would
        # look like a discovery and the harvest rate would stay at zero.
        lo, _hi = obs.window
        hits = np.asarray(obs.hits, dtype=bool)
        new = harvest = 0
        for offset in np.flatnonzero(hits):
            channel = int(lo) + int(offset)
            if channel in self._seen:
                harvest += 1
            else:
                new += 1

        super().observe(obs)

        self._dwells += 1
        self._discovery += self._alpha * (new - self._discovery)
        self._harvest += self._alpha * (harvest - self._harvest)

    def _should_switch(self, t: int) -> bool:
        """Hand over once weighted harvest overtakes discovery."""
        if self._dwells < self.min_explore_dwells:
            return False
        if self.harvest_weight * self._harvest > self._discovery + self.margin:
            self.discovery_at_switch = self._discovery
            self.harvest_at_switch = self._harvest
            return True
        return False
