"""Sweep while discovery is productive, then exploit.

Every scheduler in this repository runs one policy for the whole episode, and
the HARD results say that is the wrong shape.

The uniform-prior result is specific about *when* coverage wins. Emitter home
channels are drawn i.i.d. uniform over the band, so an emitter that has never
been intercepted has produced no observations and its channel posterior equals
that prior -- no amount of training can beat uniform coverage at finding it.
That argument holds **before first contact and nowhere else**. Once an emitter
has been seen, its channel is known exactly, its period is estimable, and
exploitation is strictly better informed than sweeping.

So the optimum is not a policy, it is a schedule over policies: sweep while
sweeping is still discovering, hand over when it stops. Measured on HARD, the
two halves of that already exist and each wins the objective the other loses --
``coprime_sweep`` takes intercept time +35.1% against the tuned baseline while
giving up 35.1% of the interception rate, and ``whittle`` takes the rate +80.9%
while its own intercept-time gain is not significant over 30 seeds. Neither
improves both, because each is running the wrong policy for half the episode.

**The switch is driven by what the receiver observes, not by a timer.** A fixed
fraction would have to be tuned per tier and would be wrong whenever emitter
density differed from the tuning run. Instead the policy tracks how many
distinct channels have ever produced a detection -- a quantity available from
``Observation.hits`` alone -- and hands over when that count has not grown for
``patience`` slots. When sweeping stops finding new channels it has stopped
earning its coverage guarantee, and that moment is observable.

Nothing here reads ground truth. ``truth_ids`` and ``pfa_flags`` exist on the
observation for evaluation and are deliberately untouched, exactly as
``BeliefState`` avoids them.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from smartscan.agents.base import Scheduler
from smartscan.agents.belief import BeliefState
from smartscan.config import Config

__all__ = ["PhasedScheduler"]


class PhasedScheduler(Scheduler):
    """Run a coverage policy until discovery stalls, then an exploit policy.

    Args:
        config: Resolved configuration.
        seed: Seed forwarded to both delegates.
        name: Optional display name.
        explore: Scheduler key for the coverage phase.
        exploit: Scheduler key for the exploitation phase.
    """

    key = "phased"
    needs_periods = True

    def __init__(
        self,
        config: Config,
        seed: int = 0,
        name: str | None = None,
        explore: str | None = None,
        exploit: str | None = None,
    ) -> None:
        super().__init__(config, seed, name)
        from smartscan.agents import build_agent

        cfg = config.agents
        self._explore_key = explore or getattr(cfg, "phased_explore", "coprime_sweep")
        self._exploit_key = exploit or getattr(cfg, "phased_exploit", "whittle")
        #: Slots without a newly-detecting channel before handing over.
        self.patience = int(getattr(cfg, "phased_patience", 1500))

        self._explore = build_agent(self._explore_key, config, seed)
        self._exploit = build_agent(self._exploit_key, config, seed)

        self._seen: set[int] = set()
        self._last_new = 0
        self._switched_at: int | None = None

        # Telemetry. A sweep over `patience` says which value scored best; it
        # does not say why, and the whole point of moving to an adaptive
        # trigger is knowing what observable should drive it. These record what
        # the hand-off actually cost and bought.
        #: Channels that had ever detected at the moment of hand-over.
        self.n_seen_at_switch = 0
        #: Retunes, as a proxy for the settle cost the policy is paying.
        self.n_retunes = 0

    def reset(self) -> None:
        """Reset both delegates and the hand-over state."""
        super().reset()
        for d in (getattr(self, "_explore", None), getattr(self, "_exploit", None)):
            if d is not None:
                d.reset()
        self._seen = set()
        self._last_new = 0
        self._switched_at = None
        self.n_seen_at_switch = 0
        self.n_retunes = 0

    def observe(self, obs: Any) -> None:
        """Track which channels have ever detected, and feed both delegates.

        Both delegates see every observation, including the ones produced while
        the other was driving. The exploit policy therefore inherits a warm
        belief at hand-over rather than starting blind, which is the whole
        point of sweeping first.
        """
        self._explore.observe(obs)
        self._exploit.observe(obs)

        lo, _hi = obs.window
        hits = np.asarray(obs.hits, dtype=bool)
        # Genuine detections only where the evaluator would count them is NOT
        # available here -- pfa_flags is evaluation-only -- so this counts
        # declared hits, which is what a real receiver sees. A false alarm can
        # therefore delay the hand-over, never bring it forward.
        for offset in np.flatnonzero(hits):
            channel = int(lo) + int(offset)
            if channel not in self._seen:
                self._seen.add(channel)
                self._last_new = int(obs.t)

    @property
    def exploring(self) -> bool:
        """Whether the coverage phase is still running."""
        return self._switched_at is None

    def act(self, belief: BeliefState, t: int) -> int:
        """Sweep until discovery stalls, then exploit."""
        if self._switched_at is None and t - self._last_new >= self.patience:
            self._switched_at = t
            self.n_seen_at_switch = len(self._seen)
        delegate = self._exploit if self._switched_at is not None else self._explore
        action = int(delegate.act(belief, t))
        if self.last_action is not None and action != self.last_action:
            self.n_retunes += 1
        self.last_action = action
        return action

    @property
    def switch_slot(self) -> int | None:
        """Slot at which the hand-over fired, or None if it never did."""
        return self._switched_at

    @property
    def n_discovered_after_switch(self) -> int:
        """Channels that first detected only after the hand-over.

        The quantity the trigger is implicitly betting against: if this stays
        high, exploration was abandoned too early.
        """
        return len(self._seen) - self.n_seen_at_switch if self._switched_at else 0
