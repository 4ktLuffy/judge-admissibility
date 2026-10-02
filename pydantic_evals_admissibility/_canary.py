"""Keep checking a judge after it is certified, on the traffic it is grading.

A judge certified once can drift: its model is upgraded, its rubric edited, its provider changes
a default. Online evaluation then keeps producing scores, and nothing says they stopped meaning
anything. `JudgeCanary` wraps a judge and, on a sampled share of calls, also asks it about a
control that must fail: an empty answer, and, if `borrow=True`, another request's answer.

Borrowing is off by default because live traffic has no known answers, so a borrowed answer is
only wrong if answers are specific to their question. Measured on this package's task: an answer
borrowed from another question of the same kind was in fact correct for 20% of weekday questions
and 27% of letter counts, so a sound judge rightly passed it and looked unhealthy. Turn it on
for open-ended outputs, where two requests rarely share a right answer.

Every verdict on real outputs passes through unchanged; the canary's result is added as
`judge_canary_rejected`, so online it lands in Logfire next to the scores, and `CanaryMonitor`
turns the recent checks into HEALTHY, DRIFTING or UNKNOWN. A control call that raises is counted
as an error, not a check, and never costs the real verdict.

Because it is an ordinary `Evaluator`, it works with `Dataset.evaluate` and with
`pydantic_evals.online.evaluate` alike.
"""

from __future__ import annotations

import random
import threading
from collections import deque
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from ._certify import assertion_of
from ._stats import wilson

Health = Literal['HEALTHY', 'DRIFTING', 'UNKNOWN']


@dataclass
class CanaryMonitor:
    """Counts of the canary checks, and what the recent ones say about the judge."""

    min_rejection: float = 0.8
    min_checks: int = 20
    rejected: int = 0
    """Lifetime count of controls the judge rejected; `health` does not use it."""
    checks: int = 0
    """Lifetime count of checks; `health` does not use it."""
    keep: int = 100
    """How many recent checks to keep in `history`: the window `health` is judged on.

    A lifetime rate would let a long healthy past outvote a judge that just broke: after 10,000
    rejected controls, 100 passed ones barely move it. Must be at least `min_checks`.
    """
    errors: int = 0
    """Control calls that raised; they are not checks, since the judge gave no verdict."""
    last_error: BaseException | None = field(default=None, repr=False)
    history: deque[dict[str, Any]] = field(default_factory=deque, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def __post_init__(self) -> None:
        if self.keep < self.min_checks:
            raise ValueError(
                f'keep ({self.keep}) must be at least min_checks ({self.min_checks}), or health stays UNKNOWN'
            )

    def record_error(self, error: BaseException) -> None:
        with self._lock:
            self.errors += 1
            self.last_error = error

    def record(self, rejected: bool, **detail: Any) -> None:
        with self._lock:
            self.checks += 1
            self.rejected += rejected
            self.history.append({'rejected': rejected, **detail})
            while len(self.history) > self.keep:
                self.history.popleft()

    def health(self) -> Health:
        """HEALTHY when the recent rejection rate's lower bound clears `min_rejection`, DRIFTING when
        its upper bound is below it, UNKNOWN until there are enough checks to say either.

        Recent means the last `keep` checks, those in `history`.
        """
        with self._lock:
            checks = len(self.history)
            rejected = sum(bool(h['rejected']) for h in self.history)
        low, high = wilson(rejected, checks)
        if checks >= self.min_checks and low >= self.min_rejection:
            return 'HEALTHY'
        if checks >= self.min_checks and high < self.min_rejection:
            return 'DRIFTING'
        return 'UNKNOWN'


@dataclass
class JudgeCanary(Evaluator[Any, Any, Any]):
    """Pass `judge`'s verdicts through, and on `rate` of calls also check it on a control."""

    judge: Evaluator[Any, Any, Any]
    rate: float = 0.05
    assertion: str | None = None
    monitor: CanaryMonitor = field(default_factory=CanaryMonitor)
    every: int | None = None
    """Check exactly every `every`-th call instead of sampling at `rate`.

    Random sampling can under-check by luck: at rate 0.25 a fixed seed in this package gave 2
    checks in 40 calls (a 0.1% draw). A monitor that must see a known number of checks per window
    should sample systematically.
    """
    borrow: bool = False
    """Also use another request's answer as a control. Only valid when answers are question-specific."""
    seed: int | None = None
    recent: int = 50
    _rng: random.Random = field(init=False, repr=False)
    _seen: deque[tuple[Any, Any]] = field(init=False, repr=False)
    _calls: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.seed)
        self._seen = deque(maxlen=self.recent)

    def get_default_evaluation_name(self) -> str:
        return self.judge.get_default_evaluation_name()

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> Any:
        verdict = await self.judge.evaluate_async(ctx)
        self._calls += 1
        due = self._calls % self.every == 0 if self.every else self._rng.random() < self.rate
        control = self._control(ctx) if due else None
        self._seen.append((repr(ctx.inputs), ctx.output))
        if control is None:
            return verdict
        kind, output = control
        try:
            passed, _ = assertion_of(await self.judge.evaluate_async(replace(ctx, output=output)), self.assertion)
        except Exception as error:  # the control is ours; its failure must not cost the caller the real verdict
            self.monitor.record_error(error)
            return verdict
        self.monitor.record(not passed, control=kind, inputs=ctx.inputs, control_output=output, live_output=ctx.output)
        canary = EvaluationReason(value=not passed, reason=f'{kind} control; judge health {self.monitor.health()}')
        if isinstance(verdict, dict):
            return {**verdict, 'judge_canary_rejected': canary}
        return {self.get_default_evaluation_name(): verdict, 'judge_canary_rejected': canary}

    def _control(self, ctx: EvaluatorContext[Any, Any, Any]) -> tuple[str, Any] | None:
        """An empty answer; with `borrow`, half the time another request's answer instead."""
        donors = [out for inputs, out in self._seen if inputs != repr(ctx.inputs) and out != ctx.output]
        if self.borrow and donors and (self._rng.random() < 0.5 or not isinstance(ctx.output, str)):
            return 'mismatched_output', self._rng.choice(donors)
        if isinstance(ctx.output, str):
            return 'empty_output', ''
        return None
