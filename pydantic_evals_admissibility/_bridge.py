"""Tell whether a history of scores survives a change of judge.

When the agent and the judge change together, a moved score could be either. Scoring both agent
versions with both judges (a 2x2: baseline and candidate outputs, old and new judge) separates
them: the agent's gain under each judge, the judge's shift on the same outputs, and their
interaction, the difference between the two gains. If the interaction is shown to be small, the
new judge moves absolute rates but ranks versions as the old one did, and gains and release
decisions made under the old judge stay comparable with new ones. If it is not, they do not.

`comparable` is an equivalence statement, so it is decided on an interval, not on a test against
zero: PASS when the whole interval of the interaction lies within ±margin, FAIL when it lies
entirely outside on one side, UNVALIDATED otherwise. An interval that merely contains zero says
the cases cannot see an interaction, not that there is none.

The case is the unit. Each case's interaction is one number, from its pass rates in the four
cells; repeats of a case sharpen that number but are not new cases. The interval inverts the
same sign-flip test `decide` uses (see `_interval`).
"""

from __future__ import annotations

import math
import random
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_evals.reporting import EvaluationReport

from ._certify import Certificate
from ._gate import GateResult, GateRules, decide
from ._reports import outcomes

Comparability = Literal['PASS', 'FAIL', 'UNVALIDATED']
Outcomes = Mapping[str, Sequence[bool]]
VERSIONS = ('baseline', 'candidate')


@dataclass(frozen=True)
class Effect:
    """A mean over cases of a per-case difference, with its interval."""

    estimate: float
    interval: tuple[float, float]

    def __str__(self) -> str:
        low, high = self.interval
        return f'{self.estimate:+.3f} [{low:+.3f}, {high:+.3f}]'


@dataclass(frozen=True)
class JudgeBridge:
    rates: dict[tuple[str, str], float]
    """Mean per-case pass rate in each cell, keyed by (version, judge): ('baseline', 'old'), ..."""
    decision_old: GateResult
    """`decide(baseline, candidate)` on the old judge's outcomes; its `mean_gain` is the gain."""
    decision_new: GateResult
    interaction: Effect
    """Gain under the new judge minus gain under the old, per case, averaged."""
    shift_baseline: Effect
    """New judge minus old judge on the baseline's outputs: how far absolute rates moved."""
    shift_candidate: Effect
    margin: float
    verdict: Comparability
    cases: int
    per_case: dict[str, float] = field(default_factory=dict, repr=False)
    """Each case's interaction."""

    @property
    def comparable(self) -> bool | None:
        """True if the interaction is shown within ±margin, False if shown beyond it, None if neither."""
        return {'PASS': True, 'FAIL': False, 'UNVALIDATED': None}[self.verdict]

    @property
    def decisions_differ(self) -> bool:
        return self.decision_old.decision != self.decision_new.decision

    def table(self) -> str:
        r = self.rates
        gain_old, gain_new = self.decision_old.mean_gain, self.decision_new.mean_gain
        lines = [
            f'{"":<12}{"old judge":>12}{"new judge":>12}{"judge shift":>32}',
            f'{"baseline":<12}{r["baseline", "old"]:>12.3f}{r["baseline", "new"]:>12.3f}{self.shift_baseline!s:>32}',
            f'{"candidate":<12}{r["candidate", "old"]:>12.3f}{r["candidate", "new"]:>12.3f}'
            f'{self.shift_candidate!s:>32}',
            f'{"agent gain":<12}{gain_old if gain_old is not None else float("nan"):>+12.3f}'
            f'{gain_new if gain_new is not None else float("nan"):>+12.3f}',
            f'{"decision":<12}{self.decision_old.decision:>12}{self.decision_new.decision:>12}',
            f'interaction (new gain - old gain): {self.interaction}, margin ±{self.margin:.3f}, {self.cases} cases',
            f'comparable: {self.verdict}. {self._reason()}',
        ]
        return '\n'.join(lines)

    def _reason(self) -> str:
        if self.verdict == 'PASS':
            reason = 'the judge change moves both versions alike, within the margin'
        elif self.verdict == 'FAIL':
            reason = 'the new judge changes the gain itself; old and new gains are not comparable'
        else:
            reason = 'the cases cannot show the interaction is within the margin'
        if self.decisions_differ:
            reason += f'; the release decision differs ({self.decision_old.decision} -> {self.decision_new.decision})'
        return reason


def _rate(values: Sequence[bool]) -> float:
    return sum(values) / len(values)


def _interval(values: Sequence[float], *, bound: float, rules: GateRules) -> tuple[float, float]:
    """Interval of the mean of per-case `values`, which lie in [-bound, bound], by inverting a sign-flip test.

    A center d is kept unless flipping signs of `values - d` at random makes the observed sum
    rarer than `level / 2` in either tail. For a set of flipped cases that happens exactly when
    their mean is on the far side of d, so the bounds are quantiles of the means of random subsets
    of cases (Hartigan). That is exact when each case's value is symmetric about the mean, which
    sparse values are not. With a true interaction exactly at a 0.05 margin, the plain interval
    claimed equivalence for 9% of 100-case samples when one case in twenty was at +1, and 25% when
    one in forty was at +2 (the rest at 0); the budget is 2.5%. So, as Wilson adds
    pseudo-observations, one pseudo-case is added at each extreme, ±bound. Measured the same way
    (400 samples each): 0% and 6% at 100 cases, 1.5% and 4.3% at 200, 1.5% and 2.8% at 400.
    It also keeps a handful of cases that happen to agree from proving anything.
    """
    groups = Counter([round(v, 12) for v in values] + [bound, -bound])
    items = list(groups.items())
    rng = random.Random(rules.seed)
    means: list[float] = []
    empty = 0
    for _ in range(rules.resamples):
        total, chosen = 0.0, 0
        for value, count in items:
            k = rng.getrandbits(count).bit_count()  # how many of this value's cases are flipped
            total += value * k
            chosen += k
        if chosen:
            means.append(total / chosen)
        else:
            empty += 1  # no case flipped: the observed labelling itself, never in the far tail
    means.sort()
    # +1 counts the observed labelling, as in `decide`.
    need = math.ceil(rules.level / 2 * (rules.resamples + 1)) - 1 - empty
    if need <= 0 or need > len(means):
        return -bound, bound
    return max(-bound, means[need - 1]), min(bound, means[len(means) - need])


def _check(old: Mapping[str, Outcomes], new: Mapping[str, Outcomes]) -> list[str]:
    for label, side in (('old', old), ('new', new)):
        if set(side) != set(VERSIONS):
            raise ValueError(f'{label} must map exactly {VERSIONS} to outcomes, got {sorted(side)}')
    cells = [side[v] for side in (old, new) for v in VERSIONS]
    names = set(cells[0])
    if not names:
        raise ValueError('there are no cases to compare')
    if any(set(cell) != names for cell in cells):
        raise ValueError('all four cells must be scored on the same cases: both judges on both versions')
    for name in sorted(names):
        counts = [len(cell[name]) for cell in cells]
        if not counts[0] or len(set(counts)) > 1:
            raise ValueError(
                f'case {name!r} has {counts} outcomes (baseline/candidate under old, then new): every cell needs '
                'the same number, at least one, since both judges are meant to score the same outputs'
            )
    return sorted(names)


def compare_judges(
    old: Mapping[str, Outcomes],
    new: Mapping[str, Outcomes],
    *,
    margin: float = 0.05,
    rules: GateRules | None = None,
    old_certificate: Certificate | None = None,
    new_certificate: Certificate | None = None,
) -> JudgeBridge:
    """Score two agent versions with two judges and say whether the judge change preserves the gain.

    Args:
        old: `{'baseline': outcomes, 'candidate': outcomes}` under the old judge, each a case name
            to its pass/fail outcomes, as `decide` takes.
        new: The same outputs scored by the new judge.
        margin: The largest change in the mean per-case gain still called comparable, in pass-rate
            units (0.05 is five points).
        rules: Level, resamples and seed for the interval, and the rules for both decisions.
        old_certificate: Passed to `decide` for the old judge's decision; REFUSED if not ADMISSIBLE.
        new_certificate: The same for the new judge.
    """
    if not 0 < margin < 2:
        raise ValueError('margin must be between 0 and 2, the range of a difference of two gains')
    rules = rules or GateRules()
    names = _check(old, new)
    rate = {(v, j): {n: _rate(side[v][n]) for n in names} for j, side in (('old', old), ('new', new)) for v in VERSIONS}
    interaction = {
        n: (rate['candidate', 'new'][n] - rate['baseline', 'new'][n])
        - (rate['candidate', 'old'][n] - rate['baseline', 'old'][n])
        for n in names
    }

    def effect(per_case: Mapping[str, float], bound: float) -> Effect:
        values = [per_case[n] for n in names]
        return Effect(sum(values) / len(values), _interval(values, bound=bound, rules=rules))

    def shift(version: str) -> Effect:
        return effect({n: rate[version, 'new'][n] - rate[version, 'old'][n] for n in names}, 1.0)

    inter = effect(interaction, 2.0)
    low, high = inter.interval
    verdict: Comparability = (
        'PASS' if -margin <= low and high <= margin else 'FAIL' if low > margin or high < -margin else 'UNVALIDATED'
    )
    return JudgeBridge(
        rates={key: sum(per.values()) / len(per) for key, per in rate.items()},
        decision_old=decide(old['baseline'], old['candidate'], certificate=old_certificate, rules=rules),
        decision_new=decide(new['baseline'], new['candidate'], certificate=new_certificate, rules=rules),
        interaction=inter,
        shift_baseline=shift('baseline'),
        shift_candidate=shift('candidate'),
        margin=margin,
        verdict=verdict,
        cases=len(names),
        per_case=interaction,
    )


def compare_judge_reports(
    baseline: EvaluationReport[Any, Any, Any],
    candidate: EvaluationReport[Any, Any, Any],
    *,
    old: str,
    new: str,
    margin: float = 0.05,
    rules: GateRules | None = None,
    old_certificate: Certificate | None = None,
    new_certificate: Certificate | None = None,
) -> JudgeBridge:
    """`compare_judges` from two pydantic-evals reports that each carry both judges' assertions.

    Put the old and the new judge on the same `Dataset` and run it once per agent version: both
    judges then score the very same outputs, which is what separates the judge from the agent.
    `old` and `new` are the two assertions' names.
    """
    return compare_judges(
        {'baseline': outcomes(baseline, old), 'candidate': outcomes(candidate, old)},
        {'baseline': outcomes(baseline, new), 'candidate': outcomes(candidate, new)},
        margin=margin,
        rules=rules,
        old_certificate=old_certificate,
        new_certificate=new_certificate,
    )
