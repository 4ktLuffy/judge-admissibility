"""Human review aimed at one release decision: which cases to label next, and when to stop.

The gate compares a baseline and a candidate with a judge. When the judge is not trusted enough,
or the comparison is inconclusive, people label cases. `ReviewPlan` groups the cases by what the
judge said (candidate better, the same, worse), samples each group at random, and estimates the
true gain over the whole dataset from the labels, each group weighted by its size. Grouping lets
an untrusted judge's verdicts still sharpen the estimate: in simulation it decided more often,
and with fewer labels, than labelling at random (`bench/review_budget.py`). Every group is
sampled: agreement with the judge is never assumed to be correctness.

It looks at the evidence on a schedule fixed in advance (`looks`, in labels) and stops at the
first look where the interval excludes zero, each look at its share of the error budget.
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from statistics import NormalDist
from typing import Literal

ReviewDecision = Literal['PROMOTE', 'REJECT', 'INCONCLUSIVE']
Stratum = str  # 'judge: candidate better', 'judge: same' or 'judge: candidate worse'
_STRATA: tuple[Stratum, ...] = ('judge: candidate better', 'judge: same', 'judge: candidate worse')


def _mean(values: Sequence[bool]) -> float:
    return sum(values) / len(values)


def _stratum(baseline: Sequence[bool], candidate: Sequence[bool]) -> Stratum:
    gain = _mean(candidate) - _mean(baseline)
    return _STRATA[0] if gain > 0 else _STRATA[2] if gain < 0 else _STRATA[1]


@dataclass
class ReviewResult:
    """What the labels so far say about the true gain of the candidate over the baseline."""

    decision: ReviewDecision
    estimate: float | None
    interval: tuple[float, float] | None
    labels: int
    look: int
    by_stratum: dict[str, tuple[int, int, float | None]]
    """Stratum to (cases in it, labelled, mean true gain among the labelled)."""
    judge_gain: float

    def summary(self) -> str:
        if self.estimate is None or self.interval is None:
            return f'{self.decision}: no labels yet; the judge alone says {self.judge_gain:+.3f}'
        low, high = self.interval
        return (
            f'{self.decision}: true gain {self.estimate:+.3f} [{low:+.3f}, {high:+.3f}] from {self.labels} labels '
            f'(look {self.look}); the judge said {self.judge_gain:+.3f}'
        )


@dataclass
class ReviewPlan:
    """A labelling plan for one baseline-vs-candidate decision, fixed before any label is seen.

    Build it with `ReviewPlan.from_verdicts`. Call `next_batch()` for the cases to label up to the
    next look, `add_labels` with people's verdicts, and `decision()` after each batch.
    """

    strata: dict[Stratum, list[str]]
    judge_gain: float
    looks: tuple[int, ...]
    allocation: dict[Stratum, float]
    alpha: float = 0.05
    seed: int = 0
    labels: dict[str, float] = field(default_factory=dict)
    """Case name to its true gain: mean of the candidate's labels minus mean of the baseline's."""
    _order: dict[Stratum, list[str]] = field(default_factory=dict, repr=False)

    @classmethod
    def from_verdicts(
        cls,
        baseline: Mapping[str, Sequence[bool]],
        candidate: Mapping[str, Sequence[bool]],
        *,
        looks: Sequence[int] = (10, 20, 40),
        allocation: Literal['proportional', 'disagreement'] = 'proportional',
        alpha: float = 0.05,
        seed: int = 0,
    ) -> ReviewPlan:
        """Plan from the judge's per-case verdicts on both versions (as `decide` takes them).

        `allocation='proportional'` samples every group at the same rate. `'disagreement'` labels
        the cases where the judge saw a difference twice as densely; it is fixed before any label
        is seen, so it does not bias the estimate, but in simulation it decided no more often than
        proportional when the judge's mistakes were spread evenly (`bench/review_budget.py`).
        """
        if set(baseline) != set(candidate) or not baseline:
            raise ValueError('baseline and candidate must cover the same, non-empty set of cases')
        if any(not baseline[c] or not candidate[c] for c in baseline):
            raise ValueError('every case needs at least one verdict for each version')
        looks = tuple(sorted(set(looks)))
        if not looks or looks[0] < 1:
            raise ValueError('looks must be positive label counts')
        strata: dict[Stratum, list[str]] = {s: [] for s in _STRATA}
        for case in sorted(baseline):
            strata[_stratum(baseline[case], candidate[case])].append(case)
        strata = {s: cases for s, cases in strata.items() if cases}
        if looks[0] < 2 * len(strata):
            raise ValueError(f'the first look needs at least {2 * len(strata)} labels: two in every group')
        density = {s: 1.0 if s == 'judge: same' or allocation == 'proportional' else 2.0 for s in strata}
        judge_gain = sum(_mean(candidate[c]) - _mean(baseline[c]) for c in baseline) / len(baseline)
        rng = random.Random(seed)
        order = {s: rng.sample(cases, len(cases)) for s, cases in strata.items()}
        return cls(strata, judge_gain, looks, density, alpha, seed, _order=order)

    @property
    def size(self) -> int:
        return sum(len(cases) for cases in self.strata.values())

    def _quota(self, total: int) -> dict[Stratum, int]:
        """Labels per group for `total` labels: at least two each, the rest by size times density."""
        quota = {s: min(2, len(c)) for s, c in self.strata.items()}
        weights = {s: len(c) * self.allocation[s] for s, c in self.strata.items()}
        while sum(quota.values()) < min(total, self.size):
            open_ = [s for s in self.strata if quota[s] < len(self.strata[s])]
            # The group furthest below its share gets the next label: a fixed rule, not the data.
            s = min(open_, key=lambda s: (quota[s] / weights[s], s))
            quota[s] += 1
        return quota

    def next_batch(self) -> list[str]:
        """The cases to label before the next look; empty when every look has been reached."""
        target = next((n for n in self.looks if n > len(self.labels)), None)
        if target is None:
            return []
        quota = self._quota(target)
        return [c for s, q in quota.items() for c in self._order[s][:q] if c not in self.labels]

    def add_labels(self, labels: Mapping[str, tuple[Sequence[bool], Sequence[bool]]]) -> None:
        """People's verdicts per case: (on the baseline's outputs, on the candidate's outputs)."""
        for case, (baseline, candidate) in labels.items():
            if not any(case in cases for cases in self.strata.values()):
                raise KeyError(f'unknown case {case!r}')
            self.labels[case] = _mean(candidate) - _mean(baseline)

    def decision(self) -> ReviewResult:
        """The decision at the latest look reached; INCONCLUSIVE until the interval excludes zero."""
        look = sum(n <= len(self.labels) for n in self.looks)
        by_stratum: dict[str, tuple[int, int, float | None]] = {}
        estimate, variance, complete = 0.0, 0.0, True
        for s, cases in self.strata.items():
            got = [self.labels[c] for c in cases if c in self.labels]
            by_stratum[s] = (len(cases), len(got), sum(got) / len(got) if got else None)
            if len(got) < 2:
                complete = False
                continue
            weight = len(cases) / self.size
            estimate += weight * sum(got) / len(got)
            variance += weight**2 * (1 - len(got) / len(cases)) * _variance(got) / len(got)
        if not complete or look == 0:
            return ReviewResult('INCONCLUSIVE', None, None, len(self.labels), look, by_stratum, self.judge_gain)
        z = NormalDist().inv_cdf(1 - self.alpha / (2 * len(self.looks)))
        half = z * math.sqrt(variance)
        low, high = max(-1.0, estimate - half), min(1.0, estimate + half)
        decision: ReviewDecision = 'PROMOTE' if low > 0 else 'REJECT' if high < 0 else 'INCONCLUSIVE'
        return ReviewResult(decision, estimate, (low, high), len(self.labels), look, by_stratum, self.judge_gain)


def _variance(values: Sequence[float]) -> float:
    """Sample variance with one pseudo-label at each extreme (+1, -1), so that a few labels that
    happen to agree cannot claim certainty; the same idea as Wilson's pseudo-observations."""
    padded = [*values, 1.0, -1.0]
    mean = sum(padded) / len(padded)
    return sum((v - mean) ** 2 for v in padded) / (len(padded) - 1)
