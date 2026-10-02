"""Decide whether a candidate (a prompt, a config) should replace the baseline.

An optimizer that proposes changes and keeps whatever scores higher will, given enough
proposals, keep changes that only scored higher by chance, or only pleased its judge. `decide`
is the step in between: it compares the two on the same cases, case by case, and promotes only
when the improvement holds up.

- **REFUSED** when the scores came from a judge whose certificate is not ADMISSIBLE: those
  scores are not evidence of anything, so there is nothing to decide.
- **PROMOTE** when a paired sign-flip test says the mean per-case gain is above zero at the
  chosen level and the observed gain exceeds `min_gain` (a floor on the estimate, not a second
  test; and, if `max_regressions` is set, few enough cases got worse).
- **REJECT** when the same test says it is below zero (or, if set, too many cases got worse).
- **INCONCLUSIVE** otherwise: the cases cannot tell the two apart. That is the expected answer
  when the candidate is the baseline.

The comparison is paired: each case is compared with itself, so a hard case drags both sides
down equally instead of adding noise to the difference. The decision uses a sign-flip
permutation test. It assumes the sharp null of no per-case difference: each case's outcomes are
then exchangeable between the two versions, so its gain is as likely negative as positive, and
the test is exact. That holds only if both versions ran each case equally often; with 1 run of
the baseline against 100 of the candidate, the baseline's rate is far noisier and the gain is not
symmetric, so `decide` refuses unequal counts. A bootstrap interval of the mean gain is reported as the
estimate, but not used to decide, because with few binary cases it is too narrow and made
false calls in this package's own A/A test.
"""

from __future__ import annotations

import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from ._certify import Certificate

Decision = Literal['PROMOTE', 'REJECT', 'INCONCLUSIVE', 'REFUSED']


@dataclass(frozen=True)
class GateRules:
    min_gain: float = 0.0
    """The observed mean per-case gain must exceed this to promote; the test is against zero.

    It is a floor on the point estimate, not a test that the true gain exceeds `min_gain`: a
    candidate is promoted when the gain is shown to be above zero and the estimate is above this.
    """
    max_regressions: float | None = None
    """Share of cases allowed to get worse in a promoted candidate; None to not check.

    Off by default: with a few noisy runs per case, a truly better candidate still shows chance
    drops on many cases, and in this package's own simulation a 10% limit rejected a candidate
    that was 0.1 better 13 times in 50. Set it when each case's score is not noisy.
    """
    level: float = 0.05
    """Two-sided: each direction gets half, so identical versions are acted on at most this often."""
    resamples: int = 10_000
    seed: int = 0

    def for_candidates(self, k: int) -> GateRules:
        """Rules for choosing among `k` candidates: the level is split between them (Bonferroni).

        Testing five candidates at 5% each and keeping any that passes promotes noise far more
        often than 5% of the time; splitting the level keeps the chance of promoting any of them,
        when none is better, at most `level`.
        """
        return replace(self, level=self.level / k)


@dataclass(frozen=True)
class GateResult:
    decision: Decision
    reason: str
    mean_gain: float | None = None
    interval: tuple[float, float] | None = None
    cases: int = 0
    improved: int = 0
    regressed: int = 0
    p_better: float | None = None
    p_worse: float | None = None
    per_case: dict[str, float] = field(default_factory=dict, repr=False)

    def summary(self) -> str:
        if self.interval is None:
            return f'{self.decision}: {self.reason}'
        low, high = self.interval
        return (
            f'{self.decision}: mean gain {self.mean_gain:+.3f} per case, bootstrap interval [{low:+.3f}, {high:+.3f}], '
            f'p(better)={self.p_better:.3f}, p(worse)={self.p_worse:.3f}; '
            f'{self.improved} improved, {self.regressed} regressed of {self.cases}. {self.reason}'
        )


def _rate(outcomes: Sequence[bool]) -> float:
    return sum(outcomes) / len(outcomes)


def _refusal(certificate: Certificate | None, judge: Any) -> GateResult | None:
    """REFUSED unless the certificate is ADMISSIBLE and, when the judge is given, is about it."""
    if certificate is None:
        return None
    if judge is not None:
        changed = certificate.differences(judge)
        if changed:
            return GateResult(
                'REFUSED', f'the certificate is for another configuration of the judge ({", ".join(changed)} changed)'
            )
    if not certificate.admissible:
        failing = ', '.join(c.name for c in certificate.checks if c.status != 'PASS')
        return GateResult('REFUSED', f'the judge is {certificate.verdict} ({failing}); its scores are not evidence')
    return None


def decide(
    baseline: Mapping[str, Sequence[bool]],
    candidate: Mapping[str, Sequence[bool]],
    *,
    certificate: Certificate | None = None,
    rules: GateRules | None = None,
    judge: Any = None,
) -> GateResult:
    """Compare per-case pass/fail outcomes (one or more repeats per case) and decide.

    Args:
        baseline: Case name to the outcomes the current version got on it.
        candidate: The same for the proposed version. Must cover the same cases, each with as
            many outcomes as the baseline has for it: the sign-flip test is exact only then.
        certificate: The certificate of the judge that produced the outcomes. Leave it out only
            when the outcomes come from ground truth rather than a judge.
        rules: The thresholds.
        judge: The judge that produced the outcomes. Given, the gate refuses a certificate issued
            for a different configuration of it (another rubric, model, or settings).
    """
    rules = rules or GateRules()
    refused = _refusal(certificate, judge)
    if refused is not None:
        return refused
    if not baseline or not candidate:
        raise ValueError('there are no cases to compare')
    if set(baseline) != set(candidate):
        raise ValueError('baseline and candidate must be scored on the same cases')
    names = sorted(baseline)
    if any(not baseline[n] or not candidate[n] for n in names):
        raise ValueError('every case needs at least one outcome on each side')
    unequal = [n for n in names if len(baseline[n]) != len(candidate[n])]
    if unequal:
        n = unequal[0]
        raise ValueError(
            f'every case needs the same number of outcomes on each side ({len(unequal)} differ; '
            f'{n!r} has {len(baseline[n])} baseline and {len(candidate[n])} candidate): with unequal counts '
            'the sign-flip test is not exact'
        )
    gains = {n: _rate(candidate[n]) - _rate(baseline[n]) for n in names}
    values = [gains[n] for n in names]
    mean = sum(values) / len(values)

    rng = random.Random(rules.seed)
    k = len(values)
    means = sorted(sum(rng.choice(values) for _ in range(k)) / k for _ in range(rules.resamples))
    low = means[int(0.025 * rules.resamples)]
    high = means[min(rules.resamples - 1, int(0.975 * rules.resamples))]

    # Sign-flip: under the sharp null (no per-case difference, equal counts) each case's gain is as
    # likely negated, so the observed mean is compared with the means of randomly sign-flipped gains.
    # +1 counts the observed labelling.
    observed = sum(values)
    at_least = at_most = 1
    for _ in range(rules.resamples):
        flipped = sum(v if rng.random() < 0.5 else -v for v in values)
        at_least += flipped >= observed - 1e-12
        at_most += flipped <= observed + 1e-12
    p_better = at_least / (rules.resamples + 1)
    p_worse = at_most / (rules.resamples + 1)

    improved = sum(g > 0 for g in values)
    regressed = sum(g < 0 for g in values)

    def result(decision: Decision, reason: str) -> GateResult:
        return GateResult(
            decision,
            reason,
            mean_gain=mean,
            interval=(low, high),
            cases=k,
            improved=improved,
            regressed=regressed,
            p_better=p_better,
            p_worse=p_worse,
            per_case=gains,
        )

    half = rules.level / 2
    if p_worse < half:
        return result('REJECT', 'the candidate is worse beyond the noise')
    if p_better < half and mean > rules.min_gain:
        if rules.max_regressions is not None and regressed > rules.max_regressions * k:
            limit = f'{rules.max_regressions:.0%}'
            return result('REJECT', f'better on average, but {regressed} cases regressed (limit {limit})')
        return result('PROMOTE', 'the gain is beyond what chance produces when nothing changed')
    return result('INCONCLUSIVE', 'the cases cannot tell the two apart')


def detectable_gain(
    baseline: Mapping[str, Sequence[bool]],
    *,
    repeats: int | None = None,
    power: float = 0.8,
    rules: GateRules | None = None,
    trials: int = 100,
    gains: Sequence[float] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5),
    seed: int = 0,
) -> float | None:
    """The smallest shift in per-case pass rate that `decide` would promote at least `power` of the time.

    Answers "can this dataset see the improvement I'm hoping for?" before an optimizer runs. Each
    case's observed pass rate stands in for its true rate; a candidate is simulated as that rate
    plus the shift, capped at 1, with the same number of repeats, and decided against a fresh
    baseline draw. Returns the shift, or None if no shift in `gains` is detected reliably enough.

    The shift is not the mean improvement it simulated: the cap takes gain away from cases already
    near 1 (a case at 0.95 shifted by 0.2 gains 0.05), so the simulated mean gain is
    `mean(min(1, p + shift) - p)` over the cases' rates, at most the shift. On a dataset with many
    near-perfect cases the improvement actually detected is smaller than the number returned.
    """
    rules = rules or GateRules(resamples=1000)
    rng = random.Random(seed)
    if not baseline or any(not outcomes for outcomes in baseline.values()):
        raise ValueError('the baseline needs at least one case, each with at least one outcome')
    rates = {name: _rate(outcomes) for name, outcomes in baseline.items()}
    n = repeats or max(len(o) for o in baseline.values())

    def draw(shift: float) -> dict[str, list[bool]]:
        return {name: [rng.random() < min(1.0, p + shift) for _ in range(n)] for name, p in rates.items()}

    for gain in sorted(gains):
        promoted = sum(
            decide(draw(0.0), draw(gain), rules=replace(rules, seed=rng.randrange(2**31))).decision == 'PROMOTE'
            for _ in range(trials)
        )
        if promoted / trials >= power:
            return gain
    return None


def decide_unpaired(
    baseline: Sequence[bool],
    candidate: Sequence[bool],
    *,
    certificate: Certificate | None = None,
    rules: GateRules | None = None,
    judge: Any = None,
) -> GateResult:
    """The gate for traffic that is not paired by case: two arms of a canary, say.

    Each request is scored once, in one arm. The test permutes which arm each outcome belongs to;
    when the arms are the same, every assignment is equally likely, so the test is exact.
    """
    rules = rules or GateRules()
    refused = _refusal(certificate, judge)
    if refused is not None:
        return refused
    if not baseline or not candidate:
        raise ValueError('both arms need at least one outcome')
    b, c = [bool(x) for x in baseline], [bool(x) for x in candidate]
    observed = _rate(c) - _rate(b)
    rng = random.Random(rules.seed)

    boots = sorted(
        _rate([rng.choice(c) for _ in c]) - _rate([rng.choice(b) for _ in b]) for _ in range(rules.resamples)
    )
    low, high = boots[int(0.025 * rules.resamples)], boots[min(rules.resamples - 1, int(0.975 * rules.resamples))]

    pooled = b + c
    at_least = at_most = 1
    for _ in range(rules.resamples):
        rng.shuffle(pooled)
        diff = _rate(pooled[len(b) :]) - _rate(pooled[: len(b)])
        at_least += diff >= observed - 1e-12
        at_most += diff <= observed + 1e-12
    p_better, p_worse = at_least / (rules.resamples + 1), at_most / (rules.resamples + 1)

    def result(decision: Decision, reason: str) -> GateResult:
        return GateResult(
            decision,
            reason,
            mean_gain=observed,
            interval=(low, high),
            cases=len(b) + len(c),
            p_better=p_better,
            p_worse=p_worse,
        )

    half = rules.level / 2
    if p_worse < half:
        return result('REJECT', 'the candidate arm is worse beyond the noise')
    if p_better < half and observed > rules.min_gain:
        return result('PROMOTE', 'the candidate arm is better beyond the noise')
    return result('INCONCLUSIVE', 'not enough traffic yet to tell the arms apart')
