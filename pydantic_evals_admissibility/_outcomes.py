"""Learning from delayed outcomes: what the judge said at once, against what happened later.

A judge's verdict is instant; the truth often is not. A refund is reopened a week later, a ticket
is escalated, a task turns out to have been done. Those outcomes are the ground truth a
certificate's controls stand in for, and they arrive for real traffic. `outcome_calibration`
measures the judge against them, case by case; `recalibrated_pass_rate` uses that to correct
the judge's pass rate on new traffic into an estimate of how often the agent actually succeeded.

Two things make delayed outcomes harder than labels:

- **Pending cases.** An outcome that has not arrived is not a success. Cases without one are
  counted, reported, and left out of every rate.
- **Who gets an outcome depends on the verdict.** Often only passed tickets can be reopened, or
  failed ones are the ones a person looks at. Then the observed cases are not a sample of all
  cases, and sensitivity and specificity computed on them are biased. The report compares the
  share of passed and of failed verdicts that have an outcome, flags a gap whose interval
  excludes zero, and gives sensitivity and specificity adjusted for it (Begg and Greenes: the
  outcome rate within each verdict, scaled to that verdict's full count). The adjustment assumes
  that, given the verdict, whether an outcome arrives does not depend on the outcome itself. If
  failures surface sooner than successes, no reweighting of these data can fix that.

Rates on observed cases carry Wilson intervals, as elsewhere in the package; the case is the unit
throughout. The adjusted values and the corrected pass rate carry percentile intervals from a
parametric bootstrap: three rates, each counted in cases (the share of cases passed, and the
success rate among passed and among failed cases with an outcome), are drawn from their Jeffreys
posteriors, Beta(k + 1/2, n - k + 1/2). Resampling cases was the first version, and it gave a
zero-width interval whenever the judge made no mistake on one side: on the real support replies,
specificity 8/8 came out as [1.00, 1.00], where Wilson says [0.68, 1.00].
"""

from __future__ import annotations

import math
import random
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from ._stats import wilson

Cells = tuple[int, int, int, int, int, int]
"""Case counts by verdict, then outcome: passed (success, failure, pending), failed (the same)."""


@dataclass(frozen=True)
class Rate:
    successes: int
    trials: int
    interval: tuple[float, float]

    @property
    def rate(self) -> float | None:
        return self.successes / self.trials if self.trials else None


@dataclass(frozen=True)
class Estimate:
    """A value that is not a simple proportion, with a bootstrap interval; None when it is undefined."""

    value: float | None
    interval: tuple[float, float] | None


def _rate(k: int, n: int) -> Rate:
    return Rate(k, n, wilson(k, n))


@dataclass(frozen=True)
class OutcomeReport:
    cells: Cells
    sensitivity: Rate
    """Of observed successes, the share the judge passed."""
    specificity: Rate
    """Of observed failures, the share the judge failed."""
    ppv: Rate
    """Of passed verdicts with an outcome, the share that succeeded."""
    npv: Rate
    """Of failed verdicts with an outcome, the share that failed."""
    agreement: Rate
    observed_passed: Rate
    """Share of passed verdicts whose outcome has arrived."""
    observed_failed: Rate
    observation_gap: Estimate
    """`observed_passed - observed_failed`, with a Newcombe interval."""
    biased: bool
    """The observation rates differ (the gap's interval excludes zero, or one verdict has no outcomes)."""
    adjusted_sensitivity: Estimate
    adjusted_specificity: Estimate
    warnings: tuple[str, ...] = ()
    slices: dict[str, OutcomeReport] = field(default_factory=dict)

    @property
    def cases(self) -> int:
        return sum(self.cells)

    @property
    def pending(self) -> int:
        return self.cells[2] + self.cells[5]

    @property
    def observed(self) -> int:
        return self.cases - self.pending

    def table(self) -> str:
        rows = [f'{self.cases} cases, {self.observed} with an outcome, {self.pending} pending', '']
        rows.append(f'{"":<12} {"":<22} {"rate":>6}  95% interval')
        for name, r in self._rates().items():
            rows.append(f'{"":<12} {name:<22} {_fmt(r.rate):>6}  {_iv(r.interval)}  {r.successes}/{r.trials}')
        for name, e in (
            ('adjusted sensitivity', self.adjusted_sensitivity),
            ('adjusted specificity', self.adjusted_specificity),
        ):
            rows.append(f'{"":<12} {name:<22} {_fmt(e.value):>6}  {_iv(e.interval)}  bootstrap')
        for name, s in self.slices.items():
            sens, spec = s.sensitivity, s.specificity
            rows.append(
                f'{name:<12} sens {_fmt(sens.rate)} ({sens.successes}/{sens.trials}) {_iv(sens.interval)}  '
                f'spec {_fmt(spec.rate)} ({spec.successes}/{spec.trials}) {_iv(spec.interval)}  pending {s.pending}'
            )
        return '\n'.join(rows + [f'- {w}' for w in self.warnings])

    def _rates(self) -> dict[str, Rate]:
        return {
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'ppv': self.ppv,
            'npv': self.npv,
            'agreement': self.agreement,
            'observed (passed)': self.observed_passed,
            'observed (failed)': self.observed_failed,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            'cases': self.cases,
            'observed': self.observed,
            'pending': self.pending,
            'cells': dict(zip(_CELL_NAMES, self.cells, strict=True)),
            **{
                name.replace(' ', '_').replace('(', '').replace(')', ''): {
                    'rate': r.rate,
                    'successes': r.successes,
                    'trials': r.trials,
                    'interval': list(r.interval),
                }
                for name, r in self._rates().items()
            },
            'observation_gap': _est(self.observation_gap),
            'biased': self.biased,
            'adjusted_sensitivity': _est(self.adjusted_sensitivity),
            'adjusted_specificity': _est(self.adjusted_specificity),
            'warnings': list(self.warnings),
            'slices': {name: s.to_dict() for name, s in self.slices.items()},
        }


_CELL_NAMES = ('pass_success', 'pass_failure', 'pass_pending', 'fail_success', 'fail_failure', 'fail_pending')


def outcome_calibration(
    verdicts: Mapping[str, bool],
    outcomes: Mapping[str, bool | None],
    *,
    slices: Mapping[str, str] | None = None,
    resamples: int = 2000,
    seed: int = 0,
) -> OutcomeReport:
    """The judge's verdicts against the outcomes that arrived later, one case per key.

    Args:
        verdicts: Case id to the judge's verdict (True: passed).
        outcomes: Case id to what happened (True: the agent succeeded). A case missing here, or
            mapped to None, is pending: counted and reported, left out of every rate.
        slices: Case id to the kind of case, for a per-slice breakdown.
        resamples: Bootstrap draws for the adjusted sensitivity and specificity's intervals.
        seed: Seeds the bootstrap.
    """
    unknown = sorted(set(outcomes) - set(verdicts))
    if unknown:
        raise ValueError(f'outcomes for cases the judge never gave a verdict on: {unknown[:5]}')
    report = _report(_cells(verdicts, outcomes), resamples, seed)
    if slices is None:
        return report
    by_slice: dict[str, list[str]] = {}
    for case in verdicts:
        if case in slices:
            by_slice.setdefault(slices[case], []).append(case)
    parts = {
        name: _report(_cells({c: verdicts[c] for c in cases}, outcomes), resamples, seed)
        for name, cases in sorted(by_slice.items())
    }
    return replace(report, slices=parts)


def _cells(verdicts: Mapping[str, bool], outcomes: Mapping[str, bool | None]) -> Cells:
    counts = [0] * 6
    for case, passed in verdicts.items():
        outcome = outcomes.get(case)
        column = 2 if outcome is None else 0 if outcome else 1
        counts[(0 if passed else 3) + column] += 1
    return (counts[0], counts[1], counts[2], counts[3], counts[4], counts[5])


def _report(cells: Cells, resamples: int, seed: int) -> OutcomeReport:
    ps, pf, pm, fs, ff, fm = cells
    passed, failed = ps + pf + pm, fs + ff + fm
    observed_passed, observed_failed = _rate(ps + pf, passed), _rate(fs + ff, failed)
    gap = _newcombe(observed_passed, observed_failed)
    sens, spec = _rate(ps, ps + fs), _rate(ff, pf + ff)
    adjusted = _interval_pair(cells, resamples, seed) if _adjusted(cells) != (None, None) else (None, None)
    adj_sens = Estimate(_adjusted(cells)[0], adjusted[0])
    adj_spec = Estimate(_adjusted(cells)[1], adjusted[1])
    warnings: list[str] = []
    pending = pm + fm
    if pending:
        warnings.append(f'{pending} of {sum(cells)} cases have no outcome yet: not counted')
    one_blind = (passed and not ps + pf) or (failed and not fs + ff)
    biased = bool(
        passed and failed and (one_blind or (gap.interval is not None and (gap.interval[0] > 0 or gap.interval[1] < 0)))
    )
    if one_blind:
        which = 'passed' if not ps + pf else 'failed'
        warnings.append(
            f'no outcomes observed for {which} verdicts: sensitivity and specificity cannot be estimated, '
            f'only the {"NPV" if which == "passed" else "PPV"}; the observed sensitivity and specificity are artifacts'
        )
    elif biased:
        assert observed_passed.rate is not None and observed_failed.rate is not None and gap.interval is not None
        warnings.append(
            f'outcomes observed for {observed_passed.rate:.0%} of passed and {observed_failed.rate:.0%} of failed '
            f'verdicts (gap {observed_passed.rate - observed_failed.rate:+.2f}, interval {_iv(gap.interval)}): '
            f'sensitivity and specificity on observed cases are biased '
            f'(observed {_fmt(sens.rate)}/{_fmt(spec.rate)}, adjusted {_fmt(adj_sens.value)}/{_fmt(adj_spec.value)}); '
            'the adjustment assumes an outcome arrives independently of what it is, given the verdict'
        )
    return OutcomeReport(
        cells,
        sensitivity=sens,
        specificity=spec,
        ppv=_rate(ps, ps + pf),
        npv=_rate(ff, fs + ff),
        agreement=_rate(ps + ff, ps + pf + fs + ff),
        observed_passed=observed_passed,
        observed_failed=observed_failed,
        observation_gap=gap,
        biased=biased,
        adjusted_sensitivity=adj_sens,
        adjusted_specificity=adj_spec,
        warnings=tuple(warnings),
    )


def _sens_spec(q: float, a: float | None, b: float | None) -> tuple[float | None, float | None]:
    """Begg-Greenes: sensitivity and specificity from the pass rate `q` and the success rate within each verdict.

    `a` is the success rate among passed verdicts with an outcome, `b` among failed ones. Each is
    unbiased when, given the verdict, whether an outcome arrives does not depend on the outcome,
    and scaling them by the verdicts' full shares undoes the verdict-dependent observation.
    """
    if a is None or b is None:
        return None, None
    succ_p, fail_p, succ_f, fail_f = q * a, q * (1 - a), (1 - q) * b, (1 - q) * (1 - b)
    sens = succ_p / (succ_p + succ_f) if succ_p + succ_f else None
    spec = fail_f / (fail_p + fail_f) if fail_p + fail_f else None
    return sens, spec


def _adjusted(cells: Cells) -> tuple[float | None, float | None]:
    ps, pf, pm, fs, ff, fm = cells
    total = sum(cells)
    if not total:
        return None, None
    return _sens_spec((ps + pf + pm) / total, ps / (ps + pf) if ps + pf else None, fs / (fs + ff) if fs + ff else None)


def _draw_adjusted(rng: random.Random, cells: Cells) -> tuple[float | None, float | None]:
    """One draw of the adjusted pair, each of its three rates from its Jeffreys posterior."""
    ps, pf, pm, fs, ff, fm = cells
    q = rng.betavariate(ps + pf + pm + 0.5, fs + ff + fm + 0.5)
    a = rng.betavariate(ps + 0.5, pf + 0.5) if ps + pf else None
    b = rng.betavariate(fs + 0.5, ff + 0.5) if fs + ff else None
    return _sens_spec(q, a, b)


def _newcombe(a: Rate, b: Rate) -> Estimate:
    """Difference of two proportions with Newcombe's hybrid score interval, built from their Wilson intervals."""
    if a.rate is None or b.rate is None:
        return Estimate(None, None)
    (la, ua), (lb, ub) = a.interval, b.interval
    d = a.rate - b.rate
    low = d - math.sqrt((a.rate - la) ** 2 + (ub - b.rate) ** 2)
    high = d + math.sqrt((ua - a.rate) ** 2 + (b.rate - lb) ** 2)
    return Estimate(d, (low, high))


def _percentile(values: list[float], alpha: float) -> tuple[float, float] | None:
    if not values:
        return None
    values.sort()
    lo = values[max(0, int(alpha / 2 * len(values)))]
    hi = values[min(len(values) - 1, int((1 - alpha / 2) * len(values)))]
    return lo, hi


def _interval_pair(
    cells: Cells, resamples: int, seed: int
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    """95% intervals for the adjusted sensitivity and specificity; undefined draws are dropped."""
    rng = random.Random(seed)
    first: list[float] = []
    second: list[float] = []
    for _ in range(resamples):
        a, b = _draw_adjusted(rng, cells)
        if a is not None:
            first.append(a)
        if b is not None:
            second.append(b)
    return _percentile(first, 0.05), _percentile(second, 0.05)


@dataclass(frozen=True)
class PassRateEstimate:
    raw: Rate
    """The judge's pass rate on the new traffic."""
    corrected: float | None
    """Rogan-Gladen estimate of the true success rate, clipped to [0, 1]; None when it cannot be computed."""
    unclipped: float | None
    interval: tuple[float, float] | None
    sensitivity: float | None
    specificity: float | None
    resamples: int
    dropped: int
    """Resamples where the judge was no better than chance (sensitivity + specificity <= 1), or undefined."""
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            'raw': {'rate': self.raw.rate, 'passed': self.raw.successes, 'cases': self.raw.trials,
                    'interval': list(self.raw.interval)},
            'corrected': self.corrected,
            'unclipped': self.unclipped,
            'interval': list(self.interval) if self.interval else None,
            'sensitivity': self.sensitivity,
            'specificity': self.specificity,
            'resamples': self.resamples,
            'dropped': self.dropped,
            'warnings': list(self.warnings),
        }  # fmt: skip


def _rogan_gladen(p: float, sens: float | None, spec: float | None) -> float | None:
    if sens is None or spec is None or sens + spec - 1 <= 0:
        return None
    return (p + spec - 1) / (sens + spec - 1)


def recalibrated_pass_rate(
    verdicts: Mapping[str, bool] | Sequence[bool],
    calibration: OutcomeReport,
    *,
    alpha: float = 0.05,
    resamples: int = 2000,
    seed: int = 0,
) -> PassRateEstimate:
    """How often the agent really succeeds on new traffic, from the judge's pass rate there.

    A lenient judge passes failures too, so its pass rate overstates success; a strict one
    understates it. Rogan and Gladen: `(pass rate + specificity - 1) / (sensitivity + specificity - 1)`,
    with sensitivity and specificity from `calibration`, adjusted for verdict-dependent
    observation (see `OutcomeReport`). The interval is a parametric bootstrap that draws the new
    traffic's pass rate and the calibration's three rates (see the module docstring), so it
    carries the uncertainty in sensitivity and specificity as well as in the pass rate. The delta
    method was not used: the denominator near zero makes the estimate's spread lopsided, and a
    symmetric interval hides it. Draws where the judge is no better than chance are dropped and
    counted. Clipped to [0, 1].

    It assumes the judge's sensitivity and specificity on the new traffic are those it had on
    the calibration cases: a change in the kind of traffic can change them (see `slices`).
    """
    flags = list(verdicts.values()) if isinstance(verdicts, Mapping) else list(verdicts)
    k, n = sum(bool(v) for v in flags), len(flags)
    raw = _rate(k, n)
    sens, spec = _adjusted(calibration.cells)
    warnings = list(calibration.warnings)
    unclipped = _rogan_gladen(k / n, sens, spec) if n else None
    if n == 0:
        warnings.append('no verdicts on the new traffic')
    elif unclipped is None:
        warnings.append(
            'cannot correct: '
            + (
                'sensitivity or specificity cannot be estimated'
                if sens is None or spec is None
                else 'the judge is no better than chance'
            )
        )
    rng = random.Random(seed)
    values: list[float] = []
    dropped = 0
    if unclipped is not None:
        for _ in range(resamples):
            s, c = _draw_adjusted(rng, calibration.cells)
            estimate = _rogan_gladen(rng.betavariate(k + 0.5, n - k + 0.5), s, c)
            if estimate is None:
                dropped += 1
            else:
                values.append(min(1.0, max(0.0, estimate)))
        if dropped > alpha / 2 * resamples:
            warnings.append(
                f'in {dropped} of {resamples} resamples the judge was no better than chance: '
                'the interval leaves them out, and the judge is too close to chance for the correction to be trusted'
            )
    corrected = None if unclipped is None else min(1.0, max(0.0, unclipped))
    return PassRateEstimate(
        raw, corrected, unclipped, _percentile(values, alpha), sens, spec, resamples, dropped, tuple(warnings)
    )


def _fmt(x: float | None) -> str:
    return '-' if x is None else f'{x:.2f}'


def _iv(interval: tuple[float, float] | None) -> str:
    return '-' if interval is None else f'[{interval[0]:.2f}, {interval[1]:.2f}]'


def _est(e: Estimate) -> dict[str, Any]:
    return {'value': e.value, 'interval': list(e.interval) if e.interval else None}
