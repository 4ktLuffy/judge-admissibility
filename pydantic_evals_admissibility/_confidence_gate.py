"""Certify a judge's confidence gate: it decides when sure, and hands the rest to a fallback.

pydantic-ai #9641 measured an `LLMJudge` on a decision model against 390 labelled answers: 308
right with every verdict taken at p >= 0.5, 356 when the model decides only at p >= 0.8 or
p <= 0.2 (a confidence of 0.6 in pydantic-ai's terms) and an LLM judge decides the rest. The plan
in #9723 (`LLMJudge(probability=True)`, `decision_boolean_confidence_threshold`, `UnsureBoolean`)
makes that gate a setting. A gate is then a claim, "what it decides alone is right often enough",
and the claim needs the same evidence as any judge: labelled cases, exact intervals, and a
verdict that can come out UNVALIDATED.

`certify_confidence_gate` measures one gate on labelled cases:

- `auto_accuracy`: of the cases the gate decides alone, the share it decides as people did,
  with an exact (Clopper-Pearson) interval. The decision rests on this check.
- `coverage`: the share it decides alone; checked only when `min_coverage` is set.
- `end_to_end`: when the fallback's verdicts on the deferred cases are given, the accuracy of
  gate plus fallback over every case; checked only when `min_end_to_end` is set.

Checks are decided as in `certify_judge` (`_check`): PASS when the exact 95% lower bound clears
the bar, FAIL when an upper bound widened for every check that can fail is below it, otherwise
UNVALIDATED, and UNVALIDATED below `min_trials` units.

**The unit of evidence.** A case given more than once (repeats of the judge) counts once, by its
first record, as `certify_judge`'s acceptance does; the repeats only measure whether the gate
decides the case the same way each time (`stability`). Outputs that are not independent (two
replies to one question, a reply and a wrong answer built from the same question) share a
`group`, and with `unit='group'` (the default) the group is the unit: the exact intervals count
groups, not outputs, and a split never puts one group on both sides. A group is right when every
output the gate decided in it was decided right, decided when every output in it was, and right
end to end when every output was, after the fallback. That is stricter than the per-output rate
(also reported, as `output_accuracy`), and exact if the groups are independent. Without a
`group`, each case is its own group, so nothing changes for independent cases. `unit='case'`
counts outputs even when they are grouped; its intervals are then too narrow.

**Choosing the thresholds is the selection problem.** Pick (low, high) to maximise coverage
subject to accuracy on the same labels, and the accuracy reported for the pick is the best of
many noisy estimates: biased up, and its interval no longer covers. #9641's 0.8/0.2 were not
shown to be chosen on held-out labels. So a report knows which cases its thresholds were chosen
on (`chosen_on`), and a PASS on any of them is downgraded to UNVALIDATED. `calibrate_gate` does
the choosing honestly, one of two ways:

- `method='split'`: choose on a random half of the groups (stratified by the labels in them),
  certify the choice on the other half. The confirmation half played no part in the choice, so its interval
  is an ordinary exact interval. Both reports are returned; only the confirmed one decides.
- `method='fixed_sequence'`: every case is used for both, and the selection is paid for by
  testing. The candidate gates are put in order from the one that decides least to the one that
  decides most, from the probabilities alone (labels unseen), and tested in that order, each at
  the full level, keeping the last that passes and stopping at the first that does not. Testing a
  fixed sequence holds the family-wise error at the level (the Learn-then-Test procedure of
  Angelopoulos et al., arXiv 2110.01052). It needs the candidates to be ordered before labels are
  seen, which is why it takes a single family (`confidence_grid`), not any grid. Coverage needs
  no correction: it is a function of the probabilities alone. What it controls is the verdict,
  not the estimate: the reported accuracy of the gate it keeps is still the best of several, and
  biased up (measured in `bench/confidence_gate.py`). Read the estimate from `split`.

The estimand of both is the gate's accuracy on cases like these, conditional on the
probabilities the judge gave; the cases must be a random sample of what the gate will see. A
review set enriched with wrong answers (as #9641's 245 were) certifies the gate on that mix only.

Probabilities are what a decision model reports. A score in another range works, with
thresholds in that range; a confidence the judge states in words is not a probability, and its
gate is certified only for that judge and prompt.
"""

from __future__ import annotations

import random
from collections.abc import Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from functools import lru_cache
from math import isfinite
from statistics import NormalDist
from typing import Any, Literal

from ._certify import Check, CheckStatus, _check  # pyright: ignore[reportPrivateUsage]
from ._stats import clopper_pearson

GateDecision = Literal['pass', 'fail', 'defer']
Unit = Literal['group', 'case']
SelectionMethod = Literal['split', 'fixed_sequence']
Criterion = Literal['lower_bound', 'estimate']


@lru_cache(maxsize=65536)
def _cp(successes: int, trials: int) -> tuple[float, float]:
    """Clopper-Pearson at 95%, cached: choosing among gates asks for the same counts many times."""
    return clopper_pearson(successes, trials)


@dataclass(frozen=True)
class GateCase:
    """One judged output: the judge's probability that it passes, and people's verdict on it."""

    name: str
    probability: float
    """The judge's probability (or score) that the output passes; compared with the gate's thresholds."""
    label: bool
    """People's verdict: does the output pass?"""
    fallback: bool | None = None
    """The fallback decider's verdict (another judge, or a person), when it was asked."""
    group: str | None = None
    """What makes outputs dependent, such as the question they answer; None makes the case its own group."""

    @property
    def unit_key(self) -> str:
        return self.name if self.group is None else self.group


@dataclass(frozen=True)
class ConfidenceGate:
    """Decide pass at `probability >= high`, fail at `probability <= low`, defer in between.

    Equality is accepted on both sides, as in #9723. `low == high` decides every case: the
    ungated judge (`ConfidenceGate(0.5, 0.5)` is `LLMJudge` as it is, pass at p >= 0.5).
    """

    low: float
    high: float

    def __post_init__(self) -> None:
        if not (isfinite(self.low) and isfinite(self.high)) or self.low > self.high:
            raise ValueError(f'a gate needs finite low <= high, got low={self.low}, high={self.high}')

    @classmethod
    def from_confidence(cls, threshold: float, cutoff: float = 0.5) -> ConfidenceGate:
        """The gate of `decision_boolean_confidence_threshold=threshold` (#9723).

        Confidence is the distance from the verdict `cutoff`, scaled to [0, 1] on each side, so
        at cutoff 0.5 a threshold of 0.6 decides at p >= 0.8 or p <= 0.2.
        """
        if not 0 <= threshold <= 1 or not 0 < cutoff < 1:
            raise ValueError(f'threshold must be in [0, 1] and cutoff in (0, 1), got {threshold}, {cutoff}')
        high = round(cutoff + threshold * (1 - cutoff), 12)
        low = round(cutoff - threshold * cutoff, 12)
        return cls(low, high)

    def decide(self, probability: float) -> GateDecision:
        if probability >= self.high:
            return 'pass'
        if probability <= self.low:
            return 'fail'
        return 'defer'

    def describe(self) -> str:
        if self.low == self.high:
            return f'ungated (pass at p >= {self.high:g})'
        return f'pass at p >= {self.high:g}, fail at p <= {self.low:g}, defer between'


def confidence_grid(step: float = 0.05, top: float = 0.95, cutoff: float = 0.5) -> tuple[ConfidenceGate, ...]:
    """Gates of confidence 0, `step`, ..., `top`: the family #9723's setting ranges over, least deferring first."""
    count = round(top / step)
    return tuple(ConfidenceGate.from_confidence(round(i * step, 10), cutoff) for i in range(count + 1))


DEFAULT_CANDIDATES = confidence_grid()


@dataclass(frozen=True)
class ConfidenceRequirements:
    """What a gate must show. Rates are compared on their exact interval, as in `Thresholds`."""

    min_accuracy: float = 0.9
    """Lower bound on the share of auto-decided cases decided as people did."""
    min_coverage: float = 0.0
    """Lower bound on the share decided without the fallback; 0 does not check it."""
    min_end_to_end: float | None = None
    """Lower bound on the accuracy of gate plus fallback over every case; None does not check it."""
    min_trials: int = 10
    """Fewer cases than this behind a check and it is UNVALIDATED, whatever its rate."""


DEFAULT_REQUIREMENTS = ConfidenceRequirements()


@dataclass(frozen=True)
class GateReport:
    """A gate measured on labelled cases. `status` is PASS only if every check passed on cases it was not chosen on."""

    gate: ConfidenceGate
    status: CheckStatus
    reason: str
    checks: tuple[Check, ...]
    cases: int
    decided: int
    """Units (groups, or cases) with at least one output decided by the gate."""
    deferred: int
    """Outputs deferred to the fallback."""
    auto_correct: int
    """Units whose every auto-decided output was decided as people did."""
    false_passes: int
    """Auto-decided pass where people said fail: the mistake an eval report hides."""
    false_fails: int
    end_to_end_correct: int | None = None
    repeated_cases: int = 0
    stability: float | None = None
    """Of the cases given more than once, the share the gate decided the same way every time."""
    chosen_on: int = 0
    """How many of these cases the thresholds were chosen on; nonzero makes the accuracy optimistic."""
    per_case: dict[str, GateDecision] = field(default_factory=dict[str, GateDecision], repr=False)
    unit: Unit = 'group'
    units: int = 0
    """Independent units the intervals count: groups, or cases with `unit='case'`."""
    fully_decided: int = 0
    """Units the gate decided entirely, with no output deferred."""
    outputs_decided: int = 0
    outputs_correct: int = 0

    def check(self, name: str) -> Check | None:
        return next((c for c in self.checks if c.name == name), None)

    @property
    def accuracy(self) -> float | None:
        return self.auto_correct / self.decided if self.decided else None

    @property
    def coverage(self) -> float | None:
        return self.fully_decided / self.units if self.units else None

    @property
    def output_accuracy(self) -> float | None:
        """Per output, descriptive only: outputs in a group are not independent evidence."""
        return self.outputs_correct / self.outputs_decided if self.outputs_decided else None

    @property
    def end_to_end(self) -> float | None:
        return None if self.end_to_end_correct is None or not self.units else self.end_to_end_correct / self.units

    def table(self) -> str:
        rows = [f'gate {self.gate.describe()}: {self.status} ({self.reason})', '']
        rows.append(f'{"check":<16} {"status":<12} {"rate":>6}  {"95% interval":<15} {"needs":>6}  n')
        for c in self.checks:
            rate = '-' if c.rate is None else f'{c.rate:.3f}'
            low, high = c.interval
            rows.append(
                f'{c.name:<16} {c.status:<12} {rate:>6}  [{low:.3f}, {high:.3f}]  >= {c.threshold:.2f}  {c.trials}'
                + (f'  {c.detail}' if c.detail else '')
            )
        rows.append(
            (f'{self.cases} cases in {self.units} groups' if self.units != self.cases else f'{self.cases} cases')
            + f': {self.outputs_decided} outputs decided '
            f'({self.false_passes} false passes, {self.false_fails} false fails), {self.deferred} deferred'
        )
        return '\n'.join(rows)

    def to_dict(self) -> dict[str, Any]:
        return {
            'gate': {'low': self.gate.low, 'high': self.gate.high, 'describe': self.gate.describe()},
            'status': self.status,
            'reason': self.reason,
            'unit': self.unit,
            'units': self.units,
            'cases': self.cases,
            'decided': self.decided,
            'fully_decided': self.fully_decided,
            'outputs_decided': self.outputs_decided,
            'outputs_correct': self.outputs_correct,
            'output_accuracy': self.output_accuracy,
            'deferred': self.deferred,
            'auto_correct': self.auto_correct,
            'false_passes': self.false_passes,
            'false_fails': self.false_fails,
            'accuracy': self.accuracy,
            'coverage': self.coverage,
            'end_to_end': self.end_to_end,
            'end_to_end_correct': self.end_to_end_correct,
            'repeated_cases': self.repeated_cases,
            'stability': self.stability,
            'chosen_on': self.chosen_on,
            'checks': [
                {
                    'name': c.name,
                    'status': c.status,
                    'successes': c.successes,
                    'trials': c.trials,
                    'rate': c.rate,
                    'interval': list(c.interval),
                    'threshold': c.threshold,
                    'detail': c.detail,
                }
                for c in self.checks
            ],
        }


def _by_case(cases: Iterable[GateCase]) -> tuple[list[GateCase], dict[str, list[GateCase]]]:
    """The first record of each case, in order, and every record by name; refuses contradictory labels."""
    records: dict[str, list[GateCase]] = {}
    for case in cases:
        if not isfinite(case.probability):
            raise ValueError(f'case {case.name!r} has a probability of {case.probability}')
        records.setdefault(case.name, []).append(case)
    for name, rs in records.items():
        if len({r.label for r in rs}) > 1:
            raise ValueError(f'case {name!r} is labelled both pass and fail: a repeat must be the same output')
        if len({r.group for r in rs}) > 1:
            raise ValueError(f'case {name!r} is given in more than one group')
    return [rs[0] for rs in records.values()], records


def _tally(gate: ConfidenceGate, cases: Sequence[GateCase]) -> tuple[int, int, int, int]:
    """(decided, correct, false passes, false fails) of `gate` on one record per case."""
    decided = correct = false_passes = false_fails = 0
    for case in cases:
        decision = gate.decide(case.probability)
        if decision == 'defer':
            continue
        decided += 1
        if (decision == 'pass') == case.label:
            correct += 1
        elif decision == 'pass':
            false_passes += 1
        else:
            false_fails += 1
    return decided, correct, false_passes, false_fails


def _units(first: Sequence[GateCase], unit: Unit) -> list[list[GateCase]]:
    """The independent units: one list of cases per group (or per case), in first-seen order."""
    if unit not in ('group', 'case'):
        raise ValueError(f"unit must be 'group' or 'case', got {unit!r}")
    out: dict[str, list[GateCase]] = {}
    for case in first:
        out.setdefault(case.unit_key if unit == 'group' else case.name, []).append(case)
    return list(out.values())


def _unit_tally(gate: ConfidenceGate, units: Sequence[Sequence[GateCase]]) -> tuple[int, int, int]:
    """(units with a decided output, units with every decided output right, units decided entirely)."""
    decided = correct = full = 0
    for unit in units:
        decisions = [(gate.decide(c.probability), c.label) for c in unit]
        made = [(d == 'pass') == label for d, label in decisions if d != 'defer']
        if made:
            decided += 1
            correct += all(made)
        full += len(made) == len(unit)
    return decided, correct, full


def certify_confidence_gate(
    cases: Iterable[GateCase],
    gate: ConfidenceGate,
    *,
    requirements: ConfidenceRequirements = DEFAULT_REQUIREMENTS,
    chosen_on: Collection[str] = (),
    unit: Unit = 'group',
) -> GateReport:
    """Measure `gate` on labelled `cases` and decide PASS, FAIL or UNVALIDATED.

    Args:
        cases: One or more records per case; a case's first record is its evidence.
        gate: The thresholds.
        requirements: The bars.
        chosen_on: Names of the cases (or groups) the thresholds were chosen on. A PASS on any
            of them, or on a case in the same group as one, is reported as UNVALIDATED: the
            accuracy of a choice, measured on the cases it was chosen to do well on, is optimistic.
        unit: `'group'` counts each group once (each case is its own group unless it has one);
            `'case'` counts every case, which is right only if the cases are independent.
    """
    first, records = _by_case(cases)
    if not first:
        raise ValueError('no cases to certify the gate on')
    decisions: dict[str, GateDecision] = {c.name: gate.decide(c.probability) for c in first}
    outputs_decided, outputs_correct, false_passes, false_fails = _tally(gate, first)
    n = len(first)
    deferred = n - outputs_decided
    units = _units(first, unit)
    m = len(units)
    decided, correct, fully_decided = _unit_tally(gate, units)

    deferred_cases = [c for c in first if decisions[c.name] == 'defer']
    with_fallback = [c for c in deferred_cases if c.fallback is not None]
    if with_fallback and len(with_fallback) < len(deferred_cases):
        missing = next(c.name for c in deferred_cases if c.fallback is None)
        raise ValueError(
            f'{len(deferred_cases) - len(with_fallback)} deferred cases have no fallback verdict (e.g. {missing!r}): '
            'end-to-end accuracy needs one for every deferred case'
        )
    # Defined when every deferred case has a fallback verdict (trivially, when none is deferred).
    has_fallback = len(with_fallback) == len(deferred_cases)
    end_correct: int | None = None
    if has_fallback:

        def right(c: GateCase) -> bool:
            d = decisions[c.name]
            return (c.fallback if d == 'defer' else d == 'pass') == c.label

        end_correct = sum(all(right(c) for c in u) for u in units)

    # Every check that can fail the gate shares one 2.5% upper tail, as in certify_judge.
    can_fail = 1 + (requirements.min_coverage > 0) + (requirements.min_end_to_end is not None)
    z_fail = NormalDist().inv_cdf(1 - 0.05 / can_fail / 2)
    minimum = requirements.min_trials
    checks = [
        _check('auto_accuracy', correct, decided, requirements.min_accuracy, minimum, z_fail=z_fail),
        _check('coverage', fully_decided, m, requirements.min_coverage, minimum, z_fail=z_fail),
    ]
    if requirements.min_end_to_end is not None:
        if end_correct is None:
            checks.append(
                Check(
                    'end_to_end',
                    'UNVALIDATED',
                    0,
                    0,
                    (0.0, 1.0),
                    requirements.min_end_to_end,
                    'no fallback verdicts on the deferred cases',
                )  # fmt: skip
            )
        else:
            checks.append(_check('end_to_end', end_correct, m, requirements.min_end_to_end, minimum, z_fail=z_fail))

    deciding = [c for c in checks if c.name != 'coverage' or requirements.min_coverage > 0]
    chosen = set(chosen_on)
    tainted = chosen | {c.unit_key for c in first if c.name in chosen}
    overlap = sum(c.name in tainted or c.unit_key in tainted for c in first)
    one_class = len({c.label for c in first}) < 2
    if any(c.status == 'FAIL' for c in deciding):
        status: CheckStatus = 'FAIL'
        reason = 'shown below the bar: ' + ', '.join(c.name for c in deciding if c.status == 'FAIL')
    elif all(c.status == 'PASS' for c in deciding):
        if overlap:
            status, reason = (
                'UNVALIDATED',
                f'thresholds chosen on {overlap} of these {n} cases: the accuracy is optimistic',
            )
        elif one_class:
            status, reason = 'UNVALIDATED', 'every label is the same: a constant judge would score as well'
        else:
            status, reason = 'PASS', 'every check clears its bar on the exact interval'
    else:
        status = 'UNVALIDATED'
        reason = 'not enough evidence: ' + ', '.join(c.name for c in deciding if c.status != 'PASS')

    repeated = [rs for rs in records.values() if len(rs) > 1]
    stability = (
        sum(len({gate.decide(r.probability) for r in rs}) == 1 for rs in repeated) / len(repeated) if repeated else None
    )
    return GateReport(
        gate, status, reason, tuple(checks), n, decided, deferred, correct, false_passes, false_fails,
        end_correct, len(repeated), stability, overlap, decisions, unit, m, fully_decided, outputs_decided,
        outputs_correct,
    )  # fmt: skip


def choose_gate(
    cases: Iterable[GateCase],
    *,
    min_accuracy: float,
    candidates: Sequence[ConfidenceGate] = DEFAULT_CANDIDATES,
    criterion: Criterion = 'lower_bound',
    min_trials: int = 10,
    unit: Unit = 'group',
) -> ConfidenceGate | None:
    """The candidate that decides the most cases while its accuracy on `cases` meets `min_accuracy`.

    `criterion='lower_bound'` asks the exact 95% lower bound to meet it; `'estimate'`, the point
    estimate, as a threshold sweep usually does. Either way, the accuracy of the choice measured on
    these same cases is optimistic: certify it on others (`calibrate_gate`). Ties go to the higher
    accuracy, then the narrower deferral band. None when no candidate qualifies.
    """
    first, _ = _by_case(cases)
    units = _units(first, unit)
    best: tuple[int, float, float] | None = None
    chosen: ConfidenceGate | None = None
    for gate in candidates:
        decided, correct, _ = _unit_tally(gate, units)
        if decided < min_trials:
            continue
        score = _cp(correct, decided)[0] if criterion == 'lower_bound' else correct / decided
        if score < min_accuracy:
            continue
        key = (decided, correct / decided, -(gate.high - gate.low))
        if best is None or key > best:
            best, chosen = key, gate
    return chosen


@dataclass(frozen=True)
class GateSelection:
    """Thresholds chosen and confirmed. `status` is the confirmed report's; `chosen` is None if none qualified."""

    method: SelectionMethod
    chosen: ConfidenceGate | None
    status: CheckStatus
    reason: str
    in_sample: GateReport | None
    """The choice measured on the cases it was chosen on: optimistic, shown for comparison only."""
    confirmed: GateReport | None
    """The report that decides: held-out cases (`split`), or every case after fixed-sequence testing."""
    selection_cases: tuple[str, ...] = ()
    confirmation_cases: tuple[str, ...] = ()
    tested: tuple[tuple[float, float, int, int, bool], ...] = ()
    """`fixed_sequence` only: (low, high, decided, correct, passed) for each gate tested, in order."""

    def to_dict(self) -> dict[str, Any]:
        return {
            'method': self.method,
            'chosen': None if self.chosen is None else {'low': self.chosen.low, 'high': self.chosen.high},
            'status': self.status,
            'reason': self.reason,
            'in_sample': None if self.in_sample is None else self.in_sample.to_dict(),
            'confirmed': None if self.confirmed is None else self.confirmed.to_dict(),
            'selection_cases': len(self.selection_cases),
            'confirmation_cases': len(self.confirmation_cases),
            'tested': [
                {'low': lo, 'high': hi, 'decided': d, 'correct': k, 'passed': ok} for lo, hi, d, k, ok in self.tested
            ],
        }


def _split(first: Sequence[GateCase], share: float, seed: int) -> tuple[set[str], set[str]]:
    """A random split of the case names that keeps each group whole.

    Stratified by the labels a group holds (only passes, only fails, or both), so both halves see
    both classes.
    """
    rng = random.Random(seed)
    groups: dict[str, list[GateCase]] = {}
    for case in first:
        groups.setdefault(case.unit_key, []).append(case)
    select: set[str] = set()
    confirm: set[str] = set()
    strata: dict[tuple[bool, ...], list[str]] = {}
    for key, members in groups.items():
        strata.setdefault(tuple(sorted({c.label for c in members})), []).append(key)
    for signature in sorted(strata):
        keys = sorted(strata[signature])
        rng.shuffle(keys)
        k = round(len(keys) * share)
        select.update(c.name for key in keys[:k] for c in groups[key])
        confirm.update(c.name for key in keys[k:] for c in groups[key])
    return select, confirm


def calibrate_gate(
    cases: Iterable[GateCase],
    *,
    requirements: ConfidenceRequirements = DEFAULT_REQUIREMENTS,
    candidates: Sequence[ConfidenceGate] = DEFAULT_CANDIDATES,
    method: SelectionMethod = 'split',
    selection_share: float = 0.5,
    seed: int = 0,
    unit: Unit = 'group',
) -> GateSelection:
    """Choose a gate's thresholds and certify the choice without the optimism of choosing on the same labels.

    Args:
        cases: Labelled cases, one or more records each; a case, or a group, is never on both sides
            of a split.
        requirements: The bars; `min_accuracy` is also what the choice must meet.
        candidates: The gates to choose among. `fixed_sequence` orders them itself, by how many
            cases each decides; it is valid for a nested family, such as `confidence_grid()`.
        method: `'split'` (choose on one part, certify on the rest) or `'fixed_sequence'`.
        selection_share: `split` only: the share of cases, per label, to choose on.
        seed: Seeds the split.
        unit: What the intervals count (`certify_confidence_gate`); the split keeps groups whole either way.
    """
    all_cases = list(cases)
    first, _ = _by_case(all_cases)
    if not first:
        raise ValueError('no cases to calibrate the gate on')
    if not candidates:
        raise ValueError('no candidate gates')
    bar = requirements.min_accuracy
    if method == 'split':
        if not 0 < selection_share < 1:
            raise ValueError(f'selection_share must be in (0, 1), got {selection_share}')
        select, confirm = _split(first, selection_share, seed)
        chosen = choose_gate(
            [c for c in all_cases if c.name in select],
            min_accuracy=bar,
            candidates=candidates,
            min_trials=requirements.min_trials,
            unit=unit,
        )
        names = (tuple(sorted(select)), tuple(sorted(confirm)))
        if chosen is None:
            return GateSelection(
                method, None, 'UNVALIDATED',
                f'no candidate shows accuracy >= {bar} on the {len(select)} selection cases', None, None, *names,
            )  # fmt: skip
        in_sample = certify_confidence_gate(
            [c for c in all_cases if c.name in select], chosen, requirements=requirements, chosen_on=select, unit=unit
        )
        confirmed = certify_confidence_gate(
            [c for c in all_cases if c.name in confirm], chosen, requirements=requirements, unit=unit
        )
        reason = f'chosen on {len(select)} cases, confirmed on {len(confirm)} others: {confirmed.reason}'
        return GateSelection(method, chosen, confirmed.status, reason, in_sample, confirmed, *names)

    # Fixed sequence: order by cases decided (the probabilities alone fix it), test each at the full level.
    units = _units(first, unit)
    ordered = sorted(candidates, key=lambda g: (_tally(g, first)[0], -(g.high - g.low)))
    tested: list[tuple[float, float, int, int, bool]] = []
    chosen = None
    for gate in ordered:
        decided, correct, _ = _unit_tally(gate, units)
        if decided < requirements.min_trials or _cp(decided, decided)[0] < bar:
            # Too few cases to pass even if all were right. Skipping it depends on how many cases
            # the gate decides, a function of the probabilities, never on the labels; testing it
            # would end the sequence before it starts.
            continue
        passed = _cp(correct, decided)[0] >= bar
        tested.append((gate.low, gate.high, decided, correct, passed))
        if not passed:
            break
        chosen = gate
    names = tuple(sorted(c.name for c in first))
    if chosen is None:
        return GateSelection(
            method, None, 'UNVALIDATED',
            f'the first gate tested is not shown to reach accuracy {bar}', None, None, names, names, tuple(tested),
        )  # fmt: skip
    report = certify_confidence_gate(all_cases, chosen, requirements=requirements, unit=unit)
    reason = f'fixed-sequence test over {len(tested)} gates on all {len(units)} {unit}s: {report.reason}'
    return GateSelection(method, chosen, report.status, reason, None, report, names, names, tuple(tested))


def gate_cases(
    probabilities: Mapping[str, float], labels: Mapping[str, bool], fallback: Mapping[str, bool] | None = None
) -> list[GateCase]:
    """`GateCase`s from three mappings keyed by case name; every probability needs a label."""
    missing = sorted(set(probabilities) - set(labels))
    if missing:
        raise ValueError(f'{len(missing)} cases have no label, e.g. {missing[0]!r}')
    fallback = fallback or {}
    return [GateCase(n, p, labels[n], fallback.get(n)) for n, p in probabilities.items()]
