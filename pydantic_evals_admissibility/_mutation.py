"""Mutation testing for an eval suite: which defects in the agent would the evaluators catch?

Mutation testing for code breaks the program on purpose and asks whether a test fails. Here the
agent's known-good outputs are broken on purpose (a mutant: the answer truncated, a number
changed, a yes turned into a no, a tool result changed under an unchanged reply), and every
evaluator is run on the original and on the mutant. Per evaluator and mutant, each case is one of:

- **caught**: the evaluator passed the original and failed the mutant.
- **missed**: it passed both.
- **unobservable**: the mutant changes something the evaluator is never shown (the inputs or tool
  trace, for an evaluator that reads only the output). Not called; counted as not caught, since
  a defect it cannot see is one it does not catch, but kept apart from missed because the fix is
  different: show it the evidence, not a better rubric.
- **original failed**: it already failed the original, so failing the mutant shows nothing.
- **error**: it raised, or returned no pass/fail, on either one. A verdict missing, not a catch.

A mutant is only a defect where something says the mutated output is wrong. With an `oracle`,
that is the oracle; a mutant it still accepts ("Answer: yes" with the explanation cut) is
reported as not a defect and never counted against an evaluator. Without one, every change is
taken as a defect except an output equal to the case's `expected_output`: that claim is yours,
and the report says no oracle was used.

Kill rates are reported with a Wilson interval, and `raise_unless_caught` decides on its lower
bound: 5 of 5 caught is not evidence the evaluator catches 80% of such defects.
"""

from __future__ import annotations

import asyncio
import dataclasses
import re
from collections import Counter
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any, Literal

from pydantic_evals.evaluators import Contains, Equals, EqualsExpected, Evaluator, IsInstance, LLMJudge, MaxDuration

from ._cases import JudgeCase
from ._certify import Judgment, _judge
from ._dataset import map_prose
from ._stats import wilson

Changes = Literal['output', 'evidence']
Status = Literal['caught', 'missed', 'unobservable', 'original_failed', 'error']
Oracle = Callable[[JudgeCase, Any], bool]
EvaluatorLike = Evaluator[Any, Any, Any] | Callable[[JudgeCase], Evaluator[Any, Any, Any]]

_OUTPUT_ONLY = (Equals, EqualsExpected, Contains, IsInstance, MaxDuration)
"""pydantic-evals evaluators that never read `ctx.inputs` or the span tree."""


class UncaughtDefects(AssertionError):
    """An evaluator does not catch a kind of defect often enough; the message says which cases."""


@dataclass(frozen=True)
class Mutant:
    """A known way to break the agent: `mutate(case)` returns the broken case, or None.

    `changes` says what the break touches, which decides who can see it: `'output'` (every
    evaluator is shown the output) or `'evidence'` (the inputs or tool trace, which an evaluator
    that reads only the output never sees). Return None for a case the mutant does not apply to;
    a mutant that returns the case unchanged is skipped the same way.
    """

    name: str
    mutate: Callable[[JudgeCase], JudgeCase | None] = field(repr=False)
    changes: Changes = 'output'

    @classmethod
    def of_output(cls, name: str, rewrite: Callable[[Any], Any | None]) -> Mutant:
        """A mutant that rewrites the output only: `rewrite(output)` returns the broken output, or None."""

        def mutate(case: JudgeCase) -> JudgeCase | None:
            changed = rewrite(case.output)
            return None if changed is None else replace(case, output=changed)

        return cls(name, mutate, 'output')

    @classmethod
    def of_evidence(cls, name: str, rewrite: Callable[[Any], Any | None]) -> Mutant:
        """A mutant that changes the inputs (a tool result, a record) and keeps the output as it was."""

        def mutate(case: JudgeCase) -> JudgeCase | None:
            changed = rewrite(case.inputs)
            return None if changed is None else replace(case, inputs=changed)

        return cls(name, mutate, 'evidence')


def _fields(output: Any) -> dict[str, Any] | None:
    """The fields of a dict, Pydantic model or dataclass, in order; None for anything else."""
    if isinstance(output, dict):
        return dict(output)  # pyright: ignore[reportUnknownArgumentType]
    if hasattr(output, 'model_dump') and hasattr(output, 'model_copy'):
        return {name: getattr(output, name) for name in output.model_dump()}
    if dataclasses.is_dataclass(output) and not isinstance(output, type):
        return {f.name: getattr(output, f.name) for f in dataclasses.fields(output)}
    return None


def _with(output: Any, **updates: Any) -> Any:
    if isinstance(output, dict):
        return {**output, **updates}
    if hasattr(output, 'model_copy'):
        return output.model_copy(update=updates)
    return dataclasses.replace(output, **updates)


def _last_field(output: Any, change: Callable[[Any], Any | None]) -> Any | None:
    """`output` with its last field that `change` applies to changed; None if none applies."""
    fields = _fields(output)
    if fields is None:
        return None
    for name, value in reversed(fields.items()):
        changed = change(value)
        if changed is not None:
            return _with(output, **{name: changed})
    return None


def _truncate(output: Any, keep: float = 0.5) -> Any | None:
    def cut(text: str) -> str:
        head = text[: int(len(text) * keep)]
        return head.rsplit(' ', 1)[0] if ' ' in head else head

    return map_prose(output, cut)


_SENTENCE_END = re.compile(r'(?<=[.!?])\s+|\n+')


def _drop_last_sentence(output: Any) -> Any | None:
    def drop(text: str) -> str | None:
        parts = [p for p in _SENTENCE_END.split(text.strip()) if p.strip()]
        if len(parts) < 2:
            return None
        end = text.rstrip().rfind(parts[-1])
        return text[:end].rstrip()

    if isinstance(output, str):
        return drop(output)
    return _last_field(output, lambda v: drop(v) if isinstance(v, str) else None)


_NUMBER = re.compile(r'\d+(?:\.\d+)?')


def _change_number(output: Any) -> Any | None:
    """The last number made one larger, keeping its decimals: `$42.50` becomes `$43.50`."""

    def bump(value: Any) -> Any | None:
        if isinstance(value, bool):
            return None
        if isinstance(value, int | float):
            return value + 1
        if isinstance(value, str):
            found = list(_NUMBER.finditer(value))
            if not found:
                return None
            last = found[-1]
            digits = last.group()
            decimals = len(digits.split('.')[1]) if '.' in digits else 0
            new = f'{float(digits) + 1:.{decimals}f}'
            return value[: last.start()] + new + value[last.end() :]
        return None

    return bump(output) if _fields(output) is None else _last_field(output, bump)


_YES_NO = re.compile(r'\b(yes|no)\b', re.I)


def _flip_yes_no(output: Any) -> Any | None:
    """The last yes/no in the text, or the last boolean field, turned around."""

    def flip(value: Any) -> Any | None:
        if isinstance(value, bool):
            return not value
        if isinstance(value, str):
            found = list(_YES_NO.finditer(value))
            if not found:
                return None
            last = found[-1]
            word = last.group()
            new = 'no' if word.lower() == 'yes' else 'yes'
            new = new.upper() if word.isupper() and len(word) > 1 else new.capitalize() if word[0].isupper() else new
            return value[: last.start()] + new + value[last.end() :]
        return None

    return flip(output) if _fields(output) is None else _last_field(output, flip)


def field_dropped(name: str) -> Mutant:
    """A required field left empty: removed from a dict, set to None on a model or dataclass."""

    def drop(output: Any) -> Any | None:
        fields = _fields(output)
        if fields is None or name not in fields or fields[name] is None:
            return None
        if isinstance(output, dict):
            return {k: v for k, v in output.items() if k != name}  # pyright: ignore[reportUnknownVariableType]
        return _with(output, **{name: None})

    return Mutant.of_output(f'field_dropped:{name}', drop)


TRUNCATED = Mutant.of_output('truncated', _truncate)
"""The second half of the text, or of every prose field, cut off at a word boundary."""
LAST_SENTENCE_DROPPED = Mutant.of_output('last_sentence_dropped', _drop_last_sentence)
"""The last sentence or line gone: often the conclusion, or the `Answer:` line."""
NUMBER_CHANGED = Mutant.of_output('number_changed', _change_number)
"""The last number one larger: a wrong amount that still looks like an amount."""
YES_NO_FLIPPED = Mutant.of_output('yes_no_flipped', _flip_yes_no)
"""The last yes/no, or the last boolean field, the other way round."""
EMPTIED = Mutant.of_output('emptied', lambda output: map_prose(output, lambda text: ''))
"""Nothing left to read: the text, or every prose field, empty."""

DEFAULT_MUTANTS: tuple[Mutant, ...] = (TRUNCATED, LAST_SENTENCE_DROPPED, NUMBER_CHANGED, YES_NO_FLIPPED, EMPTIED)


def observed_by(evaluator: Evaluator[Any, Any, Any]) -> frozenset[Changes]:
    """What an evaluator is shown, by a fixed rule: the rule is the claim, so it is spelled out.

    - `LLMJudge`: the output, and the evidence only with `include_input=True`.
    - `Equals`, `EqualsExpected`, `Contains`, `IsInstance`, `MaxDuration`: the output only.
    - Anything else: both. A custom evaluator is handed `ctx.inputs`, and whether it reads them
      cannot be seen from outside, so an evidence defect it passes is reported as missed, not
      excused. Pass `observes=` to `mutation_report` to say otherwise.
    """
    if isinstance(evaluator, LLMJudge):
        return frozenset(('output', 'evidence')) if evaluator.include_input else frozenset(('output',))
    if isinstance(evaluator, _OUTPUT_ONLY):
        return frozenset(('output',))
    return frozenset(('output', 'evidence'))


@dataclass(frozen=True)
class MutationOutcome:
    """One evaluator on one mutated case."""

    evaluator: str
    mutant: str
    case: str
    status: Status
    output: Any
    reason: str | None = None
    error: str | None = None


@dataclass(frozen=True)
class KillRate:
    """Defects caught out of defects judged.

    Unobservable defects count as judged and not caught. Errors, and cases whose original already
    failed, are left out of the rate and counted. For one mutant each case is one defect. Pooled
    over every mutant, several defects share a case and are not independent, so that interval is
    narrower than it should be: decide per mutant.
    """

    caught: int
    missed: int
    unobservable: int
    errors: int
    original_failed: int

    @property
    def judged(self) -> int:
        return self.caught + self.missed + self.unobservable

    @property
    def rate(self) -> float | None:
        return self.caught / self.judged if self.judged else None

    @property
    def interval(self) -> tuple[float, float]:
        return wilson(self.caught, self.judged)


@dataclass(frozen=True)
class MutantSummary:
    """How a mutant applied: defects judged, and the cases it was skipped on and why."""

    name: str
    changes: Changes
    defects: int
    not_defect: int
    not_applicable: int


@dataclass(frozen=True)
class MutationReport:
    evaluators: tuple[str, ...]
    mutants: tuple[MutantSummary, ...]
    outcomes: tuple[MutationOutcome, ...] = field(repr=False)
    originals: Mapping[str, tuple[int, int, int]] = field(repr=False)
    """Per evaluator: originals passed, originals judged, originals errored."""
    oracle: bool = False
    """Whether an oracle decided which mutants are defects; without one, every change was taken as one."""
    bad_originals: tuple[str, ...] = ()
    """Cases whose original output the oracle says is wrong: no mutant was made from them."""

    def _select(self, evaluator: str, mutant: str | None) -> list[MutationOutcome]:
        if evaluator not in self.evaluators:
            raise KeyError(f'no evaluator named {evaluator!r}; the report has {list(self.evaluators)}')
        if mutant is not None and mutant not in {m.name for m in self.mutants}:
            raise KeyError(f'no mutant named {mutant!r}; the report has {[m.name for m in self.mutants]}')
        return [o for o in self.outcomes if o.evaluator == evaluator and (mutant is None or o.mutant == mutant)]

    def kill_rate(self, evaluator: str, mutant: str | None = None) -> KillRate:
        """Defects `evaluator` caught, over every mutant or one."""
        counts = Counter(o.status for o in self._select(evaluator, mutant))
        return KillRate(
            counts['caught'], counts['missed'], counts['unobservable'], counts['error'], counts['original_failed']
        )

    def missed(self, evaluator: str, mutant: str | None = None) -> list[MutationOutcome]:
        """The mutated cases `evaluator` passed: the defects that would reach users unnoticed."""
        return [o for o in self._select(evaluator, mutant) if o.status == 'missed']

    def raise_unless_caught(self, evaluator: str, mutant: str | None = None, *, min_rate: float = 0.8) -> None:
        """Raise `UncaughtDefects` unless the kill rate's Wilson lower bound is at least `min_rate`.

        For a test: `report.raise_unless_caught('correctness judge', 'number_changed')`. Also raises
        when no defect was judged, or when more than a tenth of verdicts errored, since then the
        rate is not about the evaluator. 10 of 10 caught clears 0.7, not 0.8 (Wilson lower bound 0.72).
        """
        rate = self.kill_rate(evaluator, mutant)
        what = f'{evaluator} on {mutant or "every mutant"}'
        problems: list[str] = []
        low, high = rate.interval
        if rate.judged == 0:
            problems.append(f'{what}: no defect was judged')
        elif rate.errors > 0.1 * (rate.judged + rate.errors):
            problems.append(f'{what}: {rate.errors} of {rate.judged + rate.errors} verdicts errored')
        elif low < min_rate:
            problems.append(
                f'{what}: caught {rate.caught}/{rate.judged} (95% interval [{low:.2f}, {high:.2f}]); '
                f'needs a lower bound >= {min_rate:.2f}'
            )
        if not problems:
            return
        if rate.unobservable:
            problems.append(
                f'{rate.unobservable} defects are in what it is never shown (the inputs or trace): show it them, '
                'for example LLMJudge(include_input=True), or pass observes= if it does read them'
            )
        if rate.original_failed:
            problems.append(f'{rate.original_failed} cases left out: it already failed the original')
        for outcome in self.missed(evaluator, mutant)[:5]:
            shown = outcome.output if isinstance(outcome.output, str) else repr(outcome.output)
            problems.append(f'missed {outcome.mutant} on {outcome.case}: {shown[:120]!r}')
        raise UncaughtDefects('\n'.join(problems))

    def table(self) -> str:
        """Evaluators by mutants: caught/judged per cell, then the overall kill rate and its interval.

        A cell reads `blind` when the evaluator is never shown what the mutant changes, and carries
        `eN` for N errors and `oN` for N cases left out because the original already failed.
        """
        rates = {(e, m.name): self.kill_rate(e, m.name) for e in self.evaluators for m in self.mutants}

        def cell(r: KillRate) -> str:
            text = 'blind' if r.judged and r.unobservable == r.judged else f'{r.caught}/{r.judged}' if r.judged else '-'
            return (
                text + (f' e{r.errors}' if r.errors else '') + (f' o{r.original_failed}' if r.original_failed else '')
            )

        widths = {
            m.name: max(len(m.name), len(m.changes), *(len(cell(rates[e, m.name])) for e in self.evaluators))
            for m in self.mutants
        }
        first = max(12, *(len(e) for e in self.evaluators))

        def row(label: str, originals: str, values: Sequence[str]) -> str:
            cells = '  '.join(f'{v:>{widths[m.name]}}' for m, v in zip(self.mutants, values, strict=True))
            return f'{label:<{first}}  {originals:>9}  {cells}'

        rows = [row('evaluator', 'originals', [m.name for m in self.mutants]) + f'  {"kill rate":>9}  95% interval']
        for evaluator in self.evaluators:
            passed, judged, errored = self.originals[evaluator]
            total = self.kill_rate(evaluator)
            low, high = total.interval
            kill = '-' if total.rate is None else f'{total.rate:.2f}'
            originals = f'{passed}/{judged}' + (f' e{errored}' if errored else '')
            cells = [cell(rates[evaluator, m.name]) for m in self.mutants]
            rows.append(row(evaluator, originals, cells) + f'  {kill:>9}  [{low:.2f}, {high:.2f}]')
        rows.append('')
        for label, attr in (('defects', 'defects'), ('not a defect', 'not_defect'), ('n/a', 'not_applicable')):
            rows.append(row(label, '', [str(getattr(m, attr)) for m in self.mutants]))
        rows.append(row('changes', '', [m.changes for m in self.mutants]))
        if not self.oracle:
            rows.append('no oracle: every mutant that changed a case was taken as a defect')
        if self.bad_originals:
            rows.append(f'{len(self.bad_originals)} cases skipped: the oracle rejects their original output')
        return '\n'.join(rows)

    def to_dict(self, *, outcomes: bool = True) -> dict[str, Any]:
        def rate(r: KillRate) -> dict[str, Any]:
            return {**dataclasses.asdict(r), 'judged': r.judged, 'rate': r.rate, 'interval': list(r.interval)}

        out: dict[str, Any] = {
            'oracle': self.oracle,
            'bad_originals': list(self.bad_originals),
            'mutants': [dataclasses.asdict(m) for m in self.mutants],
            'evaluators': {
                e: {
                    'originals': dict(zip(('passed', 'judged', 'errors'), self.originals[e], strict=True)),
                    'overall': rate(self.kill_rate(e)),
                    'by_mutant': {m.name: rate(self.kill_rate(e, m.name)) for m in self.mutants},
                }
                for e in self.evaluators
            },
        }
        if outcomes:
            out['outcomes'] = [
                {**dataclasses.asdict(o), 'output': o.output if isinstance(o.output, str) else repr(o.output)}
                for o in self.outcomes
            ]
        return out


def _named(evaluators: Mapping[str, EvaluatorLike] | Sequence[EvaluatorLike]) -> dict[str, EvaluatorLike]:
    if isinstance(evaluators, Mapping):
        return dict(evaluators)
    out: dict[str, EvaluatorLike] = {}
    for evaluator in evaluators:
        if isinstance(evaluator, Evaluator):
            base = evaluator.get_default_evaluation_name()
        else:
            base = getattr(evaluator, '__name__', 'evaluator')
        name, n = base, 1
        while name in out:
            n += 1
            name = f'{base}#{n}'
        out[name] = evaluator
    return out


def _unchanged(case: JudgeCase, mutated: JudgeCase) -> bool:
    try:
        return bool(mutated.output == case.output and mutated.inputs == case.inputs)
    except Exception:  # an output type whose == raises or is ambiguous: treat it as changed
        return False


async def mutation_report(
    evaluators: Mapping[str, EvaluatorLike] | Sequence[EvaluatorLike],
    cases: Sequence[JudgeCase],
    mutants: Sequence[Mutant] = DEFAULT_MUTANTS,
    *,
    oracle: Oracle | None = None,
    observes: Mapping[str, Collection[Changes]] | None = None,
    assertion: str | None = None,
    max_concurrency: int = 8,
) -> MutationReport:
    """Run every evaluator on known-good outputs and on mutants of them, and report what it caught.

    Args:
        evaluators: pydantic-evals evaluators with a pass/fail assertion (`LLMJudge`, `Contains`,
            `IsInstance`, your own), a list or a mapping of names to them; for a dataset,
            `dataset.evaluators`. A callable `case -> Evaluator` builds one per case, for checks
            that are attached to single cases (`Contains(value=<that case's answer>)`); it is built
            from the original case and used on its mutants.
        cases: Known-good outputs. With an `oracle`, any the oracle rejects are skipped.
        mutants: The defects to plant.
        oracle: `oracle(case, output)` is True when `output` is right for `case`. Decides which
            mutants are defects; mutants it still accepts are reported as not a defect.
        observes: Evaluator name to what it is shown (`'output'`, `'evidence'`), overriding the
            rule in `observed_by()`.
        assertion: The result to read from evaluators that return several booleans.
        max_concurrency: Evaluations in flight at once.
    """
    if max_concurrency < 1:
        raise ValueError('max_concurrency must be >= 1')
    if not cases:
        raise ValueError('no cases to mutate')
    names = [m.name for m in mutants]
    if len(set(names)) != len(names):
        raise ValueError(f'mutant names must be unique, got {names}')
    named = _named(evaluators)
    if observes and (unknown := set(observes) - set(named)):
        raise KeyError(f'observes= names evaluators the report does not have: {sorted(unknown)}')
    limit = asyncio.Semaphore(max_concurrency)

    bad = tuple(c.name for c in cases if oracle is not None and not oracle(c, c.output))
    usable = [c for c in cases if c.name not in bad]
    built = {(e, c.name): ev if isinstance(ev, Evaluator) else ev(c) for e, ev in named.items() for c in usable}

    summaries: list[MutantSummary] = []
    defects: list[tuple[Mutant, JudgeCase, JudgeCase]] = []
    for mutant in mutants:
        found = not_defect = not_applicable = 0
        for case in usable:
            mutated = mutant.mutate(case)
            if mutated is None or _unchanged(case, mutated):
                not_applicable += 1
                continue
            if oracle is not None:
                wrong = not oracle(mutated, mutated.output)
            else:
                wrong = case.expected_output is None or mutated.output != case.expected_output
            if not wrong:
                not_defect += 1
                continue
            found += 1
            defects.append((mutant, case, mutated))
        summaries.append(MutantSummary(mutant.name, mutant.changes, found, not_defect, not_applicable))

    def sees(evaluator: str, case: str) -> frozenset[Changes]:
        if observes and evaluator in observes:
            return frozenset(observes[evaluator])
        return observed_by(built[evaluator, case])

    originals = await asyncio.gather(
        *(_judge(built[e, c.name], c, c.output, 'original', assertion, limit) for e in named for c in usable)
    )
    by_original = {(e, j.case): j for (e, _), j in zip(((e, c) for e in named for c in usable), originals, strict=True)}
    planned = [
        (e, mutant, case, mutated)
        for e in named
        for mutant, case, mutated in defects
        # A mutant is judged only against an original the evaluator passed; anything else shows nothing.
        if mutant.changes in sees(e, case.name) and by_original[e, case.name].passed
    ]
    verdicts = await asyncio.gather(
        *(
            _judge(built[e, case.name], mutated, mutated.output, mutant.name, assertion, limit)
            for e, mutant, case, mutated in planned
        )
    )
    on_mutant: dict[tuple[str, str, str], Judgment] = {
        (e, mutant.name, case.name): j for (e, mutant, case, _), j in zip(planned, verdicts, strict=True)
    }

    outcomes: list[MutationOutcome] = []
    for e in named:
        for mutant, case, mutated in defects:
            original = by_original[e, case.name]
            judged = on_mutant.get((e, mutant.name, case.name))
            status: Status
            if original.passed is None:
                status = 'error'
            elif not original.passed:
                status = 'original_failed'
            elif judged is None:
                status = 'unobservable'
            elif judged.passed is None:
                status = 'error'
            else:
                status = 'missed' if judged.passed else 'caught'
            reason, error = (judged.reason, judged.error) if judged else (None, None)
            outcomes.append(
                MutationOutcome(e, mutant.name, case.name, status, mutated.output, reason, original.error or error)
            )
    tallies = {
        e: (
            sum(by_original[e, c.name].passed is True for c in usable),
            sum(by_original[e, c.name].passed is not None for c in usable),
            sum(by_original[e, c.name].passed is None for c in usable),
        )
        for e in named
    }
    return MutationReport(tuple(named), tuple(summaries), tuple(outcomes), tallies, oracle is not None, bad)
