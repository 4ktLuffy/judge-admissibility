"""Count which cases an optimizer has already learned from, so they are not used to confirm its result.

A case whose judge feedback an optimizer read is no longer a test of the optimizer: a proposal
written to fix that case, or a candidate kept because it scored well on it, is fitted to it.
Results on it say how well the optimizer fitted, not how well the change generalizes. Each
exposure spends the case; `decide` on spent cases makes a false PROMOTE far more likely than its
level says. Measured in `bench/ledger_demo.py`, 1000 runs with no real gain, 40 cases: picking the
best of 5 candidates and confirming on the same cases promoted 6.3% of the time, best of 20 13.0%,
against a 2.5% budget; confirming the same picks on fresh cases, 1.2% and 2.2%.

`FeedbackLedger` records who saw which cases. `check_confirmation` raises `ExposedCases` when a
confirmation set contains spent cases; `confirm` is `decide` that answers REFUSED instead, the way
the gate refuses a judge whose certificate is not ADMISSIBLE. The ledger only knows what it is
told: a case the optimizer saw that nobody recorded is still spent.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ._certify import Certificate
from ._gate import GateResult, GateRules, decide


class ExposedCases(ValueError):
    """Confirmation was asked on cases an optimizer has already seen feedback on."""

    def __init__(self, exposed: Mapping[str, Sequence[str]]) -> None:
        self.exposed = dict(exposed)
        super().__init__(_explain(self.exposed))


def _explain(exposed: Mapping[str, Sequence[str]]) -> str:
    names = sorted(exposed)
    shown = ', '.join(f'{n!r} (by {", ".join(dict.fromkeys(exposed[n]))})' for n in names[:5])
    more = f' and {len(names) - 5} more' if len(names) > 5 else ''
    return (
        f'{len(names)} confirmation case(s) were already exposed to the optimizer: {shown}{more}; '
        'results on them do not confirm anything, confirm on fresh cases'
    )


@dataclass
class FeedbackLedger:
    events: list[tuple[str, tuple[str, ...]]] = field(default_factory=list)
    """In order, (who saw them, the cases seen)."""

    def record(self, case_names: Iterable[str], *, by: str) -> None:
        """Record that `by` (an optimizer round, a person tuning a prompt) saw feedback on these cases."""
        names = tuple(dict.fromkeys(case_names))
        if names:
            self.events.append((by, names))

    def exposures(self, case: str) -> int:
        return sum(case in names for _, names in self.events)

    def exposed_by(self, case: str) -> list[str]:
        return [by for by, names in self.events if case in names]

    def fresh(self, cases: Iterable[str], *, max_exposures: int = 0) -> list[str]:
        """The cases exposed at most `max_exposures` times, in the order given."""
        return [c for c in cases if self.exposures(c) <= max_exposures]

    def exposed(self, cases: Iterable[str], *, max_exposures: int = 0) -> dict[str, list[str]]:
        """The cases exposed more than `max_exposures` times, each with who saw it."""
        return {c: self.exposed_by(c) for c in cases if self.exposures(c) > max_exposures}

    def check_confirmation(self, case_names: Iterable[str], *, max_exposures: int = 0) -> None:
        """Raise `ExposedCases` if any confirmation case was exposed more than `max_exposures` times."""
        spent = self.exposed(case_names, max_exposures=max_exposures)
        if spent:
            raise ExposedCases(spent)

    def confirm(
        self,
        baseline: Mapping[str, Sequence[bool]],
        candidate: Mapping[str, Sequence[bool]],
        *,
        certificate: Certificate | None = None,
        rules: GateRules | None = None,
        judge: Any = None,
        max_exposures: int = 0,
    ) -> GateResult:
        """`decide`, REFUSED when any case in the comparison was exposed; the reason names those cases."""
        spent = self.exposed(sorted(set(baseline) | set(candidate)), max_exposures=max_exposures)
        if spent:
            return GateResult('REFUSED', _explain(spent))
        return decide(baseline, candidate, certificate=certificate, rules=rules, judge=judge)

    def to_dict(self) -> dict[str, Any]:
        return {'events': [{'by': by, 'cases': list(names)} for by, names in self.events]}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> FeedbackLedger:
        return cls([(e['by'], tuple(e['cases'])) for e in data['events']])
