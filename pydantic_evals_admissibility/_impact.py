"""Before replacing a judge, find which past release decisions it would have made differently.

`compare_judges` says whether one comparison survives a change of judge. A team switching judges
has a history of them: every release that was promoted, held back or rejected on the old judge's
scores. `decision_impact` re-decides each past comparison under both judges, from outcomes that
both judges gave on the same outputs, and lists the decisions that flip.

Not every flip matters alike. A version that was shipped (`taken` PROMOTE, or the old judge's
decision when nothing was recorded) and that the new judge would REJECT shipped a regression, as
far as the new judge can tell: that is the dangerous kind. Shipped and now INCONCLUSIVE means the
new judge cannot support the release, which is weaker. Held back and now PROMOTE is a missed gain.

Each record is decided on its own; the flips are counted, not tested. A count of flips among a
handful of past releases is a list to look at, not an estimate of how often the new judge disagrees.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Literal

from ._bridge import JudgeBridge, compare_judges
from ._certify import Certificate
from ._gate import Decision, GateRules

Outcomes = Mapping[str, Sequence[bool]]
Concern = Literal['dangerous', 'unsupported', 'missed', 'none']


@dataclass(frozen=True)
class ComparisonRecord:
    """One past comparison: both versions' outcomes on the same outputs, under the old and the new judge."""

    old: Mapping[str, Outcomes]
    """`{'baseline': outcomes, 'candidate': outcomes}` as the old judge scored them, as `compare_judges` takes."""
    new: Mapping[str, Outcomes]
    """The same outputs, scored by the new judge."""
    taken: Decision | None = None
    """The decision actually acted on, if recorded. It need not be what the gate said: a naive optimizer
    may have promoted on the raw score. Left out, the old judge's decision stands in for it."""
    old_certificate: Certificate | None = None
    new_certificate: Certificate | None = None


@dataclass(frozen=True)
class Impact:
    name: str
    old: Decision
    new: Decision
    taken: Decision | None
    gain_old: float | None
    gain_new: float | None
    concern: Concern
    bridge: JudgeBridge = field(repr=False)

    @property
    def flipped(self) -> bool:
        return self.old != self.new

    @property
    def direction(self) -> str:
        return f'{self.old}->{self.new}'

    @property
    def acted_on(self) -> Decision:
        return self.taken or self.old

    def to_dict(self) -> dict[str, object]:
        return {
            'old': self.old,
            'new': self.new,
            'flipped': self.flipped,
            'direction': self.direction,
            'taken': self.taken,
            'gain_old': self.gain_old,
            'gain_new': self.gain_new,
            'concern': self.concern,
            'interaction': self.bridge.interaction.estimate,
            'interaction_interval': self.bridge.interaction.interval,
            'comparable': self.bridge.verdict,
            'cases': self.bridge.cases,
        }


@dataclass(frozen=True)
class ImpactReport:
    records: dict[str, Impact]

    @property
    def flipped(self) -> list[str]:
        return [n for n, r in self.records.items() if r.flipped]

    @property
    def dangerous(self) -> list[str]:
        """Records acted on as PROMOTE that the new judge would REJECT."""
        return [n for n, r in self.records.items() if r.concern == 'dangerous']

    def counts(self) -> dict[str, int]:
        directions = Counter(r.direction for r in self.records.values() if r.flipped)
        concerns = Counter(r.concern for r in self.records.values())
        return {
            'records': len(self.records),
            'flipped': len(self.flipped),
            **{f'flip {d}': n for d, n in sorted(directions.items())},
            **{c: concerns[c] for c in ('dangerous', 'unsupported', 'missed')},
        }

    def table(self) -> str:
        lines = [f'{"record":<24}{"taken":>12}{"old":>14}{"new":>14}{"gain old":>10}{"gain new":>10}  concern']
        for name, r in self.records.items():
            lines.append(
                f'{name:<24}{r.taken or "-":>12}{r.old:>14}{r.new:>14}{_fmt(r.gain_old):>10}{_fmt(r.gain_new):>10}  '
                f'{"" if r.concern == "none" else r.concern}{" (flip)" if r.flipped else ""}'
            )
        c = self.counts()
        lines.append(
            f'{c["flipped"]} of {c["records"]} decisions flip; {c["dangerous"]} dangerous (acted on as PROMOTE, now '
            f'REJECT), {c["unsupported"]} unsupported, {c["missed"]} missed gains.'
        )
        return '\n'.join(lines)

    def to_dict(self) -> dict[str, object]:
        return {'counts': self.counts(), 'records': {n: r.to_dict() for n, r in self.records.items()}}


def _fmt(gain: float | None) -> str:
    return '-' if gain is None else f'{gain:+.3f}'


def _concern(acted_on: Decision, new: Decision) -> Concern:
    if acted_on == 'PROMOTE' and new == 'REJECT':
        return 'dangerous'
    if acted_on == 'PROMOTE' and new != 'PROMOTE':
        return 'unsupported'
    if acted_on != 'PROMOTE' and new == 'PROMOTE':
        return 'missed'
    return 'none'


def decision_impact(
    history: Mapping[str, ComparisonRecord], *, rules: GateRules | None = None, margin: float = 0.05
) -> ImpactReport:
    """Re-decide every past comparison under the old and the new judge and say which decisions flip.

    Args:
        history: Record name to the comparison, scored by both judges on the same outputs.
        rules: The gate's rules for both decisions; pass the ones the decisions were made with
            (`GateRules().for_candidates(k)` if a record was one of `k` candidates).
        margin: Passed to `compare_judges` for each record's comparability verdict.
    """
    if not history:
        raise ValueError('there are no past comparisons to re-decide')
    out: dict[str, Impact] = {}
    for name, record in history.items():
        bridge = compare_judges(
            record.old,
            record.new,
            margin=margin,
            rules=rules,
            old_certificate=record.old_certificate,
            new_certificate=record.new_certificate,
        )
        old, new = bridge.decision_old, bridge.decision_new
        out[name] = Impact(
            name,
            old.decision,
            new.decision,
            record.taken,
            old.mean_gain,
            new.mean_gain,
            _concern(record.taken or old.decision, new.decision),
            bridge,
        )
    return ImpactReport(out)
