"""Is this judge qualified for this decision? A certificate says what the judge showed; this says what it may do.

A certificate is evidence about one judge, gathered once. A harness or an optimizer about to act
on the judge's feedback has a narrower question: is that evidence about the judge it is running,
and is it enough for what it is about to do with the verdict? Labelling a dashboard with an
unvalidated number is a smaller act than promoting a prompt on it, or feeding the judge's reasons
back to an agent that will change its behaviour because of them. `qualify` applies one rule per
concern and returns every rule's outcome, so a log says why a judge was or was not used.

The rules, each a row in `Qualification.rules`:

- **coverage**: the certificate must be about this judge as configured now (`judge_identity`).
  A certificate for another configuration fails every decision: using it would label the
  numbers as certified when they are not. A certificate that recorded no identity cannot show
  what it covers: a warning for `report`, a failure for the rest.
- **verdict**: `report` accepts ADMISSIBLE, and UNVALIDATED with a warning (say so next to the
  numbers); INADMISSIBLE is evidence the judge is wrong, so not even a report. `gate`, `promote`
  and `steer` act on the verdict and need ADMISSIBLE.
- **checks**: the checks that decision leans on must each be present and PASS. `gate`, `promote`
  and `steer` need the judge to accept what is good and reject what is not (`acceptance` and
  `rejection`; `accuracy` and `order_consistency` for a comparison judge). A verdict computed by
  an older rule set, or a hand-built certificate, can say ADMISSIBLE without them.
- **slices** (`promote` only): an optimizer keeps whatever scores best, so a judge that is blind on
  one kind of case is exactly what it will exploit. A `slices` check that FAILs disqualifies; a
  certificate without one is a warning (`DESIGN.md`: two judges at 70/80 overall passed 0/5 on one
  kind of case).
- **context slice**: with `context={'slice': name}`, the traffic is of one kind. Acting on it
  needs the certificate to have measured that kind and not found it at or below the acceptance
  bar. A warning for `report`.
- **evidence** (`steer`, with `context={'sees_traces': True}`): a judge that will read an agent's
  traces and tell it what to do must have been shown to grade what happened, not what the agent
  claims. The certificate must have a `must_fail` evidence-control family (`EvidenceRewrite`)
  that PASSes, and the certified judge must have been shown its inputs. Measured
  (`bench/evidence_contract.py`): shown only the reply, a judge passed 8 of 24 false claims.
- **reasons** (`steer`): steering feeds the judge's reason back; a judge configured without one
  is a warning.
- **freshness**: with `max_age_days`, the certificate's `issued_at` (yours to pass: a
  `Certificate` does not record when it was made) must be known and recent enough.

Context keys other than `slice` and `sees_traces` are kept in the record and reported as not
checked, so a harness cannot believe a key was enforced when it was not.
"""

from __future__ import annotations

import re
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Literal

from ._certify import Certificate, Check
from ._diagnose import _family_results
from ._identity import differences, fingerprint, identity_reliable, judge_identity

Decision = Literal['report', 'gate', 'promote', 'steer']
Outcome = Literal['ok', 'warn', 'fail']
DECISIONS: tuple[Decision, ...] = ('report', 'gate', 'promote', 'steer')
_CHECKED_CONTEXT = ('slice', 'sees_traces')


class UnqualifiedJudge(AssertionError):
    """A judge is not qualified for the decision asked of it; the message lists every reason."""


@dataclass(frozen=True)
class Rule:
    """One rule's outcome: `ok`, `warn` (allowed, but say so) or `fail` (not qualified)."""

    name: str
    outcome: Outcome
    detail: str


@dataclass(frozen=True)
class Qualification:
    decision: Decision
    judge: str
    verdict: str
    rules: tuple[Rule, ...]
    fingerprint: str | None = None
    context: dict[str, Any] = field(default_factory=dict)
    issued_at: datetime | None = None
    checked_at: datetime | None = None

    @property
    def qualified(self) -> bool:
        return all(rule.outcome != 'fail' for rule in self.rules)

    @property
    def reasons(self) -> list[str]:
        """Why the judge is not qualified; empty when it is."""
        return [f'{r.name}: {r.detail}' for r in self.rules if r.outcome == 'fail']

    @property
    def warnings(self) -> list[str]:
        """What a qualified use must still disclose (an UNVALIDATED verdict on a report, say)."""
        return [f'{r.name}: {r.detail}' for r in self.rules if r.outcome == 'warn']

    def raise_unless_qualified(self) -> None:
        if not self.qualified:
            lines = [f'{self.judge} is not qualified to {self.decision}:'] + [f'- {r}' for r in self.reasons]
            raise UnqualifiedJudge('\n'.join(lines))

    def to_dict(self) -> dict[str, Any]:
        """Plain data for a log: the decision, the answer, and every rule's outcome."""
        return {
            'decision': self.decision,
            'qualified': self.qualified,
            'judge': self.judge,
            'verdict': self.verdict,
            'fingerprint': self.fingerprint,
            'reasons': self.reasons,
            'warnings': self.warnings,
            'rules': [{'name': r.name, 'outcome': r.outcome, 'detail': r.detail} for r in self.rules],
            'context': {
                k: v if isinstance(v, (str, int, float, bool, type(None))) else repr(v) for k, v in self.context.items()
            },
            'issued_at': self.issued_at.isoformat() if self.issued_at else None,
            'checked_at': self.checked_at.isoformat() if self.checked_at else None,
        }


def qualify(
    judge: Any,
    certificate: Certificate,
    *,
    decision: Decision,
    context: Mapping[str, Any] | None = None,
    max_age_days: float | None = None,
    issued_at: datetime | str | None = None,
    now: datetime | None = None,
    evidence_families: Collection[str] | None = None,
) -> Qualification:
    """Whether `judge`, with `certificate` as its evidence, may be used for `decision`.

    Args:
        judge: The judge as it will run (an evaluator), or its `judge_identity` as plain data,
            for a harness that logs configurations rather than holding the object.
        certificate: Its certificate, fresh or rebuilt with `Certificate.from_dict`.
        decision: `report` (label numbers), `gate` (accept or block a change), `promote` (an
            optimizer keeps a candidate) or `steer` (the judge's feedback goes back to the agent).
        context: What the use will look like: `slice` (the kind of case the traffic is) and
            `sees_traces` (the judge will read the agent's tool calls).
        max_age_days: Refuse a certificate older than this; needs `issued_at`.
        issued_at: When the certificate was made (a datetime or an ISO string).
        now: The time to measure age against; the current UTC time by default.
        evidence_families: Names of the `must_fail` families that change evidence rather than the
            answer. By default they are found in the saved judgments: a must-fail control whose
            output is the known-good answer, unchanged, can only have changed the evidence.
    """
    if decision not in DECISIONS:
        raise ValueError(f'decision must be one of {DECISIONS}, got {decision!r}')
    context = dict(context or {})
    when = datetime.fromisoformat(issued_at) if isinstance(issued_at, str) else issued_at
    now = now or datetime.now(timezone.utc)
    rules = [
        _coverage(judge, certificate, decision),
        _verdict(certificate, decision),
    ]
    if decision != 'report':
        rules.append(_checks(certificate))
    if decision == 'promote':
        rules.append(_slices(certificate))
    if 'slice' in context:
        rules.append(_context_slice(certificate, str(context['slice']), decision))
    if decision == 'steer':
        if context.get('sees_traces'):
            rules.append(_evidence(certificate, evidence_families))
        rules.append(_reasons(certificate))
    if max_age_days is not None:
        rules.append(_freshness(when, now, max_age_days))
    unchecked = sorted(k for k in context if k not in _CHECKED_CONTEXT)
    if unchecked:
        rules.append(Rule('context', 'warn', f'not checked by any rule: {", ".join(unchecked)}'))
    return Qualification(
        decision,
        certificate.judge,
        certificate.verdict,
        tuple(rules),
        certificate.fingerprint,
        context,
        when,
        now,
    )


def _coverage(judge: Any, certificate: Certificate, decision: Decision) -> Rule:
    if certificate.identity is None:
        outcome: Outcome = 'warn' if decision == 'report' else 'fail'
        return Rule(
            'coverage', outcome, 'the certificate records no judge identity, so it cannot show which judge it covers'
        )
    identity = dict(judge) if isinstance(judge, Mapping) else judge_identity(judge)
    changed = differences(certificate.identity, identity)
    if changed:
        return Rule('coverage', 'fail', f'the certificate is for another configuration: {", ".join(changed)} changed')
    if not (identity_reliable(certificate.identity) and identity_reliable(identity)):
        # Two judges can look alike here and behave differently (an unnamed function-backed model, a
        # function held as a setting): a matching identity is then no proof the certificate is about it.
        opaque = ', '.join(certificate.identity.get('opaque') or identity.get('opaque') or ['not fully identified'])
        # A warning, not a failure: certified just now from this very object, coverage holds by
        # construction. Reusing such a certificate elsewhere is refused by the cache, the pytest
        # plugin, juries and the report evaluator, which is where a look-alike judge could slip in.
        return Rule(
            'coverage',
            'warn',
            f'the judge cannot be identified reliably ({opaque}): use this certificate only for this object',
        )
    return Rule('coverage', 'ok', f'covers this judge ({fingerprint(identity)})')


def _verdict(certificate: Certificate, decision: Decision) -> Rule:
    if certificate.verdict == 'ADMISSIBLE':
        return Rule('verdict', 'ok', 'ADMISSIBLE')
    failing = ', '.join(f'{c.name} {c.status}' for c in certificate.checks if c.status != 'PASS')
    if decision == 'report' and certificate.verdict == 'UNVALIDATED':
        return Rule('verdict', 'warn', f'UNVALIDATED ({failing}): report the numbers as unvalidated')
    return Rule('verdict', 'fail', f'{certificate.verdict} ({failing}); {decision} needs ADMISSIBLE')


def _checks(certificate: Certificate) -> Rule:
    """The checks that show the judge discriminates: both directions, never one alone."""
    names = {c.name: c for c in certificate.checks}
    needed = ('accuracy', 'order_consistency') if 'accuracy' in names else ('acceptance', 'rejection')
    missing = [n for n in needed if n not in names]
    short = [f'{n} {names[n].status}' for n in needed if n in names and names[n].status != 'PASS']
    if missing or short:
        return Rule('checks', 'fail', '; '.join([f'missing {", ".join(missing)}'] * bool(missing) + short))
    return Rule('checks', 'ok', ' and '.join(needed) + ' PASS')


def _slices(certificate: Certificate) -> Rule:
    check = _find(certificate, 'slices')
    if check is None:
        return Rule('slices', 'warn', 'no slices: a kind of case the judge gets wrong can hide in the overall rate')
    if check.status == 'FAIL':
        return Rule('slices', 'fail', f'a kind of case is below the bar: {check.detail}')
    return Rule('slices', 'ok', f'slices {check.status}')


def _context_slice(certificate: Certificate, name: str, decision: Decision) -> Rule:
    soft: Outcome = 'warn' if decision == 'report' else 'fail'
    check = _find(certificate, 'slices')
    counts = _slice_counts(check.detail) if check else {}
    if name not in counts:
        return Rule('context_slice', soft, f'the certificate did not measure slice {name!r}')
    k, n = counts[name]
    assert check is not None
    if n == 0 or k <= check.threshold * n:
        return Rule('context_slice', soft, f'slice {name!r}: {k}/{n}, at or below the bar {check.threshold:.2f}')
    return Rule('context_slice', 'ok', f'slice {name!r}: {k}/{n}')


def _evidence(certificate: Certificate, names: Collection[str] | None) -> Rule:
    """The judge will read traces: it must have been shown to fail a true-looking answer when the evidence changed."""
    if certificate.identity is not None and certificate.identity.get('include_input') is False:
        return Rule('evidence', 'fail', 'the certified judge was not shown its inputs (include_input=False)')
    found = set(names) if names is not None else evidence_families_of(certificate)
    families = _family_statuses(_find(certificate, 'rejection'))
    evidence = {f: s for f, s in families.items() if f in found}
    if not evidence:
        return Rule('evidence', 'fail', 'no evidence controls (EvidenceRewrite, must_fail) in the certificate')
    short = [f'{f} {s}' for f, s in evidence.items() if s != 'PASS']
    if short:
        return Rule('evidence', 'fail', f'evidence controls not passed: {", ".join(short)}')
    return Rule('evidence', 'ok', f'evidence controls PASS: {", ".join(sorted(evidence))}')


def _reasons(certificate: Certificate) -> Rule:
    assertion = (certificate.identity or {}).get('assertion')
    if isinstance(assertion, Mapping) and assertion.get('include_reason') is False:
        return Rule('reasons', 'warn', 'the judge gives no reason: the agent is steered by a bare verdict')
    return Rule('reasons', 'ok', 'gives reasons' if isinstance(assertion, Mapping) else 'not recorded')


def _freshness(issued_at: datetime | None, now: datetime, max_age_days: float) -> Rule:
    if issued_at is None:
        return Rule('freshness', 'fail', f'max age {max_age_days:g} days, but when the certificate was made is unknown')
    age = (now - issued_at).total_seconds() / 86400
    if age > max_age_days:
        return Rule('freshness', 'fail', f'{age:.1f} days old, older than {max_age_days:g}')
    return Rule('freshness', 'ok', f'{age:.1f} days old')


def evidence_families_of(certificate: Certificate) -> set[str]:
    """`must_fail` families whose output is the case's known-good answer, unchanged: they changed the evidence."""
    good = {j.case: j.output for j in certificate.judgments if j.role == 'reference#0'}
    return {
        j.role.split(':', 1)[1]
        for j in certificate.judgments
        if j.role.startswith('must_fail:') and j.case in good and j.output == good[j.case]
    }


def _find(certificate: Certificate, name: str) -> Check | None:
    return next((c for c in certificate.checks if c.name == name), None)


def _family_statuses(check: Check | None) -> dict[str, str]:
    """Each family's status in a rejection or invariance check: `check.families`, or its detail if saved before."""
    if check is None:
        return {}
    return {f.name: f.status or 'PASS' for f in _family_results(check)}


def _slice_counts(detail: str) -> dict[str, tuple[int, int]]:
    """Each slice's count from a slices check's detail ('worst first: a 3/5, b 9/10; ...')."""
    body = detail.removeprefix('worst first: ').split('; ')[0]
    out = {}
    for part in body.split(', '):
        match = re.match(r'^(.+) (\d+)/(\d+)$', part.strip())
        if match:
            out[match.group(1)] = (int(match.group(2)), int(match.group(3)))
    return out
