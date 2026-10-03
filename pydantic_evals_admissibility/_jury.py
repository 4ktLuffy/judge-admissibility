"""A jury of judges, certified as one evaluator: several judges vote, and the vote is the verdict.

Verga et al. ("Replacing Judges with Juries", 2024, arXiv 2404.18796) found that a panel of
smaller judges from different model families agreed with people better than one large judge, at
a fraction of the cost. Whether that holds for a given rubric is a measurement, not an
assumption. A vote fixes a member's error only when the other members are right on that
judgment: members that share a blind spot pass it on to the jury, and a majority hides a member
that is wrong most of the time as long as the others outvote it.

So the jury is what gets certified, with the same `certify_judge` and checks as any judge, and
`compare_jury` puts its certificate beside each member's on the same plan, with the member
errors the vote fixed, the ones it inherited, and the ones it introduced.

The case is the unit of evidence here as everywhere in this package: a case's control variants
and labels are judged on the same question and are not independent. Accuracy is reported per
case, and the jury is compared with each member by `_gate.decide`, the paired case-level
sign-flip test, with the level split over the members rather than against the member that
happened to score best on the same data. Reused member certificates must cover their members
and be on the same evidence: the same judgments in the same order, and (`evidence_fingerprint`)
the same inputs, expected answers and metadata, in their order and with their types. A member
whose identity is not reliable (`identity_reliable`) cannot reuse a certificate at all: another
judge can share its identity, so it is certified afresh.

How a vote is counted:

- A member that raises, or returns no pass/fail assertion, has not voted. It is never counted as
  a pass or a fail.
- Fewer votes than `quorum` (by default a majority of the members) and the jury has no verdict:
  it raises `JuryUndecided`, which `certify_judge` records as an errored judgment.
- `majority`: passes when more votes pass than fail. A tie (possible when a member errored, or
  with an even panel) is decided by `tie`: fail by default, so a reply passes only when most of
  the votes cast say so; `'abstain'` gives no verdict instead.
- `any_fail`: one fail vote fails, whatever the others; otherwise passes on the votes cast.
- `unanimous_pass`: fails on one fail vote; passes only when every member voted pass. A member
  that errored blocks a pass, so the jury gives no verdict rather than pass on fewer votes. With
  every member voting it is the same rule as `any_fail`; the two differ only on errors.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import random
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from ._cache import _canonical, repr_fallback  # pyright: ignore[reportPrivateUsage]
from ._cases import HumanLabel, JudgeCase
from ._certify import (
    DEFAULT_THRESHOLDS,
    Certificate,
    Judgment,
    Thresholds,
    _describe,  # pyright: ignore[reportPrivateUsage]
    _plan,  # pyright: ignore[reportPrivateUsage]
    assertion_of,
    certify_judge,
    recertify,
)
from ._controls import DEFAULT_CONTROLS, Control
from ._gate import GateRules, decide
from ._identity import identity_reliable, judge_identity
from ._routing import truth_of
from ._stats import clopper_pearson, cohen_kappa, wilson

Rule = Literal['majority', 'unanimous_pass', 'any_fail']
Tie = Literal['fail', 'pass', 'abstain']
_TAG = re.compile(r'\[jury (majority|unanimous_pass|any_fail): ([PF?]+)\]')
_REASON_CHARS = 160


class JuryUndecided(Exception):
    """Too few members voted, or a tie with `tie='abstain'`: the jury gives no verdict.

    Raised rather than returned, so `certify_judge` records it as a judgment without a verdict,
    never as a pass or a fail. The message carries each member's vote.
    """


class _Panel(tuple[Evaluator[Any, Any, Any], ...]):
    """The members, recorded by `judge_identity` as each member's own identity.

    `judge_identity` reads a dataclass field through `model_dump()` when it has one. Without
    this, a member would be recorded by its `repr`, which for a judge on a `FunctionModel` holds
    a function's memory address: a different fingerprint in every process, for the same jury.
    """

    def model_dump(self) -> list[dict[str, Any]]:
        return [judge_identity(member) for member in self]


def jury_verdict(votes: Sequence[bool | None], rule: Rule, *, quorum: int, tie: Tie = 'fail') -> bool | None:
    """The jury's verdict from its members' votes (None: errored); None when it has no verdict."""
    cast = [v for v in votes if v is not None]
    passes = sum(cast)
    fails = len(cast) - passes
    if rule in ('any_fail', 'unanimous_pass') and fails:
        return False  # one fail decides, however few voted
    if len(cast) < quorum:
        return None
    if rule == 'any_fail':
        return True
    if rule == 'unanimous_pass':
        return True if len(cast) == len(votes) else None
    if passes != fails:
        return passes > fails
    return {'fail': False, 'pass': True, 'abstain': None}[tie]


def _code(votes: Sequence[bool | None]) -> str:
    return ''.join('?' if v is None else 'P' if v else 'F' for v in votes)


def votes_of(text: str | None) -> list[bool | None] | None:
    """Each member's vote, read back from a jury verdict's reason (or its `JuryUndecided` error)."""
    found = _TAG.search(text or '')
    if found is None:
        return None
    return [None if c == '?' else c == 'P' for c in found.group(2)]


def _reason(
    rule: Rule, votes: Sequence[bool | None], labels: Sequence[str], notes: Sequence[str | None], decided: bool | None
) -> str:
    """`[jury majority: PFP] PASS 2-1: m1 ...: PASS (why); ...`, so a certificate shows every vote."""
    cast = [v for v in votes if v is not None]
    outcome = 'NO VERDICT' if decided is None else 'PASS' if decided else 'FAIL'
    head = f'[jury {rule}: {_code(votes)}] {outcome} {sum(cast)}-{len(cast) - sum(cast)}'
    if len(cast) < len(votes):
        head += f' ({len(votes) - len(cast)} errored, not a vote)'
    parts = [
        f'{label}: {"ERROR" if v is None else "PASS" if v else "FAIL"}'
        + (f' ({" ".join(note.split())[:_REASON_CHARS]})' if note else '')
        for label, v, note in zip(labels, votes, notes, strict=True)
    ]
    return f'{head}: ' + '; '.join(parts)


@dataclass
class Jury(Evaluator[Any, Any, Any]):
    """Several judges vote on each output; the rule turns their votes into one pass/fail.

    Members are called concurrently for each judgment, so with `certify_judge(max_concurrency=c)`
    up to `c * len(members)` member calls are in flight. Each member is read with `assertion`
    (as in `certify_judge`). See the module docstring for how errors, quorum and ties are counted.
    """

    members: Sequence[Evaluator[Any, Any, Any]]
    rule: Rule = 'majority'
    assertion: str | None = None
    quorum: int | None = None
    """Fewest votes for a verdict; None: a majority of the members."""
    tie: Tie = 'fail'
    """What a tied `majority` vote decides."""

    def __post_init__(self) -> None:
        if not self.members:
            raise ValueError('a jury needs at least one member')
        if self.rule not in ('majority', 'unanimous_pass', 'any_fail'):
            raise ValueError(f'unknown rule {self.rule!r}')
        self.members = _Panel(self.members)
        if self.quorum is not None and not 1 <= self.quorum <= len(self.members):
            raise ValueError(f'quorum must be between 1 and {len(self.members)}, got {self.quorum}')

    @property
    def needed(self) -> int:
        return self.quorum if self.quorum is not None else len(self.members) // 2 + 1

    @property
    def labels(self) -> list[str]:
        """`m1 LLMJudge(model)`, ...: the members as the reasons and reports name them."""
        return [f'm{i} {_describe(member)}' for i, member in enumerate(self.members, 1)]

    def get_default_evaluation_name(self) -> str:
        return f'jury_{self.rule}'

    def decide(self, votes: Sequence[bool | None], notes: Sequence[str | None] | None = None) -> EvaluationReason:
        """The verdict on votes already cast, with every vote in the reason; raises `JuryUndecided`."""
        if len(votes) != len(self.members):
            raise ValueError(f'{len(votes)} votes for {len(self.members)} members')
        decided = jury_verdict(votes, self.rule, quorum=self.needed, tie=self.tie)
        reason = _reason(self.rule, votes, self.labels, notes or [None] * len(votes), decided)
        if decided is None:
            raise JuryUndecided(reason)
        return EvaluationReason(decided, reason)

    async def _vote(
        self, member: Evaluator[Any, Any, Any], ctx: EvaluatorContext[Any, Any, Any]
    ) -> tuple[bool | None, str | None]:
        try:
            return assertion_of(await member.evaluate_async(ctx), self.assertion)
        except Exception as error:  # an errored member has not voted
            return None, f'{type(error).__name__}: {error}'

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> EvaluationReason:
        results = await asyncio.gather(*(self._vote(member, ctx) for member in self.members))
        return self.decide([v for v, _ in results], [note for _, note in results])


Key = tuple[str, str, int]
Slot = tuple[str, str, int, str]
_ADDRESS = re.compile(r' at 0x[0-9a-f]+')


def _keyed(judgments: Sequence[Judgment]) -> dict[Key, Judgment]:
    """Judgments by (case, role, occurrence): the same slot of the same plan in two certificates."""
    seen: Counter[tuple[str, str]] = Counter()
    out: dict[Key, Judgment] = {}
    for j in judgments:
        out[(j.case, j.role, seen[(j.case, j.role)])] = j
        seen[(j.case, j.role)] += 1
    return out


def _saved(output: Any) -> str:
    return output if isinstance(output, str) else repr(output)


def _slots(judgments: Sequence[Judgment]) -> list[Slot]:
    """Each judgment's place in the plan, in order: case, role, occurrence and the output judged, as saved."""
    return [(case, role, n, _saved(j.output)) for (case, role, n), j in zip(_keyed(judgments), judgments, strict=True)]


def _first_difference(want: Sequence[Slot], got: Sequence[Slot]) -> str:
    for i, (a, b) in enumerate(zip(want, got, strict=False)):
        if a != b:
            what = 'the output judged differs' if a[:3] == b[:3] else f'{a[:3]} vs {b[:3]}'
            return f'judgment {i}: {what}'
    return f'{len(want)} judgments vs {len(got)}'


def _stable(value: Any) -> Any:
    """What a judge is shown, as JSON that keeps everything it can tell apart, and is the same in every process.

    Order- and type-preserving (`_cache._canonical`): `{'a': 1, 'b': 2}` is not `{'b': 2, 'a': 1}`,
    since `LLMJudge` writes the items into its prompt in their order, and `1` is not `1.0` or `True`.
    Only a set is sorted. A value with no faithful form is recorded by its `repr`, tagged with its class.
    """
    return _canonical(value, repr_fallback, 'evidence')


def evidence_plan(
    cases: Sequence[JudgeCase],
    *,
    controls: Sequence[Control] = DEFAULT_CONTROLS,
    human_labels: Sequence[HumanLabel] = (),
    repeats: int = 3,
    seed: int = 0,
    assertion: str | None = None,
) -> list[dict[str, Any]]:
    """Every judgment `certify_judge` makes with these arguments, in order, with everything the judge is shown.

    A certificate saves each judgment's case, role and output, but not the inputs, expected answer
    or metadata the judge saw, nor how its verdict was read. Two certificates with the same case
    names and outputs can rest on different evidence; this plan, and its `evidence_fingerprint`,
    record all of it.
    """
    seen: Counter[tuple[str, str]] = Counter()
    out: list[dict[str, Any]] = []
    for case, output, role in _plan(cases, controls, human_labels, repeats, random.Random(seed)):
        out.append(
            {
                'case': case.name,
                'role': role,
                'occurrence': seen[(case.name, role)],
                'output': _saved(output),
                'inputs': _stable(case.inputs),
                'expected_output': _stable(case.expected_output),
                'metadata': _stable(case.metadata),
                'assertion': assertion,
            }
        )
        seen[(case.name, role)] += 1
    return out


def _fingerprint(plan: Sequence[dict[str, Any]]) -> str:
    return hashlib.sha256(json.dumps(list(plan), sort_keys=True).encode()).hexdigest()[:16]


def evidence_fingerprint(
    cases: Sequence[JudgeCase],
    *,
    controls: Sequence[Control] = DEFAULT_CONTROLS,
    human_labels: Sequence[HumanLabel] = (),
    repeats: int = 3,
    seed: int = 0,
    assertion: str | None = None,
) -> str:
    """A short hash of `evidence_plan`: save it with a certificate, so it can be reused for a jury later."""
    return _fingerprint(
        evidence_plan(
            cases, controls=controls, human_labels=human_labels, repeats=repeats, seed=seed, assertion=assertion
        )
    )


def _unreliable(identity: Mapping[str, Any] | None) -> list[str]:
    """What an identity could not identify (its `opaque` items), or a placeholder when it is unreliable anyway."""
    if identity is None or identity_reliable(dict(identity)):
        return []
    return [str(item) for item in identity.get('opaque', ())] or ['an unnamed function-backed model']


def _check_member(i: int, member: Evaluator[Any, Any, Any], cert: Certificate) -> None:
    unknown = _unreliable(judge_identity(member)) or _unreliable(cert.identity)
    if unknown:
        raise ValueError(
            f'member {i} ({_describe(member)}) has an identity that does not hold everything that makes it the judge '
            f'it is ({"; ".join(unknown)}): another judge can share it, so no saved certificate can be shown to be '
            'about this member. Give its model a name of its own, or let compare_jury certify it (no certificate)'
        )
    if cert.identity is None:
        raise ValueError(
            f'certificate {i} records no judge identity, so nothing shows it is evidence about member {i} '
            f'({_describe(member)}): certify the member again'
        )
    if not cert.covers(member):
        raise ValueError(
            f'certificate {i} does not cover member {i} ({_describe(member)}): it is for another configuration '
            f'({", ".join(cert.differences(member))} differ)'
        )


def jury_certificate(
    jury: Jury,
    members: Sequence[Certificate],
    *,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    slices: Mapping[str, str] | None = None,
) -> Certificate:
    """The jury's certificate from its members' certificates on the same plan: no judge calls.

    The jury's verdict is a function of its members' verdicts, so each member judged once is
    enough: the jury's verdict on each judgment is the vote of the members' verdicts on that same
    judgment. Calling the members again inside the jury would pay for every judgment twice and
    compare the jury with different samples of its own members. The certificate's `calls` are
    the member calls the jury's verdicts rest on.

    Each certificate must record the identity of, and cover, the member in its place, and all of
    them must hold the same judgments in the same order (case, role, occurrence, output judged).
    A certificate does not record the inputs and expected answers its judge saw: `compare_jury`
    checks those against `evidence_fingerprint`. Sequential certificates keep their number of
    looks, so the jury's FAIL decisions spend the same widened budget.

    A member whose identity is not reliable (`identity_reliable`: an unnamed function-backed model,
    or state that cannot be identified) is refused: another judge can share its identity, so a
    certificate cannot be shown to be about it.
    """
    return _combine(jury, members, [True] * len(members), thresholds=thresholds, slices=slices)


def _combine(
    jury: Jury,
    members: Sequence[Certificate],
    saved: Sequence[bool],
    *,
    thresholds: Thresholds,
    slices: Mapping[str, str] | None,
) -> Certificate:
    """`jury_certificate`, checking that each `saved` certificate is about its member (fresh ones were made by it)."""
    if len(members) != len(jury.members):
        raise ValueError(f'{len(members)} certificates for {len(jury.members)} members')
    for i, (member, cert, check) in enumerate(zip(jury.members, members, saved, strict=True), 1):
        if check:
            _check_member(i, member, cert)
    plans = [_slots(cert.judgments) for cert in members]
    for i, plan in enumerate(plans[1:], 2):
        if plan != plans[0]:
            raise ValueError(
                f'member {i} was certified on a different plan ({_first_difference(plans[0], plan)}): '
                'same cases, controls, labels, repeats and seed, judged in the same order'
            )
    keyed = [_keyed(cert.judgments) for cert in members]
    judgments: list[Judgment] = []
    for key, first in keyed[0].items():
        slots = [k[key] for k in keyed]
        votes = [j.passed for j in slots]
        try:
            verdict = jury.decide(votes, [j.reason or j.error for j in slots])
        except JuryUndecided as undecided:
            judgments.append(Judgment(first.case, first.role, first.output, None, error=f'JuryUndecided: {undecided}'))
            continue
        judgments.append(Judgment(first.case, first.role, first.output, bool(verdict.value), verdict.reason))
    calls = sum(cert.calls if cert.calls is not None else len(cert.judgments) for cert in members)
    planned = sum(cert.planned if cert.planned is not None else len(cert.judgments) for cert in members)
    draft = Certificate(
        'UNVALIDATED',
        (),
        tuple(judgments),
        judge=f'Jury({jury.rule}: {", ".join(_describe(m) for m in jury.members)})',
        calls=calls,
        planned=planned,
        looks=max(cert.looks for cert in members),
        identity=judge_identity(jury),
    )
    return recertify(draft, thresholds=thresholds, slices=slices)


def _case_interval(per_case: Mapping[str, tuple[int, int]], resamples: int = 2000, seed: int = 0) -> list[float]:
    """A 95% percentile interval of the rate, resampling cases (each with all its judgments), not judgments."""
    counts = [c for c in per_case.values() if c[1]]
    if not counts:
        return [0.0, 1.0]
    rng = random.Random(seed)
    rates: list[float] = []
    for _ in range(resamples):
        drawn = [rng.choice(counts) for _ in counts]
        rates.append(sum(r for r, _ in drawn) / sum(n for _, n in drawn))
    rates.sort()
    return [rates[int(0.025 * resamples)], rates[min(resamples - 1, int(0.975 * resamples))]]


_MIN_RESAMPLED_CASES = 10
"""Fewer cases than this, and resampling them says little about the cases not seen: the exact interval is used."""


def _accuracy(per_case: Mapping[str, Sequence[bool]]) -> dict[str, Any]:
    """Right of judged, by case: the case is the unit, its control variants and labels are not independent.

    The interval resamples cases. When that cannot show the uncertainty, because there are fewer
    than `_MIN_RESAMPLED_CASES` cases or every case has the same rate (every resample is the same,
    and the interval is a point), it is an exact Clopper-Pearson interval with the case as the trial
    instead: `rate x cases` successes of `cases` trials, rounded outward (down for the lower bound,
    up for the upper), so three cases all right give [0.29, 1], not [1, 1]. `interval_method` says which.
    """
    counts = {case: (sum(r), len(r)) for case, r in per_case.items()}
    right, n = sum(r for r, _ in counts.values()), sum(k for _, k in counts.values())
    judged = [c for c in counts.values() if c[1]]
    all_right = sum(r == k for r, k in counts.values())
    interval = _case_interval(counts)
    same_rate = len({r / k for r, k in judged}) <= 1
    if judged and (len(judged) < _MIN_RESAMPLED_CASES or same_rate or interval[0] == interval[1]):
        m, rate = len(judged), right / n
        interval = [
            clopper_pearson(math.floor(rate * m + 1e-9), m)[0],
            clopper_pearson(math.ceil(rate * m - 1e-9), m)[1],
        ]
        why = 'every case has the same rate' if same_rate else f'fewer than {_MIN_RESAMPLED_CASES} cases'
        method = f'exact, the case as the trial (Clopper-Pearson): {why}, so resampling cases shows no uncertainty'
    else:
        method = 'cases resampled (95% percentile bootstrap)'
    return {
        'right': right,
        'judged': n,
        'rate': right / n if n else None,
        'interval': interval,
        'interval_method': method,
        'cases': len(counts),
        'cases_all_right': all_right,
        'cases_interval': list(wilson(all_right, len(counts))),
        'per_case': {case: [r, k] for case, (r, k) in counts.items()},
    }


@dataclass
class JuryComparison:
    """The jury's certificate beside its members', on the same plan, and what the vote did.

    Accuracy counts judgments whose right verdict the plan implies (`truth_of`: known-good and
    must-hold pass, must-fail fails, human labels as labelled; repeats are left out). The case is
    the unit of evidence: a case's control variants and labels are not independent judgments, so
    accuracy is reported per case with an interval that resamples cases, and the jury is compared
    with each member by `_gate.decide`, the paired case-level sign-flip test, with the level split
    over the members (each member is one comparison; none is picked as 'best' on the same data).
    Member votes are read from the jury's own verdicts, so `fixed`, `inherited` and agreement
    describe the votes the jury actually counted.
    """

    jury: Certificate
    members: dict[str, Certificate]
    rule: Rule
    reused: bool
    accuracy: dict[str, dict[str, Any]]
    """Per judge (members, then 'jury'): right of judged, per case, with an interval over cases
    (resampled, or exact when resampling cannot show the uncertainty; `interval_method` says which)."""
    effect: dict[str, dict[str, int]]
    """Per member: its errors the jury `fixed`, those it `inherited`, and the jury's errors on
    judgments the member got right (`introduced`)."""
    agreement: list[dict[str, Any]]
    """Pairwise between members: judgments both voted on, agreement, Cohen's kappa."""
    shared_errors: list[dict[str, Any]]
    """Judgments where more than one member was wrong: the blind spots a vote cannot fix."""
    against_members: dict[str, dict[str, Any]]
    """Per member: the jury against it, case by case (`_gate.decide`; the decision is the jury's as candidate)."""
    calls: dict[str, int]
    tokens: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    evidence: str | None = None
    """The `evidence_fingerprint` of the plan every certificate here was made on."""

    def table(self) -> str:
        judges = {**self.members, 'jury': self.jury}
        names = list(judges)
        width = 14
        rows = [f'{n.split()[0]} = {n}' for n in self.members]
        rows.append(f'{"":<22}' + ''.join(f'{n.split()[0]:>{width + 2}}' for n in names))
        rows.append(f'{"verdict":<22}' + ''.join(f'{judges[n].verdict:>{width + 2}}' for n in names))
        check_names = list(dict.fromkeys(c.name for cert in judges.values() for c in cert.checks))
        for check in check_names:
            cells = []
            for n in names:
                found = next((c for c in judges[n].checks if c.name == check), None)
                cells.append('-' if found is None else f'{found.successes}/{found.trials} {found.status[:4]}')
            rows.append(f'{check:<22}' + ''.join(f'{cell:>{width + 2}}' for cell in cells))
        rows.append(
            f'{"accuracy (truth)":<22}'
            + ''.join(f'{self.accuracy[n]["right"]}/{self.accuracy[n]["judged"]:<3}'.rjust(width + 2) for n in names)
        )
        rows.append(
            f'{"cases all right":<22}'
            + ''.join(
                f'{self.accuracy[n]["cases_all_right"]}/{self.accuracy[n]["cases"]:<3}'.rjust(width + 2) for n in names
            )
        )
        rows.append(f'{"calls":<22}' + ''.join(f'{self.calls.get(n, 0):>{width + 2}}' for n in names))
        if self.tokens:
            rows.append(f'{"tokens":<22}' + ''.join(f'{self.tokens.get(n, 0):>{width + 2}}' for n in names))
        for member, e in self.effect.items():
            rows.append(f'{member}: jury fixed {e["fixed"]}, inherited {e["inherited"]}, introduced {e["introduced"]}')
        for a in self.agreement:
            kappa = '-' if a['kappa'] is None else f'{a["kappa"]:.2f}'
            rows.append(f'agreement {a["a"]} ~ {a["b"]}: {a["agree"]}/{a["n"]}, kappa {kappa}')
        rows += [f'- {note}' for note in self.notes]
        return '\n'.join(rows)

    def to_dict(self, *, judgments: bool = True) -> dict[str, Any]:
        return {
            'rule': self.rule,
            'reused_member_verdicts': self.reused,
            'evidence_fingerprint': self.evidence,
            'jury': self.jury.to_dict(judgments=judgments),
            'members': {name: cert.to_dict(judgments=judgments) for name, cert in self.members.items()},
            'accuracy': self.accuracy,
            'effect': self.effect,
            'agreement': self.agreement,
            'shared_errors': self.shared_errors,
            'against_members': self.against_members,
            'calls': self.calls,
            'tokens': self.tokens,
            'notes': self.notes,
        }


def _against(
    member: Callable[[Key], bool | None],
    jury: Callable[[Key], bool | None],
    truth: Mapping[Key, bool],
    rules: GateRules,
) -> dict[str, Any]:
    """The jury against one member on the judgments both decided, compared case by case."""
    baseline: dict[str, list[bool]] = {}
    candidate: dict[str, list[bool]] = {}
    jury_only = member_only = 0
    for k, right in truth.items():
        mine, theirs = member(k), jury(k)
        if mine is None or theirs is None:
            continue
        baseline.setdefault(k[0], []).append(mine == right)
        candidate.setdefault(k[0], []).append(theirs == right)
        jury_only += theirs == right and mine != right
        member_only += mine == right and theirs != right
    out: dict[str, Any] = {
        'paired_judgments': sum(len(v) for v in baseline.values()),
        'cases': len(baseline),
        'jury_right_member_wrong': jury_only,
        'member_right_jury_wrong': member_only,
        'level': rules.level,
        'decision': None,
        'mean_gain': None,
        'p_better': None,
        'p_worse': None,
        'improved_cases': 0,
        'regressed_cases': 0,
    }
    if baseline:
        result = decide(baseline, candidate, rules=rules)
        out.update(
            decision=result.decision,
            mean_gain=result.mean_gain,
            p_better=result.p_better,
            p_worse=result.p_worse,
            improved_cases=result.improved,
            regressed_cases=result.regressed,
        )
    return out


def summarize_jury(
    jury: Jury,
    certificate: Certificate,
    members: Sequence[Certificate],
    *,
    reused: bool,
    tokens: Sequence[int | None] | None = None,
    evidence: str | None = None,
    rules: GateRules | None = None,
) -> JuryComparison:
    """What the vote did, read from the member votes recorded in the jury's verdicts.

    `rules` is the gate's for one comparison; it is split over the members (`for_candidates`),
    since the jury is compared with each of them on the same cases.
    """
    labels = jury.labels
    votes: dict[Key, list[bool | None]] = {}
    jury_passed: dict[Key, bool | None] = {}
    truth: dict[Key, bool] = {}
    for key, j in _keyed(certificate.judgments).items():
        recorded = votes_of(j.reason or j.error)
        if recorded is None or len(recorded) != len(labels):
            continue  # not a jury verdict (should not happen); left out
        votes[key], jury_passed[key] = recorded, j.passed
        right = truth_of(j.role)
        if right is not None:
            truth[key] = right

    def member_vote(i: int) -> Callable[[Key], bool | None]:
        return lambda k: votes[k][i]

    def jury_vote(k: Key) -> bool | None:
        return jury_passed[k]

    def by_case(vote: Callable[[Key], bool | None]) -> dict[str, list[bool]]:
        out: dict[str, list[bool]] = {}
        for k, right in truth.items():
            v = vote(k)
            if v is not None:
                out.setdefault(k[0], []).append(v == right)
        return out

    accuracy: dict[str, dict[str, Any]] = {label: _accuracy(by_case(member_vote(i))) for i, label in enumerate(labels)}
    accuracy['jury'] = _accuracy(by_case(jury_vote))
    accuracy['jury']['undecided'] = sum(jury_passed[k] is None for k in truth)

    effect: dict[str, dict[str, int]] = {}
    for i, label in enumerate(labels):
        fixed = inherited = introduced = 0
        for k in truth:
            mine, theirs = votes[k][i], jury_passed[k]
            if mine is None or theirs is None:
                continue
            if mine != truth[k]:
                fixed += theirs == truth[k]
                inherited += theirs != truth[k]
            elif theirs != truth[k]:
                introduced += 1
        effect[label] = {'fixed': fixed, 'inherited': inherited, 'introduced': introduced}

    agreement = []
    for a in range(len(labels)):
        for b in range(a + 1, len(labels)):
            pairs = [(bool(v[a]), bool(v[b])) for v in votes.values() if v[a] is not None and v[b] is not None]
            kappa = cohen_kappa(pairs)
            agreement.append(
                {
                    'a': labels[a],
                    'b': labels[b],
                    'n': len(pairs),
                    'agree': sum(x == y for x, y in pairs),
                    'kappa': kappa,
                }
            )

    shared = [
        {'case': k[0], 'role': k[1], 'votes': _code(votes[k]), 'truth': truth[k], 'jury': jury_passed[k]}
        for k in truth
        if sum(v is not None and v != truth[k] for v in votes[k]) > 1
    ]

    split = (rules or GateRules()).for_candidates(len(labels))
    against = {label: _against(member_vote(i), jury_vote, truth, split) for i, label in enumerate(labels)}

    member_calls = [cert.calls if cert.calls is not None else len(cert.judgments) for cert in members]
    calls = {label: n for label, n in zip(labels, member_calls, strict=True)}
    calls['jury'] = (certificate.calls or 0) if reused else (certificate.calls or 0) * len(labels)
    token_counts: dict[str, int] = {}
    if tokens is not None and all(t is not None for t in tokens):
        token_counts = {label: int(t or 0) for label, t in zip(labels, tokens, strict=True)}
        token_counts['jury'] = sum(token_counts.values()) if reused else 0

    member_errors: Counter[str] = Counter()
    jury_errors: Counter[str] = Counter()
    for k, right in truth.items():
        member_errors[k[0]] += sum(v is not None and v != right for v in votes[k])
        jury_errors[k[0]] += jury_passed[k] is not None and jury_passed[k] != right

    notes = _notes(certificate, members, labels, accuracy, effect, against, calls, token_counts, shared)
    if +member_errors or +jury_errors:
        cases = [c for c, _ in (member_errors + jury_errors).most_common()]
        notes.append(
            'errors by case, members (jury): '
            + ', '.join(f'{c} {member_errors[c]} ({jury_errors[c]})' for c in cases)
            + f'; {sum(member_errors.values())} member errors and {sum(jury_errors.values())} jury errors in all'
        )
    exact = [name for name, a in accuracy.items() if a['interval_method'].startswith('exact')]
    if exact:
        notes.append(
            f'accuracy interval of {", ".join(exact)}: exact with the case as the trial (Clopper-Pearson), since '
            'resampling so few or such uniform cases gives an interval narrower than the evidence'
        )
    if not reused:
        notes.append('jury calls and tokens: the jury called its members itself, so calls are judgments x members')
    return JuryComparison(
        certificate,
        dict(zip(labels, members, strict=True)),
        jury.rule,
        reused,
        accuracy,
        effect,
        agreement,
        shared,
        against,
        calls,
        token_counts,
        notes,
        evidence,
    )


def _notes(
    certificate: Certificate,
    members: Sequence[Certificate],
    labels: Sequence[str],
    accuracy: dict[str, dict[str, Any]],
    effect: dict[str, dict[str, int]],
    against: dict[str, dict[str, Any]],
    calls: dict[str, int],
    tokens: dict[str, int],
    shared: Sequence[dict[str, Any]],
) -> list[str]:
    """The findings in words: the comparison with each member, hidden members, blind spots."""
    notes: list[str] = []
    j = accuracy['jury']
    for label in labels:
        a, m = against[label], accuracy[label]
        b, c = a['jury_right_member_wrong'], a['member_right_jury_wrong']
        if a['decision'] is None:
            notes.append(f'jury vs {label}: no judgment both decided')
            continue
        if b == c == 0:
            verdict = 'the same on every paired judgment as'
        else:
            verdict = {'PROMOTE': 'better than', 'REJECT': 'worse than'}.get(a['decision'], 'not shown different from')
        notes.append(
            f'jury {j["right"]}/{j["judged"]} vs {label} {m["right"]}/{m["judged"]}: {verdict} it '
            f'(paired by case over {a["cases"]} cases: jury better on {a["improved_cases"]}, worse on '
            f'{a["regressed_cases"]}, sign-flip p(better)={a["p_better"]:.2f}, p(worse)={a["p_worse"]:.2f} at level '
            f'{a["level"]:.3f}, 0.05 split over {len(labels)} members; judgments only the jury got right {b}, '
            f'the reverse {c})'
        )
    ratios = [f'{calls["jury"] / calls[label]:.1f}x {label.split()[0]}' for label in labels if calls.get(label)]
    if ratios:
        line = 'cost in calls: ' + ', '.join(ratios)
        spent = [f'{tokens["jury"] / tokens[label]:.1f}x {label.split()[0]}' for label in labels if tokens.get(label)]
        if spent:
            line += '; in tokens: ' + ', '.join(spent)
        notes.append(line)
    for label, cert in zip(labels, members, strict=True):
        wrong = accuracy[label]['judged'] - accuracy[label]['right']
        if not wrong:
            continue
        e = effect[label]
        failing = [ch.name for ch in cert.checks if ch.status == 'FAIL']
        hidden = cert.verdict != 'ADMISSIBLE' and certificate.verdict == 'ADMISSIBLE'
        line = f'{label}: wrong on {wrong}/{accuracy[label]["judged"]}'
        line += (
            f' in {accuracy[label]["cases"] - accuracy[label]["cases_all_right"]} of {accuracy[label]["cases"]} cases'
        )
        line += f'; outvoted on {e["fixed"]}, carried on {e["inherited"]}'
        if hidden:
            line += f'. Its own certificate is {cert.verdict}' + (f' ({", ".join(failing)} FAIL)' if failing else '')
            line += ' and the jury is ADMISSIBLE: the majority hides it'
        notes.append(line)
    if shared:
        notes.append(f'{len(shared)} judgments with more than one member wrong: blind spots a vote cannot fix')
    return notes


async def compare_jury(
    jury: Jury,
    cases: Sequence[JudgeCase],
    *,
    controls: Sequence[Control] = DEFAULT_CONTROLS,
    human_labels: Sequence[HumanLabel] = (),
    repeats: int = 3,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    max_concurrency: int = 8,
    seed: int = 0,
    slice_by: Callable[[JudgeCase], str] | None = None,
    member_certificates: Sequence[Certificate | None] | None = None,
    member_plans: Sequence[str | None] | None = None,
    member_tokens: Sequence[int | None] | None = None,
    reuse: bool = True,
    rules: GateRules | None = None,
) -> JuryComparison:
    """Certify the jury and each member on the same plan, and report them side by side.

    Each member is certified with `certify_judge` (same cases, controls, labels, repeats and
    seed, so the plans match judgment for judgment). With `reuse=True` (the default) the jury's
    certificate is computed from those member verdicts (`jury_certificate`): no further calls,
    and a paired comparison. `reuse=False` certifies the `Jury` itself with `certify_judge`,
    which calls every member again on every judgment.

    Args:
        member_certificates: Certificates already made, one per member (None: certify it here),
            for example rebuilt with `Certificate.from_dict`. Each must record and cover its
            member as configured now, hold this plan's judgments in this plan's order, and not be
            sequential (`batch_size`: judged in another order, and it can stop early).
        member_plans: For each given certificate, the `evidence_fingerprint` recorded when it was
            made. Required: a certificate saves outputs, not the inputs, expected answers and
            metadata its judge saw, so only this shows it was made on these cases.
        member_tokens: Tokens each member's certification used, for the cost line.
        rules: The gate's rules for comparing the jury with one member; split over the members.
    """
    plan: dict[str, Any] = dict(
        controls=controls, human_labels=human_labels, repeats=repeats, thresholds=thresholds,
        assertion=jury.assertion, max_concurrency=max_concurrency, seed=seed, slice_by=slice_by,
    )  # fmt: skip
    given = list(member_certificates) if member_certificates is not None else [None] * len(jury.members)
    made_on = list(member_plans) if member_plans is not None else [None] * len(jury.members)
    if len(given) != len(jury.members) or len(made_on) != len(jury.members):
        raise ValueError(f'{len(given)} certificates and {len(made_on)} plans for {len(jury.members)} members')
    evidence = evidence_plan(
        cases, controls=controls, human_labels=human_labels, repeats=repeats, seed=seed, assertion=jury.assertion
    )
    fingerprint_now = _fingerprint(evidence)
    expected: list[Slot] = [(e['case'], e['role'], e['occurrence'], e['output']) for e in evidence]
    certs: list[Certificate] = []
    for i, (member, cert, recorded) in enumerate(zip(jury.members, given, made_on, strict=True), 1):
        if cert is None:
            cert = await certify_judge(member, cases, **plan)
        else:
            _check_member(i, member, cert)
            if cert.looks > 1:
                raise ValueError(
                    f'certificate {i} is sequential ({cert.looks} looks): its judgments are in another order and it '
                    'can stop early. Certify the member without batch_size, or use jury_certificate, which keeps looks'
                )
            got = _slots(cert.judgments)
            if got != expected:
                raise ValueError(
                    f'certificate {i} is not on this plan ({_first_difference(expected, got)}): '
                    'same cases, controls, labels, repeats and seed, judged in the same order'
                )
            if recorded is None:
                raise ValueError(
                    f'certificate {i} was given without the evidence it was made on: pass member_plans, the '
                    '`evidence_fingerprint` recorded when it was certified. A certificate saves outputs, not the '
                    'inputs, expected answers or metadata its judge saw'
                )
            if recorded != fingerprint_now:
                raise ValueError(
                    f'certificate {i} was made on other evidence: plan {recorded}, this plan {fingerprint_now} '
                    '(inputs, expected answers, metadata or the assertion read differ)'
                )
        certs.append(cert)
    slices = {case.name: slice_by(case) for case in cases} if slice_by else None
    if reuse:
        saved = [cert is not None for cert in given]
        certificate = _combine(jury, certs, saved, thresholds=thresholds, slices=slices)
    else:
        certificate = await certify_judge(jury, cases, **plan)
    return summarize_jury(
        jury, certificate, certs, reused=reuse, tokens=member_tokens, evidence=fingerprint_now, rules=rules
    )
