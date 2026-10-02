"""Improve a judge from reviewed disagreements, and keep the change only if it holds up on cases it never saw.

A reviewer who overrules a judge has said something the rubric did not: what the rubric means on
that kind of answer. Writing that back into the rubric (a clarification, or the reviewed cases as
examples) is the obvious fix, and also the obvious way to fool yourself. A revision written from
a few disagreements will agree with the reviewer on those disagreements by construction; it says
nothing about the next case. And agreement can rise for the wrong reason: reviewed sets are
mostly answers the reviewer passed, so a revision that teaches the judge to pass more agrees more
often while losing the ability to fail anything.

`apprentice` does the fix under the package's own discipline:

- The reviewed cases are split by case (never by reply: two replies to one question share its
  difficulty) into a training set and a held-out set. Only training disagreements reach `propose`,
  and the `FeedbackLedger` records them, so the held-out cases are fresh by construction and the
  gate refuses if they were seen in an earlier round.
- Both judges are certified on the held-out cases, with controls and the reviewers' verdicts as
  labels. The same calls give each judge's agreement with the reviewer, reply by reply.
- Agreement before and after is compared case by case with the paired gate (`decide`, through
  `ledger.confirm`). The revision is promoted only on PROMOTE, and not if its certificate is
  INADMISSIBLE: a judge that agrees more because it passes everything fails its rejection controls.

The proposer is injected (`propose`), so tests are scripted and any model can propose.
"""

from __future__ import annotations

import inspect
import random
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from pydantic_evals.evaluators import Evaluator

from ._cases import HumanLabel, JudgeCase
from ._certify import DEFAULT_THRESHOLDS, Certificate, Thresholds, certify_judge
from ._controls import DEFAULT_CONTROLS, Control
from ._gate import GateResult, GateRules
from ._ledger import ExposedCases, FeedbackLedger
from ._stats import wilson


@dataclass(frozen=True)
class ReviewedCase:
    """One output a person reviewed, and, when known, what the judge said about it.

    `case` groups replies to the same question: the split is by case. `judge_passed` is the
    verdict the reviewer saw and overruled or upheld; training disagreements are the reviewed
    outputs where it differs from `reviewer_passed`. `expected_output` is passed to the judge's
    context and lets `MismatchedOutput` choose a donor with a different answer.
    """

    case: str
    inputs: Any
    output: Any
    reviewer_passed: bool
    judge_passed: bool | None = None
    judge_reason: str | None = None
    note: str | None = None
    """What the reviewer said, if anything: the most useful line a proposer can be shown."""
    expected_output: Any = None
    metadata: Any = None

    @property
    def disagrees(self) -> bool:
        return self.judge_passed is not None and self.judge_passed != self.reviewer_passed


@dataclass(frozen=True)
class Proposal:
    """A revision of the rubric: a clarification, and reviewed examples to show the judge."""

    clarification: str
    examples: tuple[ReviewedCase, ...] = ()


ProposeResult = Proposal | str
Propose = Callable[[str, Sequence[ReviewedCase]], ProposeResult | Awaitable[ProposeResult]]
"""`propose(rubric, training_disagreements)`: a `Proposal`, or just the clarification text."""


def _clip(value: Any, limit: int = 600) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= limit else text[: limit - 3] + '...'


def revised_rubric(rubric: str, proposal: Proposal) -> str:
    """The rubric with the clarification and the examples appended; the original wording is kept."""
    parts = [rubric.rstrip()]
    if proposal.clarification.strip():
        parts.append(f'Clarification: {proposal.clarification.strip()}')
    if proposal.examples:
        lines = ["Reviewed examples (a reviewer's verdict; follow it on outputs like these):"]
        for example in proposal.examples:
            verdict = 'PASS' if example.reviewer_passed else 'FAIL'
            note = f' Reviewer: {example.note}' if example.note else ''
            lines.append(f'- Output: {_clip(example.output)}\n  Verdict: {verdict}.{note}')
        parts.append('\n'.join(lines))
    return '\n\n'.join(parts)


def revise_llm_judge(judge: Any, proposal: Proposal) -> Any:
    """A copy of an `LLMJudge` (or any dataclass judge with a `rubric`) with the revised rubric.

    Everything else (model, `include_input`, ...) is kept: the comparison is about the rubric.
    """
    return replace(judge, rubric=revised_rubric(judge.rubric, proposal))


def split_cases(names: Sequence[str], split: float, seed: int) -> tuple[list[str], list[str]]:
    """Training and held-out case names: a seeded shuffle, at least one case on each side."""
    unique = sorted(set(names))
    if len(unique) < 2:
        raise ValueError('at least two cases are needed: one to learn from and one to test on')
    if not 0 < split < 1:
        raise ValueError(f'split must be between 0 and 1, got {split}')
    order = unique[:]
    random.Random(seed).shuffle(order)
    n = min(len(order) - 1, max(1, round(split * len(order))))
    return sorted(order[:n]), sorted(order[n:])


@dataclass(frozen=True)
class _Plan:
    """The held-out certification, and which judgment answers which reviewed output."""

    cases: list[JudgeCase]
    labels: list[HumanLabel]
    labelled: dict[str, list[ReviewedCase]]
    """Case to its reviewed outputs, in the order they are judged as human labels."""
    skipped: list[str]


def _plan(held_out: Mapping[str, Sequence[ReviewedCase]]) -> _Plan:
    """Each held-out case's first approved output is its known-good answer; every reviewed output is a label.

    The approved output is judged twice, as acceptance evidence and as a label. Counting its one
    judgment as both was tried: then most cases' labels were only the outputs the reviewer failed,
    kappa against them was undefined, and the certificate could never be ADMISSIBLE. A case with
    no approved output uses its `expected_output` as the known-good answer, or is skipped if it
    has none.
    """
    cases: list[JudgeCase] = []
    labels: list[HumanLabel] = []
    labelled: dict[str, list[ReviewedCase]] = {}
    skipped: list[str] = []
    for name, reviews in held_out.items():
        first = reviews[0]
        approved = next((r for r in reviews if r.reviewer_passed), None)
        if approved is not None:
            good: Any = approved.output
        elif first.expected_output is not None:
            good = first.expected_output
        else:
            skipped.append(name)
            continue
        cases.append(JudgeCase(name, first.inputs, good, first.expected_output, first.metadata))
        labelled[name] = list(reviews)
        labels.extend(HumanLabel(name, r.output, r.reviewer_passed) for r in reviews)
    return _Plan(cases, labels, labelled, skipped)


def _agreement(certificate: Certificate, plan: _Plan) -> dict[str, list[bool | None]]:
    """Per case, whether each reviewed output's verdict matched the reviewer (None: the judge errored)."""
    out: dict[str, list[bool | None]] = {}
    for case in plan.cases:
        name = case.name
        human = [j for j in certificate.judgments if j.case == name and j.role.startswith('human:')]
        out[name] = [
            None if judgment.passed is None else judgment.passed == review.reviewer_passed
            for review, judgment in zip(plan.labelled[name], human, strict=True)
        ]
    return out


def _paired(
    before: Mapping[str, list[bool | None]], after: Mapping[str, list[bool | None]]
) -> tuple[dict[str, list[bool]], dict[str, list[bool]], int]:
    """Keep each reviewed output both judges answered; an error on either side drops it from both.

    The sign-flip test needs the same outcomes on both sides. Scoring an error as a disagreement
    would charge one judge for a timeout; dropping it on one side only would unpair the case.
    """
    b: dict[str, list[bool]] = {}
    a: dict[str, list[bool]] = {}
    dropped = 0
    for name in before:
        for x, y in zip(before[name], after[name], strict=True):
            if x is None or y is None:
                dropped += 1
                continue
            b.setdefault(name, []).append(x)
            a.setdefault(name, []).append(y)
    return b, a, dropped


@dataclass(frozen=True)
class ApprenticeResult:
    proposal: Proposal
    revised: Any = field(repr=False)
    """The revised judge; use it only if `promoted`."""
    train: tuple[str, ...]
    held_out: tuple[str, ...]
    disagreements: tuple[ReviewedCase, ...] = field(repr=False)
    """The training disagreements the proposer was shown."""
    before: dict[str, list[bool]] = field(repr=False)
    """Held-out case to whether the original judge agreed with the reviewer, per reviewed output."""
    after: dict[str, list[bool]] = field(repr=False)
    gate: GateResult
    baseline_certificate: Certificate = field(repr=False)
    revised_certificate: Certificate = field(repr=False)
    promoted: bool
    reason: str
    ledger: FeedbackLedger = field(repr=False)
    dropped: int = 0
    """Held-out reviewed outputs left out because a judge errored on them."""
    skipped: tuple[str, ...] = ()
    """Held-out cases with no approved output and no expected output: not certified."""

    @staticmethod
    def _count(outcomes: Mapping[str, list[bool]]) -> tuple[int, int]:
        flat = [v for vs in outcomes.values() for v in vs]
        return sum(flat), len(flat)

    @property
    def agreement_before(self) -> tuple[int, int]:
        """Held-out reviewed outputs on which the original judge agreed with the reviewer, of all compared."""
        return self._count(self.before)

    @property
    def agreement_after(self) -> tuple[int, int]:
        return self._count(self.after)

    @property
    def calls(self) -> int:
        """Judge calls made by both certifications (the proposer's calls are not counted here)."""
        return (self.baseline_certificate.calls or 0) + (self.revised_certificate.calls or 0)

    def table(self) -> str:
        rows = [
            f'apprentice: {"PROMOTED" if self.promoted else "NOT PROMOTED"}: {self.reason}',
            f'trained on {len(self.train)} cases ({len(self.disagreements)} disagreements shown), '
            f'held out {len(self.held_out)}',
            '',
            f'{"judge":<10} {"agreement":>10}  {"95% interval":<15} certificate',
        ]
        for label, (k, n), cert in (
            ('original', self.agreement_before, self.baseline_certificate),
            ('revised', self.agreement_after, self.revised_certificate),
        ):
            low, high = wilson(k, n)
            rows.append(f'{label:<10} {f"{k}/{n}":>10}  [{low:.2f}, {high:.2f}]    {cert.verdict}')
        rows += ['', f'gate: {self.gate.summary()}']
        if self.dropped:
            rows.append(f'{self.dropped} reviewed outputs dropped from both sides: a judge errored on them')
        return '\n'.join(rows)

    def to_dict(self, *, judgments: bool = True) -> dict[str, Any]:
        def reviewed(r: ReviewedCase) -> dict[str, Any]:
            return {
                'case': r.case,
                'output': _clip(r.output, 2000),
                'reviewer_passed': r.reviewer_passed,
                'judge_passed': r.judge_passed,
                'judge_reason': r.judge_reason,
                'note': r.note,
            }

        (kb, nb), (ka, na) = self.agreement_before, self.agreement_after
        return {
            'promoted': self.promoted,
            'reason': self.reason,
            'proposal': {
                'clarification': self.proposal.clarification,
                'examples': [reviewed(e) for e in self.proposal.examples],
            },
            'revised_rubric': getattr(self.revised, 'rubric', None),
            'train': list(self.train),
            'held_out': list(self.held_out),
            'disagreements_shown': [reviewed(r) for r in self.disagreements],
            'agreement_before': {'agreed': kb, 'of': nb, 'interval': list(wilson(kb, nb))},
            'agreement_after': {'agreed': ka, 'of': na, 'interval': list(wilson(ka, na))},
            'gate': {
                'decision': self.gate.decision,
                'reason': self.gate.reason,
                'mean_gain': self.gate.mean_gain,
                'interval': list(self.gate.interval) if self.gate.interval else None,
                'cases': self.gate.cases,
                'improved': self.gate.improved,
                'regressed': self.gate.regressed,
                'p_better': self.gate.p_better,
                'p_worse': self.gate.p_worse,
                'per_case': self.gate.per_case,
            },
            'baseline_certificate': self.baseline_certificate.to_dict(judgments=judgments),
            'revised_certificate': self.revised_certificate.to_dict(judgments=judgments),
            'dropped': self.dropped,
            'skipped': list(self.skipped),
            'ledger': self.ledger.to_dict(),
            'calls': self.calls,
        }


async def _call(propose: Propose, rubric: str, shown: Sequence[ReviewedCase]) -> Proposal:
    result = propose(rubric, shown)
    if inspect.isawaitable(result):
        result = await result
    if isinstance(result, str):
        return Proposal(result)
    if isinstance(result, Proposal):
        return result
    raise TypeError(f'propose returned a {type(result).__name__}, not a Proposal or a str')


async def apprentice(
    judge: Evaluator[Any, Any, Any],
    reviewed: Sequence[ReviewedCase],
    *,
    propose: Propose,
    revise: Callable[[Any, Proposal], Any] = revise_llm_judge,
    split: float = 0.5,
    seed: int = 0,
    controls: Sequence[Control] = DEFAULT_CONTROLS,
    repeats: int = 1,
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
    rules: GateRules | None = None,
    ledger: FeedbackLedger | None = None,
    max_examples: int = 4,
    max_concurrency: int = 4,
    by: str = 'apprentice proposer',
) -> ApprenticeResult:
    """Propose a revised judge from training disagreements; promote it only if it agrees more on held-out cases.

    Args:
        judge: The judge to improve, for example an `LLMJudge`.
        reviewed: Reviewed outputs. Those in the training split with a known `judge_passed` that
            differs from the reviewer's verdict are shown to `propose`; the held-out ones never are.
        propose: `propose(rubric, disagreements)` returns a `Proposal` or the clarification text.
            Its examples must come from the training split.
        revise: Builds the revised judge from the original and the proposal.
        split: Share of cases (not outputs) used for training.
        seed: Seeds the split.
        controls: Controls for both held-out certifications. Keep a rejection control: it is what
            catches a revision that agrees more by passing everything.
        repeats: Judgments per known-good answer in each certification (1 skips `stability`).
        thresholds: For both certifications.
        rules: For the gate.
        ledger: Exposures from earlier rounds. Held-out cases already exposed raise `ExposedCases`
            before any judge call; the training cases shown are recorded under `by`.
        max_examples: The most examples the revised rubric may carry.
        max_concurrency: Judgments in flight at once, per certification.
        by: Who the ledger says saw the training disagreements.
    """
    if not reviewed:
        raise ValueError('no reviewed cases')
    if max_examples < 0:
        raise ValueError('max_examples must be >= 0')
    ledger = ledger if ledger is not None else FeedbackLedger()
    by_case: dict[str, list[ReviewedCase]] = {}
    for r in reviewed:
        by_case.setdefault(r.case, []).append(r)
    train, held_out = split_cases(list(by_case), split, seed)
    spent = ledger.exposed(held_out)
    if spent:
        raise ExposedCases(spent)

    shown = [r for name in train for r in by_case[name] if r.disagrees]
    if not shown:
        raise ValueError(
            f'none of the {len(train)} training cases has a reviewed disagreement with a known judge verdict: '
            'there is nothing to learn from'
        )
    ledger.record([r.case for r in shown], by=by)
    rubric = getattr(judge, 'rubric', '') or ''
    proposal = await _call(propose, rubric, shown)
    held = set(held_out)
    leaked = sorted({e.case for e in proposal.examples if e.case in held or e.case not in by_case})
    if leaked:
        raise ValueError(f'the proposal uses examples from outside the training split: {leaked}')
    if len(proposal.examples) > max_examples:
        proposal = replace(proposal, examples=proposal.examples[:max_examples])
    ledger.record([e.case for e in proposal.examples], by=by)
    revised = revise(judge, proposal)

    plan = _plan({name: by_case[name] for name in held_out})
    if not plan.cases:
        raise ValueError('no held-out case has an approved output or an expected output to certify on')

    async def certify(j: Any) -> Certificate:
        return await certify_judge(
            j,
            plan.cases,
            controls=controls,
            human_labels=plan.labels,
            repeats=repeats,
            thresholds=thresholds,
            max_concurrency=max_concurrency,
            seed=seed,
        )

    baseline_certificate = await certify(judge)
    revised_certificate = await certify(revised)
    before, after, dropped = _paired(_agreement(baseline_certificate, plan), _agreement(revised_certificate, plan))
    if not before:
        raise ValueError('every held-out reviewed output errored on one judge or the other: nothing to compare')
    gate = ledger.confirm(before, after, rules=rules)

    if gate.decision != 'PROMOTE':
        promoted, reason = False, f'the gate said {gate.decision}: {gate.reason}'
    elif revised_certificate.verdict == 'INADMISSIBLE':
        failing = ', '.join(c.name for c in revised_certificate.failures())
        promoted, reason = False, f'it agrees more, but its certificate is INADMISSIBLE ({failing})'
    else:
        promoted = True
        reason = 'it agrees with reviewers more on held-out cases, beyond chance'
        if revised_certificate.verdict != 'ADMISSIBLE':
            reason += f'; its certificate is {revised_certificate.verdict}: certify it on more cases before use'
    return ApprenticeResult(
        proposal=proposal,
        revised=revised,
        train=tuple(train),
        held_out=tuple(held_out),
        disagreements=tuple(shown),
        before=before,
        after=after,
        gate=gate,
        baseline_certificate=baseline_certificate,
        revised_certificate=revised_certificate,
        promoted=promoted,
        reason=reason,
        ledger=ledger,
        dropped=dropped,
        skipped=tuple(plan.skipped),
    )
