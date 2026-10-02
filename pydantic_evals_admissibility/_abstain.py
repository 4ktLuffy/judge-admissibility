"""A judge allowed to say it cannot decide, and why; and a certificate for when it should.

A pass/fail judge shown a trace with the decisive tool result missing has to guess, and its
guess lands in the eval report like any other verdict. `AbstainingJudge` may answer `abstain`
with a reason (`missing_evidence`, `ambiguous_rubric`, `unsupported_task`). That is only useful
if it abstains when, and only when, it should, so `certify_abstention` measures it on three
kinds of case:

- known-good answers it must pass, and `must_fail` answers it must fail: it should decide
  (`coverage`) and decide right (`decided_accuracy`);
- `must_abstain` cases, whose evidence is missing: it should abstain, for the right reason
  (`abstention_recall`), and its abstentions should fall on those cases (`abstention_precision`).

Any pass/fail evaluator, such as an `LLMJudge`, can be certified the same way for comparison:
it never abstains, so its recall is 0 and the detail says what it guessed instead.
"""

from __future__ import annotations

import asyncio
import json
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

from pydantic import BaseModel, Field
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from ._cases import JudgeCase
from ._certify import Certificate, Check, Judgment, Verdict, _check, _context, assertion_of
from ._identity import judge_identity

Decision = Literal['pass', 'fail', 'abstain']
AbstainReason = Literal['missing_evidence', 'ambiguous_rubric', 'unsupported_task']
Role = Literal['good', 'must_fail', 'must_abstain']

_INSTRUCTIONS = (
    'You are grading output according to a user-specified rubric. If the statement in the rubric is true '
    'for the provided input and output, then the output passes the test: decide "pass", otherwise "fail". '
    'If you cannot tell, answer "abstain" and say why: "missing_evidence" when the evidence that would '
    'decide it is not in the input, '
    '"ambiguous_rubric" when the rubric can be read to give either verdict, "unsupported_task" when '
    'the rubric does not apply to this kind of output. Do not guess. Give abstain_reason only when '
    'you abstain; otherwise null.'
)


class AbstainVerdict(BaseModel):
    """The reason first, then the decision: the judge reasons before it commits (see DESIGN.md)."""

    reason: str = Field(description='One or two sentences on what the evidence shows, or what is missing.')
    decision: Decision
    abstain_reason: AbstainReason | None = Field(description='Why it cannot decide; null unless decision is abstain.')


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=1, default=str)


@dataclass(repr=False)
class AbstainingJudge(Evaluator[object, object, object]):
    """A judge on any Pydantic AI model that may abstain, returning an `AbstainVerdict`; it sees the inputs.

    As an evaluator it returns the decision as a label ('pass', 'fail' or 'abstain'), with the
    reason. Its abstentions are not failures: count them apart, as `certify_abstention` does.
    """

    rubric: str
    model: Any = None

    async def verdict(self, inputs: Any, output: Any) -> AbstainVerdict:
        from pydantic_ai import Agent

        agent = Agent(self.model, output_type=AbstainVerdict, instructions=_INSTRUCTIONS)
        prompt = (
            f'<Input>\n{_render(inputs)}\n</Input>\n<Output>\n{_render(output)}\n</Output>\n'
            f'<Rubric>\n{self.rubric}\n</Rubric>'
        )
        return (await agent.run(prompt)).output

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        found = await self.verdict(ctx.inputs, ctx.output)
        why = f' ({found.abstain_reason})' if found.decision == 'abstain' else ''
        return EvaluationReason(value=found.decision, reason=f'{found.reason}{why}')


@dataclass(frozen=True)
class AbstainJudgment:
    case: str
    role: Role
    decision: Decision | None
    """None when the judge errored."""
    abstain_reason: str | None
    reason: str | None
    error: str | None = None


@dataclass(frozen=True)
class AbstainThresholds:
    min_coverage: float = 0.8
    """Share of answerable cases (good and must-fail) the judge decides rather than abstains on."""
    min_accuracy: float = 0.8
    """Share of decided answerable cases decided right."""
    min_recall: float = 0.8
    """Share of must-abstain cases it abstains on, with the expected reason."""
    min_precision: float = 0.8
    """Share of its abstentions that fall on must-abstain cases."""
    min_trials: int = 10


@dataclass(frozen=True)
class AbstentionCertificate:
    """A certificate for a judge that may abstain, with every decision for audit.

    In `certificate.judgments` an abstention has `passed=None` and no error; its reason starts
    with `abstain(<why>)`.
    """

    certificate: Certificate
    decisions: tuple[AbstainJudgment, ...]

    @property
    def verdict(self) -> Verdict:
        return self.certificate.verdict

    def table(self) -> str:
        return self.certificate.table()

    def check(self, name: str) -> Check:
        return next(c for c in self.certificate.checks if c.name == name)

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.certificate.to_dict(judgments=False),
            'decisions': [
                {'case': d.case, 'role': d.role, 'decision': d.decision, 'abstain_reason': d.abstain_reason,
                 'reason': d.reason, 'error': d.error}
                for d in self.decisions
            ],
        }  # fmt: skip


async def _decide(judge: Any, case: JudgeCase) -> tuple[Decision, str | None, str | None]:
    """(decision, abstain_reason, reason) from an `AbstainingJudge`, or from any pass/fail evaluator."""
    if isinstance(judge, AbstainingJudge):
        found = await judge.verdict(case.inputs, case.output)
        return found.decision, found.abstain_reason, found.reason
    passed, reason = assertion_of(await judge.evaluate_async(_context(case, case.output)))
    return ('pass' if passed else 'fail'), None, reason


async def certify_abstention(
    judge: AbstainingJudge | Evaluator[Any, Any, Any],
    cases: Sequence[JudgeCase],
    *,
    must_fail: Sequence[JudgeCase] = (),
    must_abstain: Sequence[JudgeCase] = (),
    abstain_reason: AbstainReason | None = 'missing_evidence',
    thresholds: AbstainThresholds | None = None,
    max_concurrency: int = 8,
) -> AbstentionCertificate:
    """Judge every case once and certify when the judge decides and when it abstains.

    Args:
        judge: An `AbstainingJudge`, or any pass/fail evaluator (which never abstains).
        cases: Known-good answers: the judge should decide, and pass them.
        must_fail: Answers it should decide, and fail.
        must_abstain: Cases it cannot decide, for example with the decisive tool result removed.
        abstain_reason: The reason a must-abstain case should be given; None accepts any.

    Each case is judged once (the case is the unit). Coverage and accuracy pool good and
    must-fail cases, with each family's counts in the detail; a judge that always passes is still
    caught, since half its decided answers are wrong.
    """
    if max_concurrency < 1:
        raise ValueError('max_concurrency must be >= 1')
    thresholds = thresholds or AbstainThresholds()
    limit = asyncio.Semaphore(max_concurrency)

    async def ask(case: JudgeCase, role: Role) -> AbstainJudgment:
        async with limit:
            try:
                decision, why, reason = await _decide(judge, case)
            except Exception as error:  # an error is neither a verdict nor an abstention
                return AbstainJudgment(case.name, role, None, None, None, f'{type(error).__name__}: {error}'[:300])
        return AbstainJudgment(case.name, role, decision, why, reason)

    def tag(group: Sequence[JudgeCase], role: Role) -> list[tuple[JudgeCase, Role]]:
        return [(c, role) for c in group]

    plan = tag(cases, 'good') + tag(must_fail, 'must_fail') + tag(must_abstain, 'must_abstain')
    decisions = tuple(await asyncio.gather(*(ask(c, role) for c, role in plan)))
    verdict, checks = _assess_abstention(decisions, abstain_reason, thresholds)
    judgments = tuple(
        Judgment(
            d.case,
            d.role,
            case.output,
            None if d.decision in (None, 'abstain') else d.decision == 'pass',
            (f'abstain({d.abstain_reason}) ' if d.decision == 'abstain' else '') + (d.reason or '') or None,
            d.error,
        )
        for d, (case, _) in zip(decisions, plan, strict=True)
    )
    model = getattr(judge, 'model', None)
    certificate = Certificate(
        verdict,
        checks,
        judgments,
        judge=f'{type(judge).__name__}({getattr(model, "model_name", model)})',
        calls=len(plan),
        planned=len(plan),
        identity=judge_identity(judge),
    )
    return AbstentionCertificate(certificate, decisions)


def _assess_abstention(
    decisions: Sequence[AbstainJudgment], expected_reason: str | None, thresholds: AbstainThresholds
) -> tuple[Verdict, tuple[Check, ...]]:
    done = [d for d in decisions if d.decision is not None]
    answerable = [d for d in done if d.role != 'must_abstain']
    missing = [d for d in done if d.role == 'must_abstain']
    errors_answerable = sum(d.decision is None and d.role != 'must_abstain' for d in decisions)
    errors_missing = sum(d.decision is None and d.role == 'must_abstain' for d in decisions)
    z_fail = NormalDist().inv_cdf(1 - 0.025 / (4 if missing or errors_missing else 2))

    decided = [d for d in answerable if d.decision != 'abstain']
    right = [d for d in decided if d.decision == ('pass' if d.role == 'good' else 'fail')]

    def family(role: str) -> str:
        group = [d for d in answerable if d.role == role]
        counts = Counter(d.decision for d in group)
        return f'{role}: {counts["pass"]} pass, {counts["fail"]} fail, {counts["abstain"]} abstain'

    families = '; '.join(family(r) for r in ('good', 'must_fail') if any(d.role == r for d in answerable))
    checks = [
        _check(
            'coverage',
            len(decided),
            len(answerable),
            thresholds.min_coverage,
            thresholds.min_trials,
            detail=families,
            errors=errors_answerable,
            z_fail=z_fail,
        ),
        _check(
            'decided_accuracy',
            len(right),
            len(decided),
            thresholds.min_accuracy,
            thresholds.min_trials,
            errors=errors_answerable,
            z_fail=z_fail,
        ),
    ]
    if missing or errors_missing:
        counts = Counter(d.decision for d in missing)
        hit = [
            d
            for d in missing
            if d.decision == 'abstain' and (expected_reason is None or d.abstain_reason == expected_reason)
        ]
        other = counts['abstain'] - len(hit)
        detail = f'guessed instead: {counts["pass"]} pass, {counts["fail"]} fail' + (
            f'; {other} abstained for another reason' if other else ''
        )
        checks.append(
            _check(
                'abstention_recall',
                len(hit),
                len(missing),
                thresholds.min_recall,
                thresholds.min_trials,
                detail=detail,
                errors=errors_missing,
                z_fail=z_fail,
            )
        )
        abstained = [d for d in done if d.decision == 'abstain']
        checks.append(
            _check(
                'abstention_precision',
                sum(d.role == 'must_abstain' for d in abstained),
                len(abstained),
                thresholds.min_precision,
                thresholds.min_trials,
                detail='share of its abstentions that were on cases missing the evidence',
                z_fail=z_fail,
            )
        )
    if any(c.status == 'FAIL' for c in checks):
        verdict: Verdict = 'INADMISSIBLE'
    elif any(c.status == 'UNVALIDATED' for c in checks):
        verdict = 'UNVALIDATED'
    else:
        verdict = 'ADMISSIBLE'
    return verdict, tuple(checks)
