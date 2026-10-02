"""A judge that must cite its evidence, and a check that the citations are real and relevant.

A judge's reason can sound right and rest on nothing: an id it made up, or a tool call that has
nothing to do with the verdict. `CitedJudge` asks for the ids of the items in the inputs its
verdict depends on (spans, tool calls, messages), and `verify_citations` checks two things a
program can check without another model:

- **validity**: every cited id exists in the inputs, and at least one was cited;
- **support**: the cited items hold the fact the verdict is about. The default rule is
  deterministic: each of the case's `facts` (strings) must appear in a cited item, as a key or
  as a string value at any depth. On the evidence episodes, the fact is the action's tool name,
  so citing the order lookup instead of the refund call is unsupported. Pass `supports=` for a
  rule of your own.

Support is about *where* the judge looked, not whether its verdict follows: a judge that cites
the failed refund call and still passes the reply is supported and wrong, which acceptance and
rejection report separately.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any

from pydantic import BaseModel, Field
from pydantic_evals.evaluators import EvaluationReason, Evaluator, EvaluatorContext

from ._cases import JudgeCase
from ._certify import Certificate, Check, Judgment, Verdict, _check
from ._identity import judge_identity

ID_KEYS = ('span_id', 'tool_call_id', 'message_id')
"""Keys whose value names an item of evidence: a dict carrying one is an item, under that id."""

# The first sentence is `LLMJudge`'s own framing. A first draft opened with "using the evidence in
# the input"; on one Codex call it failed a true reply for a delivery estimate the tools never gave.
_INSTRUCTIONS = (
    'You are grading output according to a user-specified rubric. If the statement in the rubric is true '
    'for the provided input and output, then the output passes the test. Cite the evidence your verdict '
    'depends on: the ids ({keys} values) of the items in the input that decide it. Cite only ids that '
    'appear in the input, and only the items that decide the verdict.'
)


class CitedVerdict(BaseModel, populate_by_name=True):
    """A verdict with its evidence. The reason and the citations come before the verdict, on purpose.

    Structured output is written in schema order; a judge asked for its verdict first commits
    before it has reasoned (see DESIGN.md, "Why must the verdict come after the reason").
    """

    reason: str = Field(description='One or two sentences on why the output does or does not meet the rubric.')
    citations: list[str] = Field(description='Ids of the input items the verdict depends on.')
    pass_: bool = Field(validation_alias='pass', serialization_alias='pass')


Supports = Callable[[JudgeCase, Mapping[str, Any], CitedVerdict], bool]
"""`supports(case, cited_items, verdict)`: do the (real) cited items hold what the verdict is about?"""
Facts = Collection[str] | Callable[[JudgeCase], Collection[str]]


def _render(value: Any) -> str:
    return value if isinstance(value, str) else json.dumps(value, indent=1, default=str)


@dataclass(repr=False)
class CitedJudge(Evaluator[object, object, object]):
    """A judge on any Pydantic AI model that returns a `CitedVerdict`; it always sees the inputs.

    As an evaluator it returns the verdict with the reason and the cited ids, so it can run in a
    `Dataset` or through `certify_judge`. `verdict(inputs, output)` returns the typed verdict,
    which `verify_citations` and `certify_citations` read.
    """

    rubric: str
    model: Any = None
    id_keys: tuple[str, ...] = ID_KEYS

    async def verdict(self, inputs: Any, output: Any) -> CitedVerdict:
        from pydantic_ai import Agent

        agent = Agent(
            self.model, output_type=CitedVerdict, instructions=_INSTRUCTIONS.format(keys=', '.join(self.id_keys))
        )
        prompt = (
            f'<Input>\n{_render(inputs)}\n</Input>\n<Output>\n{_render(output)}\n</Output>\n'
            f'<Rubric>\n{self.rubric}\n</Rubric>'
        )
        return (await agent.run(prompt)).output

    async def evaluate(self, ctx: EvaluatorContext[object, object, object]) -> EvaluationReason:
        found = await self.verdict(ctx.inputs, ctx.output)
        return EvaluationReason(value=found.pass_, reason=f'{found.reason} [cites: {", ".join(found.citations)}]')


def evidence_items(inputs: Any, id_keys: Sequence[str] = ID_KEYS) -> dict[str, Any]:
    """Every item in `inputs` that carries an id: a mapping (at any depth) with one of `id_keys`."""
    items: dict[str, Any] = {}

    def walk(value: Any) -> None:
        if isinstance(value, BaseModel):
            value = value.model_dump()
        if isinstance(value, Mapping):
            for key in id_keys:
                if isinstance(value.get(key), str):
                    items.setdefault(value[key], value)
            for child in value.values():
                walk(child)
        elif isinstance(value, list | tuple):
            for child in value:
                walk(child)

    walk(inputs)
    return items


def _mentions(value: Any, fact: str) -> bool:
    """`fact` is a key or a string value somewhere in `value`."""
    if isinstance(value, Mapping):
        return fact in value or any(_mentions(v, fact) for v in value.values())
    if isinstance(value, list | tuple):
        return any(_mentions(v, fact) for v in value)
    return value == fact


def facts_rule(facts: Facts) -> Supports:
    """The default support rule: every fact the case's verdict depends on appears in a cited item."""

    def supports(case: JudgeCase, cited: Mapping[str, Any], verdict: CitedVerdict) -> bool:
        needed = facts(case) if callable(facts) else facts
        return bool(needed) and all(any(_mentions(item, fact) for item in cited.values()) for fact in needed)

    return supports


@dataclass(frozen=True)
class CitationCheck:
    """One verdict's citations, checked against the inputs it was given."""

    case: str
    role: str
    passed: bool | None
    reason: str | None
    cited: tuple[str, ...]
    unknown: tuple[str, ...]
    """Cited ids that are not in the inputs: made up, or copied from somewhere else."""
    valid: bool | None
    """At least one id cited, and every one of them real. None when the judge errored."""
    supported: bool | None
    """The real cited items hold the fact the verdict is about. None when unchecked or errored."""
    error: str | None = None


def verify_citations(
    case: JudgeCase,
    verdict: CitedVerdict,
    *,
    facts: Facts | None = None,
    supports: Supports | None = None,
    id_keys: Sequence[str] = ID_KEYS,
    role: str = 'case',
) -> CitationCheck:
    """Are the cited ids real (in `case.inputs`), and do the cited items support the verdict?

    Support is judged on the real cited items only, by `supports` or else the `facts` rule; with
    neither, it is not checked (None). A verdict that cites nothing is not valid: an empty list
    is vacuously "all real", and a judge could satisfy it by never citing.
    """
    items = evidence_items(case.inputs, id_keys)
    cited = tuple(dict.fromkeys(verdict.citations))
    unknown = tuple(c for c in cited if c not in items)
    rule = supports or (facts_rule(facts) if facts is not None else None)
    real = {c: items[c] for c in cited if c in items}
    supported = rule(case, real, verdict) if rule is not None else None
    return CitationCheck(
        case.name, role, verdict.pass_, verdict.reason, cited, unknown, bool(cited) and not unknown, supported
    )


@dataclass(frozen=True)
class CitationThresholds:
    min_acceptance: float = 0.7
    min_rejection: float = 0.8
    min_validity: float = 0.9
    """Share of verdicts whose citations are all real. A made-up id is never acceptable, so the bar is high."""
    min_support: float = 0.8
    """Share of verdicts whose cited items hold the fact the verdict is about."""
    min_trials: int = 10


@dataclass(frozen=True)
class CitedCertificate:
    """A certificate for a citing judge, with every verdict's citation check for audit."""

    certificate: Certificate
    citations: tuple[CitationCheck, ...]

    @property
    def verdict(self) -> Verdict:
        return self.certificate.verdict

    def table(self) -> str:
        return self.certificate.table()

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.certificate.to_dict(judgments=False),
            'citations': [
                {'case': c.case, 'role': c.role, 'passed': c.passed, 'reason': c.reason, 'cited': list(c.cited),
                 'unknown': list(c.unknown), 'valid': c.valid, 'supported': c.supported, 'error': c.error}
                for c in self.citations
            ],
        }  # fmt: skip


async def certify_citations(
    judge: CitedJudge,
    cases: Sequence[JudgeCase],
    *,
    must_fail: Sequence[JudgeCase] = (),
    facts: Facts | None = None,
    supports: Supports | None = None,
    thresholds: CitationThresholds | None = None,
    max_concurrency: int = 8,
) -> CitedCertificate:
    """Judge known-good `cases` and `must_fail` ones once each, and check every verdict's citations.

    Checks: `acceptance` (good cases passed), `rejection` (must-fail cases failed, when given),
    `citation_validity` and `citation_support` over every verdict. Each case is judged once:
    the unit is the case. Errored judgments are left out of the rates and counted.
    """
    if facts is None and supports is None:
        raise ValueError('pass `facts` or `supports`: without one, support cannot be checked')
    if max_concurrency < 1:
        raise ValueError('max_concurrency must be >= 1')
    thresholds = thresholds or CitationThresholds()
    limit = asyncio.Semaphore(max_concurrency)

    async def ask(case: JudgeCase, role: str) -> CitationCheck:
        async with limit:
            try:
                found = await judge.verdict(case.inputs, case.output)
            except Exception as error:  # an error is not a verdict
                message = f'{type(error).__name__}: {error}'[:300]
                return CitationCheck(case.name, role, None, None, (), (), None, None, message)
        return verify_citations(case, found, facts=facts, supports=supports, id_keys=judge.id_keys, role=role)

    plan = [(c, 'good') for c in cases] + [(c, 'must_fail') for c in must_fail]
    checked: list[CitationCheck] = list(await asyncio.gather(*(ask(c, role) for c, role in plan)))
    verdict, checks = _assess_citations(checked, thresholds)
    judgments = tuple(
        Judgment(
            c.case, c.role, case.output, c.passed, c.reason and f'{c.reason} [cites: {", ".join(c.cited)}]', c.error
        )
        for c, (case, _) in zip(checked, plan, strict=True)
    )
    certificate = Certificate(
        verdict,
        checks,
        judgments,
        judge=f'CitedJudge({getattr(judge.model, "model_name", judge.model)})',
        calls=len(plan),
        planned=len(plan),
        identity=judge_identity(judge),
    )
    return CitedCertificate(certificate, tuple(checked))


def _assess_citations(
    checked: Sequence[CitationCheck], thresholds: CitationThresholds
) -> tuple[Verdict, tuple[Check, ...]]:
    good = [c for c in checked if c.role == 'good']
    bad = [c for c in checked if c.role == 'must_fail']
    done = [c for c in checked if c.error is None]
    errors = len(checked) - len(done)
    families = 3 + bool(bad)
    z_fail = NormalDist().inv_cdf(1 - 0.025 / families)  # FAILs share one 2.5% upper tail
    unknown = sum(bool(c.unknown) for c in done)
    empty = sum(not c.cited for c in done)

    def verdicts(group: list[CitationCheck], want: bool, name: str, bar: float) -> Check:
        ok = [c for c in group if c.error is None]
        return _check(
            name,
            sum(c.passed is want for c in ok),
            len(ok),
            bar,
            thresholds.min_trials,
            errors=len(group) - len(ok),
            z_fail=z_fail,
        )

    checks = [verdicts(good, True, 'acceptance', thresholds.min_acceptance)]
    if bad:
        checks.append(verdicts(bad, False, 'rejection', thresholds.min_rejection))
    checks.append(
        _check(
            'citation_validity',
            sum(c.valid is True for c in done),
            len(done),
            thresholds.min_validity,
            thresholds.min_trials,
            detail=f'{unknown} cited an id not in the inputs, {empty} cited nothing',
            errors=errors,
            z_fail=z_fail,
        )
    )
    checks.append(
        _check(
            'citation_support',
            sum(c.supported is True for c in done),
            len(done),
            thresholds.min_support,
            thresholds.min_trials,
            detail='the real cited items hold the fact the verdict is about',
            errors=errors,
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
