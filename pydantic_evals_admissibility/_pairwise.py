"""Certify a judge that picks the better of two answers, including whether position decides.

Pydantic's own writing names this failure first: "Swap the order of two answers and an LLM judge
will often flip its verdict on the same pair, favoring whichever it saw first." A comparison
judge is asked about each pair twice, once each way round, and must pick the same answer both
times. Each pair has a known better answer, so accuracy is measured too, and when the judge does
flip, the certificate says which position it went with.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field

from ._certify import Certificate, Judgment, Verdict, _check
from ._stats import wilson

Choice = Literal['A', 'B']
Compare = Callable[[Any, Any, Any], Awaitable[Choice]]
"""`await compare(inputs, a, b)` returns which of `a` and `b` is better: 'A' or 'B'."""

_INSTRUCTIONS = (
    'You compare two answers, A and B, to the same input, according to a rubric. Decide which '
    'answer satisfies the rubric better. The order of the answers carries no information.'
)


class PairVerdict(BaseModel):
    reason: str = Field(description='One or two sentences on why the chosen answer is better.')
    choice: Choice


@dataclass(frozen=True)
class PairCase:
    """A pair with a known better answer: `better` should win whichever way round it is shown."""

    name: str
    inputs: Any
    better: Any
    worse: Any


@dataclass
class PairwiseJudge:
    """A comparison judge on any Pydantic AI model, returning a typed `PairVerdict`."""

    rubric: str
    model: Any = None

    async def __call__(self, inputs: Any, a: Any, b: Any) -> Choice:
        from pydantic_ai import Agent

        agent = Agent(self.model, output_type=PairVerdict, instructions=_INSTRUCTIONS)
        prompt = f'<Input>\n{inputs}\n</Input>\n<A>\n{a}\n</A>\n<B>\n{b}\n</B>\n<Rubric>\n{self.rubric}\n</Rubric>'
        return (await agent.run(prompt)).output.choice


@dataclass(frozen=True)
class PairThresholds:
    min_accuracy: float = 0.8
    """Lower bound on the share of presentations in which the judge picks the better answer."""
    min_consistency: float = 0.9
    """Lower bound on the share of pairs where it picks the same answer both ways round."""
    min_trials: int = 10


async def certify_pairwise(
    compare: Compare,
    cases: Sequence[PairCase],
    *,
    repeats: int = 1,
    thresholds: PairThresholds | None = None,
    max_concurrency: int = 8,
) -> Certificate:
    """Ask `compare` about every pair both ways round, `repeats` times, and certify it.

    Checks:
        accuracy: it picks the better answer.
        order_consistency: it picks the same answer when the order is swapped. The detail says,
            for the pairs where it did not, how often it went with the answer shown first.
    """
    thresholds = thresholds or PairThresholds()
    limit = asyncio.Semaphore(max_concurrency)

    async def ask(case: PairCase, better_first: bool, repeat: int) -> Judgment:
        a, b = (case.better, case.worse) if better_first else (case.worse, case.better)
        role = f'{"better_first" if better_first else "worse_first"}#{repeat}'
        async with limit:
            try:
                choice = await compare(case.inputs, a, b)
            except Exception as error:  # an error is not a choice
                return Judgment(case.name, role, (a, b), None, error=f'{type(error).__name__}: {error}'[:300])
        picked_better = choice == ('A' if better_first else 'B')
        return Judgment(case.name, role, (a, b), picked_better, reason=f'chose {choice}')

    plan = [(case, first, r) for case in cases for r in range(repeats) for first in (True, False)]
    judgments = await asyncio.gather(*(ask(c, first, r) for c, first, r in plan))

    accuracy = _check(
        'accuracy',
        sum(j.passed is True for j in judgments),
        len(judgments),
        thresholds.min_accuracy,
        thresholds.min_trials,
        errors=sum(j.error is not None for j in judgments),
    )
    by_pair: dict[tuple[str, str], dict[bool, Judgment]] = {}
    for j in judgments:
        first = j.role.startswith('better_first')
        by_pair.setdefault((j.case, j.role.split('#')[1]), {})[first] = j
    consistent = 0
    flipped_to_first = flipped = 0
    for pair in by_pair.values():
        both = pair.get(True), pair.get(False)
        if any(x is None or x.passed is None for x in both):
            continue
        assert both[0] is not None and both[1] is not None
        if both[0].passed == both[1].passed:
            consistent += 1
        else:
            flipped += 1
            # It flipped: one presentation picked A and the other B, so it followed position.
            # Picking the better answer when it came first means it chose A both times.
            flipped_to_first += both[0].passed is True
    detail = (
        f'when it flipped, it chose the answer shown first in {flipped_to_first}/{flipped}'
        if flipped
        else 'never flipped'
    )
    consistency = _check(
        'order_consistency', consistent, len(by_pair), thresholds.min_consistency, thresholds.min_trials, detail=detail
    )
    checks = (accuracy, consistency)
    if any(c.status == 'FAIL' for c in checks):
        verdict: Verdict = 'INADMISSIBLE'
    elif any(c.status == 'UNVALIDATED' for c in checks):
        verdict = 'UNVALIDATED'
    else:
        verdict = 'ADMISSIBLE'
    name = type(compare).__name__ if not hasattr(compare, '__name__') else compare.__name__  # type: ignore[attr-defined]
    model = getattr(compare, 'model', None)
    label = f'{name}({getattr(model, "model_name", model)})' if model is not None else name
    return Certificate(verdict, checks, tuple(judgments), judge=label, calls=len(plan), planned=len(plan))


def first_position_rate(certificate: Certificate) -> tuple[float, tuple[float, float]] | None:
    """Share of all presentations in which the judge chose whichever answer was shown first.

    0.5 is no preference. Returned with its Wilson interval, or None if nothing was judged.
    """
    picks = [j for j in certificate.judgments if j.passed is not None]
    if not picks:
        return None
    first = sum((j.role.startswith('better_first')) == j.passed for j in picks)
    return first / len(picks), wilson(first, len(picks))


def both_orders(compare: Compare) -> Callable[[Any, Any, Any], Awaitable[Choice | None]]:
    """Ask both ways round and answer only when the two agree; otherwise None (abstain).

    The standard defence against position bias, at twice the calls. On the Codex pairwise judge
    in this package's README it raised accuracy from 0.77 (83/108 single presentations) to 0.85
    (35/41) on the pairs it answered, abstaining on 13 of 54.
    """

    async def wrapped(inputs: Any, a: Any, b: Any) -> Choice | None:
        first, swapped = await asyncio.gather(compare(inputs, a, b), compare(inputs, b, a))
        if first == 'A' and swapped == 'B':
            return 'A'
        if first == 'B' and swapped == 'A':
            return 'B'
        return None

    return wrapped
