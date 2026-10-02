"""Certify a judge that picks the better of two answers, including whether position decides.

Pydantic's own writing names this failure first: "Swap the order of two answers and an LLM judge
will often flip its verdict on the same pair, favoring whichever it saw first." A comparison
judge is asked about each pair twice, once each way round, and must pick the same answer both
times. Each pair has a known better answer, so accuracy is measured too, and when the judge does
flip, the certificate says which position it went with.
"""

from __future__ import annotations

import asyncio
import random
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from statistics import NormalDist
from typing import Any, Literal

from pydantic import BaseModel, Field

from ._certify import Certificate, Check, Judgment, Verdict, _check

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
    """Lower bound on the share of pairs in which the judge picks the better answer.

    One presentation per pair counts, the better answer first in alternate pairs: the two
    presentations of a pair share its difficulty and are not two pieces of evidence.
    """
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
    if max_concurrency < 1 or repeats < 1:
        raise ValueError('max_concurrency and repeats must be >= 1')
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

    verdict, checks = _assess_pairs(judgments, [case.name for case in cases], thresholds)
    name = type(compare).__name__ if not hasattr(compare, '__name__') else compare.__name__  # type: ignore[attr-defined]
    model = getattr(compare, 'model', None)
    label = f'{name}({getattr(model, "model_name", model)})' if model is not None else name
    return Certificate(verdict, checks, tuple(judgments), judge=label, calls=len(plan), planned=len(plan))


def _assess_pairs(
    judgments: Sequence[Judgment], names: Sequence[str], thresholds: PairThresholds
) -> tuple[Verdict, tuple[Check, ...]]:
    """Both checks and the verdict from a comparison judge's verdicts; `names` fixes the pair order."""
    # Two checks can fail the certificate: each FAIL gets half of the 2.5% upper tail.
    z_fail = NormalDist().inv_cdf(1 - 0.05 / 4)
    order = {name: i for i, name in enumerate(names)}
    counted = [j for j in judgments if j.role == ('better_first#0' if order[j.case] % 2 == 0 else 'worse_first#0')]
    accuracy = _check(
        'accuracy',
        sum(j.passed is True for j in counted),
        sum(j.passed is not None for j in counted),
        thresholds.min_accuracy,
        thresholds.min_trials,
        detail='one presentation per pair, better answer first in alternate pairs',
        errors=sum(j.passed is None for j in counted),
        z_fail=z_fail,
    )
    by_pair: dict[str, dict[bool, Judgment]] = {}
    for j in judgments:
        if j.role.endswith('#0'):  # the first repeat: pairs, not repeats, are the unit
            by_pair.setdefault(j.case, {})[j.role.startswith('better_first')] = j
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
    complete = consistent + flipped
    consistency = _check(
        'order_consistency',
        consistent,
        complete,
        thresholds.min_consistency,
        thresholds.min_trials,
        detail=detail,
        errors=len(by_pair) - complete,
        z_fail=z_fail,
    )
    checks = (accuracy, consistency)
    if any(c.status == 'FAIL' for c in checks):
        verdict: Verdict = 'INADMISSIBLE'
    elif any(c.status == 'UNVALIDATED' for c in checks):
        verdict = 'UNVALIDATED'
    else:
        verdict = 'ADMISSIBLE'
    return verdict, checks


def recertify_pairwise(certificate: Certificate, thresholds: PairThresholds | None = None) -> Certificate:
    """Decide a comparison judge's certificate again from its saved verdicts, without calling it."""
    names = list(dict.fromkeys(j.case for j in certificate.judgments))
    verdict, checks = _assess_pairs(certificate.judgments, names, thresholds or PairThresholds())
    return Certificate(
        verdict, checks, certificate.judgments, certificate.judge, certificate.calls, certificate.planned
    )


def first_position_rate(certificate: Certificate) -> tuple[float, tuple[float, float]] | None:
    """Share of all presentations in which the judge chose whichever answer was shown first.

    0.5 is no preference. Returned with a 95% interval that resamples pairs, since the
    presentations of one pair are not independent, or None if nothing was judged. An interval
    around 0.5 means no preference was detected, not that there is none.
    """
    by_pair: dict[str, list[bool]] = {}
    for j in certificate.judgments:
        if j.passed is not None:
            by_pair.setdefault(j.case, []).append(j.role.startswith('better_first') == j.passed)
    if not by_pair:
        return None
    pairs = list(by_pair.values())
    picks = [p for ps in pairs for p in ps]
    rng = random.Random(0)
    rates = []
    for _ in range(2000):
        sample = [p for _ in pairs for p in pairs[rng.randrange(len(pairs))]]
        rates.append(sum(sample) / len(sample))
    rates.sort()
    return sum(picks) / len(picks), (rates[50], rates[1949])


def both_orders(compare: Compare) -> Callable[[Any, Any, Any], Awaitable[Choice | None]]:
    """Ask both ways round and answer only when the two agree; otherwise None (abstain).

    The standard defence against position bias, at twice the calls. On the Codex pairwise judge
    in this package's README it raised accuracy from 0.85 (92/108 single presentations) to 0.90
    (43/48) on the pairs it answered, abstaining on 6 of 54.
    """

    async def wrapped(inputs: Any, a: Any, b: Any) -> Choice | None:
        first, swapped = await asyncio.gather(compare(inputs, a, b), compare(inputs, b, a))
        if first == 'A' and swapped == 'B':
            return 'A'
        if first == 'B' and swapped == 'A':
            return 'B'
        return None

    return wrapped
