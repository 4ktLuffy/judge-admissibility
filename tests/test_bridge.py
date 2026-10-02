"""The judge migration bridge: a judge change that moves both versions alike keeps gains comparable; one that
favours a version does not; few cases prove neither."""

from __future__ import annotations

import random
from dataclasses import dataclass

import pytest
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from pydantic_evals_admissibility import GateRules, compare_judge_reports, compare_judges

FAST = GateRules(resamples=2000)


def scored(n: int, seed: int, *, lenient: float, favours_candidate: float = 0.0, gain: float = 0.1):
    """Old and new judge on the same outputs: the new one also passes a share of what the old failed."""
    rng = random.Random(seed)
    old: dict[str, dict[str, list[bool]]] = {'baseline': {}, 'candidate': {}}
    new: dict[str, dict[str, list[bool]]] = {'baseline': {}, 'candidate': {}}
    for i in range(n):
        p = rng.uniform(0.1, 0.9)
        base, cand = rng.random() < p, rng.random() < min(1.0, p + gain)
        old['baseline'][f'c{i}'], old['candidate'][f'c{i}'] = [base], [cand]
        new['baseline'][f'c{i}'] = [base or rng.random() < lenient]
        new['candidate'][f'c{i}'] = [cand or rng.random() < lenient + favours_candidate]
    return old, new


def test_a_uniformly_lenient_judge_moves_rates_but_keeps_the_gain() -> None:
    old, new = scored(800, seed=0, lenient=0.1)
    bridge = compare_judges(old, new, margin=0.05, rules=FAST)
    assert bridge.shift_baseline.interval[0] > 0  # absolute rates moved
    assert bridge.rates['baseline', 'new'] > bridge.rates['baseline', 'old']
    assert bridge.verdict == 'PASS' and bridge.comparable is True
    assert abs(bridge.interaction.estimate) < 0.02
    assert 'comparable: PASS' in bridge.table()


def test_a_judge_that_favours_the_candidate_breaks_comparability_and_flips_the_decision() -> None:
    old, new = scored(200, seed=1, lenient=0.0, favours_candidate=0.6, gain=0.0)
    bridge = compare_judges(old, new, rules=FAST)
    assert bridge.decision_old.decision == 'INCONCLUSIVE'
    assert bridge.decision_new.decision == 'PROMOTE'
    assert bridge.decisions_differ
    assert bridge.verdict == 'FAIL' and bridge.comparable is False
    assert bridge.interaction.interval[0] > 0.05
    assert bridge.shift_baseline.estimate == 0  # the baseline's outputs are scored exactly as before


def test_a_few_cases_that_agree_prove_nothing() -> None:
    old, new = scored(8, seed=2, lenient=0.0)
    bridge = compare_judges(old, new, rules=FAST)
    assert bridge.interaction.estimate == 0
    assert bridge.verdict == 'UNVALIDATED' and bridge.comparable is None


def test_an_interval_that_contains_zero_is_not_equivalence() -> None:
    """A real but unseen interaction: the interval straddles zero and the margin, so UNVALIDATED, not PASS."""
    old, new = scored(60, seed=3, lenient=0.1, favours_candidate=0.05)
    bridge = compare_judges(old, new, rules=FAST)
    low, high = bridge.interaction.interval
    assert low < 0 < high and high > 0.05
    assert bridge.verdict == 'UNVALIDATED'


def test_equivalence_at_the_margin_is_rarely_claimed() -> None:
    """Sparse, one-sided interaction exactly at the margin: one case in twenty gains a point under the new judge."""
    passes = 0
    for t in range(200):
        rng = random.Random(t)
        flipped = {f'c{i}' for i in range(200) if rng.random() < 0.05}
        old = {'baseline': {f'c{i}': [False] for i in range(200)}, 'candidate': {f'c{i}': [False] for i in range(200)}}
        new = {'baseline': old['baseline'], 'candidate': {n: [n in flipped] for n in old['candidate']}}
        passes += compare_judges(old, new, rules=GateRules(resamples=1000, seed=t)).verdict == 'PASS'
    assert passes / 200 <= 0.05, passes


def test_repeats_are_averaged_within_a_case() -> None:
    old = {'baseline': {'a': [True, False]}, 'candidate': {'a': [True, True]}}
    new = {'baseline': {'a': [True, True]}, 'candidate': {'a': [True, True]}}
    bridge = compare_judges(old, new, rules=FAST)
    assert bridge.cases == 1
    assert bridge.per_case == {'a': -0.5}


@pytest.mark.parametrize(
    ('old', 'new', 'message'),
    [
        ({'baseline': {}, 'candidate': {}}, {'baseline': {}, 'candidate': {}}, 'no cases'),
        ({'baseline': {'a': [True]}, 'candidate': {'a': [True]}}, {'baseline': {'a': [True]}}, 'exactly'),
        (
            {'baseline': {'a': [True]}, 'candidate': {'a': [True]}},
            {'baseline': {'b': [True]}, 'candidate': {'b': [True]}},
            'same cases',
        ),
        (
            {'baseline': {'a': [True]}, 'candidate': {'a': [True]}},
            {'baseline': {'a': [True, True]}, 'candidate': {'a': [True, True]}},
            'same number',
        ),
        (
            {'baseline': {'a': []}, 'candidate': {'a': []}},
            {'baseline': {'a': []}, 'candidate': {'a': []}},
            'at least one',
        ),
    ],
)
def test_invalid_inputs_are_refused(old, new, message) -> None:  # pyright: ignore[reportMissingParameterType, reportUnknownParameterType]
    with pytest.raises(ValueError, match=message):
        compare_judges(old, new)  # pyright: ignore[reportUnknownArgumentType]


def test_margin_must_be_positive() -> None:
    cell = {'a': [True]}
    with pytest.raises(ValueError, match='margin'):
        compare_judges({'baseline': cell, 'candidate': cell}, {'baseline': cell, 'candidate': cell}, margin=0)


@dataclass
class TwoJudges(Evaluator[int, int, None]):
    """A strict and a lenient check of the same output, standing in for an old and a new judge."""

    def evaluate(self, ctx: EvaluatorContext[int, int, None]) -> dict[str, bool]:
        return {'strict': ctx.output == ctx.expected_output, 'lenient': ctx.output % 2 == 0}


DATASET = Dataset[int, int, None](
    name='doubling',
    cases=[Case(name=f'c{i}', inputs=i, expected_output=i * 2) for i in range(40)],
    evaluators=[TwoJudges()],
)


async def test_reports_carrying_both_judges_are_bridged() -> None:
    weak = await DATASET.evaluate(lambda x: x * 2 if x % 2 == 0 else x, repeat=2, progress=False)
    strong = await DATASET.evaluate(lambda x: x * 2, repeat=2, progress=False)
    bridge = compare_judge_reports(weak, strong, old='strict', new='lenient', rules=FAST)
    assert bridge.rates['baseline', 'old'] == 0.5 and bridge.rates['candidate', 'new'] == 1.0
    assert bridge.decision_old.decision == 'PROMOTE'
    assert bridge.cases == 40
