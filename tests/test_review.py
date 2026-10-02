"""`ReviewPlan`: label cases to settle one release decision, grouped by what the judge said."""

from __future__ import annotations

import random

import pytest

from pydantic_evals_admissibility import ReviewPlan


def population(n: int, gain: float, error: float, seed: int):  # type: ignore[no-untyped-def]
    rng = random.Random(seed)
    labels, judge_b, judge_c = {}, {}, {}
    for i in range(n):
        b, c = rng.random() < 0.6 - gain / 2, rng.random() < 0.6 + gain / 2
        labels[f'c{i}'] = ([b], [c])
        judge_b[f'c{i}'] = [b if rng.random() > error else not b]
        judge_c[f'c{i}'] = [c if rng.random() > error else not c]
    return labels, judge_b, judge_c


def settle(plan: ReviewPlan, labels):  # type: ignore[no-untyped-def]
    while batch := plan.next_batch():
        plan.add_labels({case: labels[case] for case in batch})
        if plan.decision().decision != 'INCONCLUSIVE':
            break
    return plan.decision()


def test_every_group_the_judge_made_is_sampled() -> None:
    labels, judge_b, judge_c = population(200, 0.2, 0.1, 0)
    plan = ReviewPlan.from_verdicts(judge_b, judge_c, looks=(12, 24))
    plan.add_labels({c: labels[c] for c in plan.next_batch()})
    counts = {stratum: labelled for stratum, (_, labelled, _) in plan.decision().by_stratum.items()}
    assert len(counts) == 3 and all(n >= 2 for n in counts.values()) and sum(counts.values()) == 12


def test_a_real_gain_is_promoted_and_no_gain_is_rarely_called() -> None:
    promoted = sum(
        settle(
            ReviewPlan.from_verdicts(*population(200, 0.3, 0.05, s)[1:], looks=(12, 24, 48, 96), seed=s),
            population(200, 0.3, 0.05, s)[0],
        ).decision
        == 'PROMOTE'
        for s in range(40)
    )
    assert promoted >= 36
    wrong = 0
    for s in range(200):
        labels, judge_b, judge_c = population(200, 0.0, 0.2, s)
        actual = sum(c[0] - b[0] for b, c in labels.values())
        result = settle(ReviewPlan.from_verdicts(judge_b, judge_c, seed=s), labels)
        wrong += (result.decision == 'PROMOTE' and actual <= 0) or (result.decision == 'REJECT' and actual >= 0)
    assert wrong <= 5  # 2.5% of 200, the one-sided budget


def test_labelling_every_case_gives_the_exact_answer() -> None:
    labels, judge_b, judge_c = population(30, 0.3, 0.2, 3)
    plan = ReviewPlan.from_verdicts(judge_b, judge_c, looks=(10, 30))
    plan.add_labels(labels)
    result = plan.decision()
    actual = sum(c[0] - b[0] for b, c in labels.values()) / 30
    assert result.estimate == pytest.approx(actual) and result.interval == pytest.approx((actual, actual))


def test_the_plan_is_fixed_by_its_seed_and_refuses_bad_input() -> None:
    _, judge_b, judge_c = population(100, 0.1, 0.1, 5)
    assert (
        ReviewPlan.from_verdicts(judge_b, judge_c, seed=1).next_batch()
        == ReviewPlan.from_verdicts(judge_b, judge_c, seed=1).next_batch()
    )
    with pytest.raises(ValueError, match='two in every group'):
        ReviewPlan.from_verdicts(judge_b, judge_c, looks=(4,))
    with pytest.raises(ValueError, match='same'):
        ReviewPlan.from_verdicts(judge_b, {})
    with pytest.raises(KeyError):
        ReviewPlan.from_verdicts(judge_b, judge_c).add_labels({'nope': ([True], [True])})
