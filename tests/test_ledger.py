"""The feedback ledger: cases an optimizer has seen feedback on cannot confirm its result, and confirming on them
promotes noise more often than the gate's level says."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from pydantic_evals_admissibility import ExposedCases, FeedbackLedger, GateRules

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from ledger_demo import simulate  # noqa: E402

FAST = GateRules(resamples=2000)


def ledger() -> FeedbackLedger:
    book = FeedbackLedger()
    book.record(['a', 'b'], by='round 1')
    book.record(['b', 'c', 'c'], by='round 2')
    return book


def test_exposures_are_counted_per_round_and_fresh_filters() -> None:
    book = ledger()
    assert [book.exposures(c) for c in 'abcd'] == [1, 2, 1, 0]
    assert book.exposed_by('b') == ['round 1', 'round 2']
    assert book.fresh(['d', 'a', 'b', 'e']) == ['d', 'e']
    assert book.fresh(['d', 'a', 'b', 'e'], max_exposures=1) == ['d', 'a', 'e']


def test_confirmation_on_exposed_cases_is_refused_naming_them() -> None:
    book = ledger()
    book.check_confirmation(['d', 'e'])
    with pytest.raises(ExposedCases, match="'b' \\(by round 1, round 2\\)") as raised:
        book.check_confirmation(['b', 'd'])
    assert raised.value.exposed == {'b': ['round 1', 'round 2']}

    better = {'a': [True], 'd': [True]}
    worse = {'a': [False], 'd': [False]}
    refused = book.confirm(worse, better, rules=FAST)
    assert refused.decision == 'REFUSED' and "'a' (by round 1)" in refused.reason and "'d'" not in refused.reason
    assert book.confirm({'d': [False]}, {'d': [True]}, rules=FAST).decision == 'INCONCLUSIVE'  # fresh: decided


def test_the_ledger_round_trips_through_json() -> None:
    book = ledger()
    again = FeedbackLedger.from_dict(json.loads(json.dumps(book.to_dict())))
    assert again == book
    assert again.exposures('b') == 2


def test_picking_the_best_of_many_and_confirming_on_the_same_cases_promotes_noise() -> None:
    rates = simulate(20, 150, seed=1, resamples=500)
    budget = GateRules().level / 2
    assert rates['exposed'] > 3 * budget
    assert rates['fresh'] <= 2 * budget  # within the budget, give or take 150 trials of noise
