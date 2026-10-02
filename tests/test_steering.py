"""`assess_steering`: does a runtime judge's advice help, compared with a content-free message?"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'bench'))

from steering_sim import (  # noqa: E402
    agent,
    attentive_agent,
    correct,
    helpful_judge,
    overeager_judge,
    stale_judge,
    states,
    steps,
)

from pydantic_evals_admissibility import assess_steering  # noqa: E402

SAVED = states()


def kinds(*wanted: str) -> list[str]:
    return [n for n, s in SAVED.items() if s.kind in wanted]


def test_helpful_advice_beats_a_neutral_message() -> None:
    result = assess_steering(SAVED, resume=agent(), outcome=correct, steer=helpful_judge, steps=steps)
    assert result.verdict == 'HELPS' and result.control == 'neutral'
    assert result.comparisons['steer vs neutral'].decision == 'PROMOTE'
    assert result.comparisons['neutral vs silent'].decision == 'INCONCLUSIVE'
    assert set(result.intervened) == set(kinds('needs_correction'))
    assert result.harm is not None and result.harm.decision != 'REJECT' and not result.harmed
    assert result.steps is not None and result.steps['steer'] >= result.steps['silent']


def test_an_overeager_judge_harms_states_already_on_a_good_path_and_says_so() -> None:
    result = assess_steering(SAVED, resume=agent(), outcome=correct, steer=overeager_judge)
    assert result.harm is not None and result.harm.decision == 'REJECT'
    assert set(result.good_path) <= set(kinds('on_track', 'recovered'))
    assert result.harmed and 'good path' in result.reason
    assert 'good path' in result.summary()
    # Its advice fixes the states that needed it, so the average hides the harm.
    assert result.verdict != 'HELPS'


def test_a_stale_judge_never_helps() -> None:
    result = assess_steering(SAVED, resume=agent(), outcome=correct, steer=stale_judge)
    assert set(result.intervened) == set(kinds('recovered'))
    assert result.verdict in ('HURTS', 'INCONCLUSIVE')


def test_only_the_neutral_arm_tells_any_message_from_this_advice() -> None:
    result = assess_steering(SAVED, resume=attentive_agent(), outcome=correct, steer=helpful_judge)
    assert result.comparisons['steer vs silent'].decision == 'PROMOTE'
    assert result.comparisons['neutral vs silent'].decision == 'PROMOTE'
    assert result.verdict == 'INCONCLUSIVE'
    without = assess_steering(
        SAVED, resume=attentive_agent(), outcome=correct, steer=helpful_judge, arms=('silent', 'steer')
    )
    assert without.verdict == 'HELPS' and without.control == 'silent' and 'any message' in without.reason


def test_a_judge_that_never_speaks_changes_nothing() -> None:
    result = assess_steering(SAVED, resume=agent(), outcome=correct, steer=lambda s: None)
    assert result.verdict == 'INCONCLUSIVE' and not result.intervened
    assert result.completion['silent'] == result.completion['steer'] == result.completion['neutral']


def test_the_neutral_message_goes_only_where_the_judge_spoke() -> None:
    seen: list[str | None] = []

    def resume(state: object, message: str | None, rng: random.Random) -> bool:
        seen.append(message)
        return True

    assess_steering(
        {'a': 1, 'b': 2},
        resume=resume,
        outcome=lambda t: t,
        steer=lambda s: 'advice' if s == 1 else None,
        neutral=lambda advice: 'x' * len(advice),
        repeats=1,
    )
    assert seen == [None, 'advice', 'xxxxxx', None, None, None]


def test_input_is_validated() -> None:
    def run(**kwargs: object) -> None:
        args: dict[str, object] = {'resume': lambda s, m, r: True, 'outcome': lambda t: t, 'steer': lambda s: 'go'}
        args.update(kwargs)
        assess_steering(args.pop('states', {'a': 1}), **args)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match='no states'):
        run(states={})
    with pytest.raises(ValueError, match='repeats'):
        run(repeats=0)
    with pytest.raises(ValueError, match="include 'steer'"):
        run(arms=('silent', 'neutral'))
    with pytest.raises(ValueError, match="include 'steer'"):
        run(arms=('steer',))
    with pytest.raises(ValueError, match='distinct'):
        run(arms=('steer', 'steer'))
    with pytest.raises(ValueError, match='distinct'):
        run(arms=('steer', 'loud'))
    with pytest.raises(ValueError, match='unknown states'):
        run(good_path=['z'])
    with pytest.raises(ValueError, match='good_path_rate'):
        run(good_path_rate=1.5)
    with pytest.raises(TypeError, match='bool'):
        run(outcome=lambda t: 1)
