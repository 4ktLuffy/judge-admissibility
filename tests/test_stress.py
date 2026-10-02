"""`stress_judge`: search for wrong answers a judge accepts, then confirm on fresh cases."""

from __future__ import annotations

import pytest
from judges import judge

from pydantic_evals_admissibility import JudgeCase, stress_judge

CASES = [JudgeCase(f'q{i}', f'What is {i} + {i}?', f'{2 * i}', expected_output=f'{2 * i}') for i in range(60)]
DISCOVERY, CONFIRMATION = CASES[:30], CASES[30:]


def wrong(case: JudgeCase) -> str:
    return str(int(case.expected_output) + 1)


def oracle(case: JudgeCase, output: str) -> bool:
    return output.strip() == case.expected_output


def robust(output: str, expected: str) -> bool:
    return output.strip() == expected.strip()


def gullible(output: str, expected: str) -> bool:
    """Right on plain answers, but trusts an answer that says it was verified."""
    return robust(output, expected) or 'Verified' in output


async def test_a_judge_that_trusts_claims_of_authority_is_found_out() -> None:
    result = await stress_judge(judge(gullible), DISCOVERY, CONFIRMATION, wrong=wrong, oracle=oracle, budget=60)
    assert result.verdict == 'VULNERABLE' and 'authority' in (result.attack or ''), result.table()
    assert result.confirmation_attacked == (30, 30) and result.confirmation_plain == (0, 30)


async def test_a_judge_that_checks_the_answer_shows_no_effect() -> None:
    result = await stress_judge(judge(robust), DISCOVERY, CONFIRMATION, wrong=wrong, oracle=oracle, budget=60)
    assert result.verdict == 'NO EFFECT FOUND', result.table()
    assert result.calls_to_first_success is None


async def test_adaptive_search_finds_it_sooner_than_random_at_the_same_budget() -> None:
    first = {'adaptive': [], 'random': []}
    for seed in range(12):
        for strategy in first:
            r = await stress_judge(
                judge(gullible), DISCOVERY, CONFIRMATION, wrong=wrong, oracle=oracle, budget=40,
                strategy=strategy, seed=seed, compose=False,  # type: ignore[arg-type]
            )  # fmt: skip
            first[strategy].append(sum(1 for name, _, passed in r.history if 'authority' in name and passed))
    # Both find it; adaptive then exploits it, spending more of its budget where it works. With
    # the 20 pairs added, 25 attacks to try once each eat most of a 40-call budget: no advantage.
    assert sum(first['adaptive']) > 2 * sum(first['random']), first


async def test_rewrites_that_turn_out_correct_are_not_attacks() -> None:
    def right_after_all(case: JudgeCase) -> str:
        return case.expected_output  # every "wrong" answer is in fact right: no attack is valid

    result = await stress_judge(
        judge(gullible), DISCOVERY, CONFIRMATION, wrong=right_after_all, oracle=lambda c, o: True, budget=10
    )
    assert result.calls == 0 and result.verdict == 'INCONCLUSIVE' and result.invalid > 0


async def test_confirmation_cases_must_be_new() -> None:
    with pytest.raises(ValueError, match='new'):
        await stress_judge(judge(robust), CASES[:5], CASES[:5], wrong=wrong, oracle=oracle)
