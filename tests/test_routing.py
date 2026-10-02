"""`RoutedJudge` and `certify_routing`: a cheap judge where it agrees with itself, an expensive one elsewhere."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pytest
from judges import judge, oracle, yes_man
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from pydantic_evals_admissibility import JudgeCase, MismatchedOutput, WhitespaceReformat, certify_judge
from pydantic_evals_admissibility._routing import (
    RoutedJudge,
    certify_routing,
    replay_routing,
    route_of,
    summarize_routing,
    truth_of,
    unanimous,
)

CASES = [JudgeCase(f'q{i}', f'What is {i} + {i}?', f'{i} + {i} = {2 * i}.', f'= {2 * i}.') for i in range(20)]
CONTROLS = (MismatchedOutput(), WhitespaceReformat())


@dataclass
class Counted(Evaluator[Any, Any, Any]):
    """Counts the calls made to the judge it wraps."""

    judge: Evaluator[Any, Any, Any]
    calls: list[int] = field(default_factory=list)

    async def evaluate(self, ctx: EvaluatorContext[Any, Any, Any]) -> Any:
        self.calls.append(1)
        return await self.judge.evaluate_async(ctx)


def flaky_on_hard_cases() -> Any:
    """Right on most answers; on answers to every fourth question it flips on each call."""
    flips: dict[str, bool] = {}

    def decide(output: str, expected: str) -> bool:
        if int(expected.strip('=. ')) % 8 != 0:
            return oracle(output, expected)
        flips[output] = not flips.get(output, False)
        return flips[output]

    return decide


def test_unanimous_needs_every_sample_and_no_errors() -> None:
    assert unanimous([True, True]) is True and unanimous([False, False, False]) is False
    assert unanimous([True, False]) is None and unanimous([True, None]) is None and unanimous([]) is None


async def test_a_consistent_cheap_judge_never_escalates_and_never_calls_the_expensive_one() -> None:
    cheap, expensive = Counted(judge(oracle)), Counted(judge(oracle))
    routed = await certify_routing(cheap, expensive, CASES, k=2, controls=CONTROLS, repeats=1)
    assert routed.certificate.verdict == 'ADMISSIBLE', routed.table()
    assert routed.summary.escalated == 0 and not expensive.calls
    assert len(cheap.calls) == 2 * 60  # 20 known-good, 20 mismatched, 20 reformatted; k=2 each
    assert routed.summary.relative_cost == 2.0  # two cheap calls at the same price as one expensive


async def test_escalation_fixes_what_the_cheap_judge_is_unsure_of() -> None:
    cheap, expensive = Counted(judge(flaky_on_hard_cases())), Counted(judge(oracle))
    routed = await certify_routing(cheap, expensive, CASES, k=2, controls=CONTROLS, repeats=1)
    s = routed.summary
    # q0, q4, q8, q12, q16 have answers divisible by 8: their known-good answers flip-flop, and so
    # do their reformatted versions and the mismatched controls that borrow them.
    assert s.escalated == len(expensive.calls) > 0
    assert s.routed_correct == s.judgments, routed.table()
    assert s.cheap_correct < s.judgments and s.escalated_cheap_wrong > 0 and s.confidently_wrong == 0
    assert all(route_of(j.reason) is not None for j in routed.certificate.judgments)


async def test_a_confidently_wrong_cheap_judge_is_never_escalated_and_the_policy_fails() -> None:
    """Self-consistency is not correctness: a judge that passes everything agrees with itself."""
    expensive = Counted(judge(oracle))
    routed = await certify_routing(judge(yes_man), expensive, CASES, k=3, controls=CONTROLS, repeats=1)
    assert routed.summary.escalated == 0 and not expensive.calls
    assert routed.summary.confidently_wrong == 20  # every mismatched control
    assert routed.certificate.verdict == 'INADMISSIBLE', routed.table()


async def test_with_no_expensive_judge_uncertain_judgments_go_to_a_person_not_to_a_verdict() -> None:
    routed = await certify_routing(judge(flaky_on_hard_cases()), None, CASES, k=2, controls=CONTROLS, repeats=1)
    escalated = [j for j in routed.certificate.judgments if j.error and 'Escalated' in j.error]
    assert escalated and all(j.passed is None for j in escalated)
    assert routed.summary.unresolved == routed.summary.escalated == len(escalated)


async def test_the_expensive_baseline_is_matched_judgment_by_judgment() -> None:
    expensive = judge(oracle)
    alone = await certify_judge(expensive, CASES, controls=CONTROLS, repeats=1)
    routed = await certify_routing(
        judge(flaky_on_hard_cases()), expensive, CASES, k=2, controls=CONTROLS, repeats=1,
        baselines={'expensive alone': alone}, expensive_baseline='expensive alone', expensive_cost=10.0,
    )  # fmt: skip
    s = routed.summary
    assert s.expensive_correct == s.expensive_trials == s.judgments == 60
    assert s.relative_cost == pytest.approx((2 * 1.0 + s.escalation_rate * 10.0) / 10.0)
    again = summarize_routing(routed.certificate, k=2, expensive_alone=alone, expensive_cost=10.0)
    assert again.to_dict() == s.to_dict()


def test_replay_counts_what_routing_caught_and_what_it_could_not() -> None:
    cheap = {'a': [True, True], 'b': [True, False], 'c': [False, False], 'd': [False, True]}
    expensive = {'a': True, 'b': False, 'c': True, 'd': True}
    truth = {'a': True, 'b': False, 'c': True, 'd': True}
    s = replay_routing(cheap, expensive, truth)
    assert (s.judgments, s.escalated, s.routed_correct, s.cheap_correct) == (4, 2, 3, 1)
    assert s.confidently_wrong == 1  # 'c': wrong twice, kept
    assert s.escalated_cheap_wrong == 2 and s.expensive_correct == 4


def test_roles_imply_the_right_verdict_and_repeats_do_not_count_twice() -> None:
    assert truth_of('reference#0') is True and truth_of('reference#1') is None
    assert truth_of('must_fail:empty_output') is False and truth_of('must_hold:whitespace_reformat') is True
    assert truth_of('human:0') is False and truth_of('human:1') is True


def test_k_must_be_positive() -> None:
    with pytest.raises(ValueError):
        RoutedJudge(judge(oracle), None, k=0)
