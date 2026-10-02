"""`compare_reports` on real `Dataset.evaluate` reports, with repeats, failures and missing results."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import Evaluator, EvaluatorContext

from pydantic_evals_admissibility import Certificate, Check, compare_reports, outcomes


@dataclass
class Correct(Evaluator[int, int, None]):
    def evaluate(self, ctx: EvaluatorContext[int, int, None]) -> dict[str, bool]:
        return {'correct': ctx.output == ctx.expected_output}


DATASET = Dataset[int, int, None](
    name='doubling',
    cases=[Case(name=f'c{i}', inputs=i, expected_output=i * 2) for i in range(40)],
    evaluators=[Correct()],
)


def doubles(x: int) -> int:
    return x * 2


def doubles_only_evens(x: int) -> int:
    return x * 2 if x % 2 == 0 else x


def crashes_on_threes(x: int) -> int:
    if x % 3 == 0:
        raise RuntimeError('boom')
    return x * 2


async def test_outcomes_group_repeats_by_source_case() -> None:
    report = await DATASET.evaluate(doubles, repeat=3, progress=False)
    seen = outcomes(report, 'correct')
    assert len(seen) == 40 and all(v == [True, True, True] for v in seen.values())


async def test_a_task_failure_counts_as_a_fail_not_a_missing_run() -> None:
    report = await DATASET.evaluate(crashes_on_threes, repeat=2, progress=False)
    seen = outcomes(report, 'correct')
    assert seen['c3'] == [False, False] and seen['c4'] == [True, True]


async def test_a_better_task_is_promoted_and_a_worse_one_rejected() -> None:
    weak = await DATASET.evaluate(doubles_only_evens, repeat=2, progress=False)
    strong = await DATASET.evaluate(doubles, repeat=2, progress=False)
    assert compare_reports(weak, strong, assertion='correct').decision == 'PROMOTE'
    assert compare_reports(strong, weak, assertion='correct').decision == 'REJECT'
    assert compare_reports(strong, strong, assertion='correct').decision == 'INCONCLUSIVE'


async def test_an_uncertified_judge_is_refused_even_when_it_looks_better() -> None:
    weak = await DATASET.evaluate(doubles_only_evens, repeat=2, progress=False)
    strong = await DATASET.evaluate(doubles, repeat=2, progress=False)
    bad = Certificate('INADMISSIBLE', (Check('acceptance', 'FAIL', 0, 40, (0.0, 0.09), 0.7),), ())
    assert compare_reports(weak, strong, assertion='correct', certificate=bad).decision == 'REFUSED'
