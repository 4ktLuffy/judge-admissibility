"""JudgeCanary on real `LLMJudge` calls (scripted model) through `Dataset.evaluate`."""

from __future__ import annotations

from judges import judge, oracle, yes_man
from pydantic_evals import Case, Dataset

from pydantic_evals_admissibility import CanaryMonitor, JudgeCanary

CAPITALS = {'France': 'Paris', 'Japan': 'Tokyo', 'Kenya': 'Nairobi', 'Peru': 'Lima', 'Egypt': 'Cairo',
            'Norway': 'Oslo', 'Chile': 'Santiago', 'Ghana': 'Accra', 'Spain': 'Madrid', 'Italy': 'Rome'}  # fmt: skip


def dataset(canary: JudgeCanary) -> Dataset[str, str, None]:
    return Dataset[str, str, None](
        name='capitals',
        cases=[Case(name=c, inputs=f'Capital of {c}?', expected_output=city) for c, city in CAPITALS.items()],
        evaluators=[canary],
    )


def answer(question: str) -> str:
    return next(city for c, city in CAPITALS.items() if c in question)


async def test_real_verdicts_pass_through_unchanged_and_a_sound_judge_stays_healthy() -> None:
    monitor = CanaryMonitor(min_checks=10)
    canary = JudgeCanary(judge(oracle), rate=1.0, monitor=monitor, seed=0)
    report = await dataset(canary).evaluate(answer, repeat=3, progress=False)
    assert all(case.assertions['LLMJudge'].value for case in report.cases)  # real verdicts unchanged
    assert monitor.checks == 30 and monitor.rejected == 30
    assert monitor.health() == 'HEALTHY'


async def test_a_judge_that_drifted_into_passing_everything_is_caught() -> None:
    monitor = CanaryMonitor(min_checks=10)
    canary = JudgeCanary(judge(yes_man), rate=1.0, monitor=monitor, seed=0)
    report = await dataset(canary).evaluate(answer, repeat=3, progress=False)
    assert all(case.assertions['LLMJudge'].value for case in report.cases)  # looks perfect
    assert monitor.rejected == 0 and monitor.health() == 'DRIFTING'
    assert not report.cases[0].assertions['judge_canary_rejected'].value


async def test_sampling_rate_limits_extra_judge_calls() -> None:
    monitor = CanaryMonitor()
    canary = JudgeCanary(judge(oracle), rate=0.2, monitor=monitor, seed=1)
    await dataset(canary).evaluate(answer, repeat=10, progress=False)
    assert 8 <= monitor.checks <= 35  # about 20 of 100 calls, not all of them
    assert monitor.health() == 'UNKNOWN' or monitor.checks >= 20


async def test_works_as_an_online_evaluator_on_a_live_function() -> None:
    """The same wrapper on `pydantic_evals.online.evaluate`: live calls, background evaluation."""
    import re

    from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
    from pydantic_ai.models.function import AgentInfo, FunctionModel
    from pydantic_evals.evaluators import LLMJudge
    from pydantic_evals.online import evaluate, wait_for_evaluations

    # Live traffic has no expected answer, so this judge checks the answer against the question.
    def by_question(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(p.content for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart))
        question = re.search(r'<Input>\n(.*?)\n</Input>', prompt, re.S)
        output = re.search(r'<Output>\n(.*?)\n</Output>', prompt, re.S)
        passed = bool(question and output and output.group(1) == answer(question.group(1)))
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

    live_judge = LLMJudge(
        rubric='The output answers the question.', model=FunctionModel(by_question), include_input=True
    )
    monitor = CanaryMonitor(min_checks=10)
    canary = JudgeCanary(live_judge, rate=1.0, monitor=monitor, seed=0)

    @evaluate(canary)
    async def live_answer(question: str) -> str:
        return answer(question)

    for country in CAPITALS:
        assert await live_answer(f'Capital of {country}?') == CAPITALS[country]
    await wait_for_evaluations(timeout=30)
    assert monitor.checks == len(CAPITALS) and monitor.rejected == len(CAPITALS)


async def test_borrowing_is_opt_in() -> None:
    """Off by default: a borrowed answer can be right for another question with the same answer."""
    monitor = CanaryMonitor(min_checks=10)
    canary = JudgeCanary(judge(oracle), rate=1.0, monitor=monitor, seed=0)
    await dataset(canary).evaluate(answer, repeat=2, progress=False)
    assert {h['control'] for h in monitor.history} == {'empty_output'}
    monitor = CanaryMonitor(min_checks=10)
    canary = JudgeCanary(judge(oracle), rate=1.0, monitor=monitor, seed=0, borrow=True)
    await dataset(canary).evaluate(answer, repeat=2, progress=False)
    assert {h['control'] for h in monitor.history} == {'empty_output', 'mismatched_output'}


async def test_systematic_sampling_checks_exactly_every_nth_call() -> None:
    monitor = CanaryMonitor()
    canary = JudgeCanary(judge(oracle), every=4, monitor=monitor)
    await dataset(canary).evaluate(answer, repeat=4, progress=False)  # 40 calls
    assert monitor.checks == 10
