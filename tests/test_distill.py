"""`HybridJudge`: checkable clauses in code, the rest to the model, certified like any judge."""

from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Any

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import EvaluatorContext, LLMJudge

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from distill import CLAUSES, CONTROLS, RUDE  # noqa: E402
from support_task import cases  # noqa: E402

from pydantic_evals_admissibility import JudgeCase, certify_judge, judge_identity  # noqa: E402
from pydantic_evals_admissibility._certify import _context  # noqa: E402
from pydantic_evals_admissibility._distill import (  # noqa: E402
    CODE_PREFIX,
    Clause,
    HybridJudge,
    compare_hybrid,
    model_calls,
    split_rubric,
)


class Model:
    """A scripted model that records every rubric it was asked, and passes polite, non-empty replies.

    `checks_answers=False` makes it a model that cannot do the arithmetic: it never fails a reply
    for its answer, only for its tone.
    """

    def __init__(self, *, checks_answers: bool = True) -> None:
        self.rubrics: list[str] = []
        self.checks_answers = checks_answers
        self.by_inputs = {c.inputs: c for c in cases()}

    def function_model(self) -> FunctionModel:
        from support_task import is_correct

        def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
            prompt = ''.join(
                p.content
                for m in messages
                for p in getattr(m, 'parts', [])
                if isinstance(p, UserPromptPart) and isinstance(p.content, str)
            )
            section = {t: re.search(rf'<{t}>\n(.*?)\n</{t}>', prompt, re.S) for t in ('Input', 'Output', 'Rubric')}
            inputs, output, rubric = (m.group(1) if m else '' for m in section.values())
            self.rubrics.append(rubric)
            passed = bool(output.strip()) and 'bothering us' not in output
            if self.checks_answers and 'correct according to the store policy' in rubric:
                passed = passed and is_correct(self.by_inputs[inputs], output)
            tool = info.output_tools[0]
            return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

        return FunctionModel(model, model_name='scripted')


def support_cases(n: int = 40) -> list[JudgeCase]:
    out = []
    for c in cases()[:n]:
        answer = c.answer if c.answer in ('yes', 'no') else f'${c.answer}'
        reply = f'Thanks for getting in touch! Here is what the policy says for your case.\n\nAnswer: {answer}'
        out.append(JudgeCase(c.name, c.inputs, reply, expected_output=c.answer))
    return out


def ctx(case: JudgeCase, output: Any) -> EvaluatorContext[Any, Any, Any]:
    return _context(case, output)


async def test_a_failed_check_fails_without_a_model_call() -> None:
    model = Model()
    judge = HybridJudge(CLAUSES, model=model.function_model())
    case = support_cases(1)[0]
    result = await judge.evaluate(ctx(case, 'I think so.'))  # no Answer: line
    assert result.value is False and (result.reason or '').startswith(CODE_PREFIX)
    assert 'Answer:' in (result.reason or '') and model.rubrics == []


async def test_the_model_is_asked_only_the_unchecked_clauses() -> None:
    model = Model()
    judge = HybridJudge(CLAUSES, model=model.function_model())
    case = support_cases(1)[0]
    assert (await judge.evaluate(ctx(case, case.output))).value is True
    assert (await judge.evaluate(ctx(case, RUDE + case.output))).value is False
    assert model.rubrics == ['The reply is polite and respectful to the customer.'] * 2


async def test_every_clause_checked_means_no_model_at_all() -> None:
    judge = HybridJudge([Clause('non-empty', lambda c: bool(c.output))], model='test')
    assert judge.model_rubric is None and judge.llm_judge() is None
    case = support_cases(1)[0]
    result = await judge.evaluate(ctx(case, 'x'))
    assert result.value is True and (result.reason or '').startswith(CODE_PREFIX)


def test_split_rubric_attaches_each_check_to_one_clause() -> None:
    def ends_with_answer(c: Any) -> bool:
        return 'Answer:' in c.output

    clauses = split_rubric(
        'The reply ends with an Answer: line. It is polite! It is short.', {'answer: line': ends_with_answer}
    )
    assert [c.text for c in clauses] == ['The reply ends with an Answer: line.', 'It is polite!', 'It is short.']
    assert clauses[0].check is ends_with_answer and clauses[1].check is None
    with pytest.raises(ValueError, match='matches 2 clauses'):
        split_rubric('It is polite. It is short.', {'it is': ends_with_answer})
    with pytest.raises(ValueError, match='matches 0 clauses'):
        split_rubric('It is polite.', {'typo': ends_with_answer})


def test_identity_names_the_checks_not_their_addresses() -> None:
    a, b = HybridJudge(CLAUSES, model='test'), HybridJudge(CLAUSES, model='test')
    assert judge_identity(a) == judge_identity(b)
    text = str(judge_identity(a))
    assert 'answer_is_right' in text and '0x' not in text
    other = HybridJudge(CLAUSES[:2] + (Clause('The reply is short.'),), model='test')
    assert judge_identity(other) != judge_identity(a)


async def test_same_certificate_fewer_calls_same_verdicts() -> None:
    """A model that grades the full rubric correctly: both admissible, and every verdict agrees."""
    model = Model()
    jcs = support_cases()
    hybrid = HybridJudge(CLAUSES, model=model.function_model())
    plain = LLMJudge(rubric=hybrid.full_rubric, model=model.function_model(), include_input=True)
    plain_cert = await certify_judge(plain, jcs, controls=CONTROLS, repeats=2)
    hybrid_cert = await certify_judge(hybrid, jcs, controls=CONTROLS, repeats=2)
    assert plain_cert.verdict == hybrid_cert.verdict == 'ADMISSIBLE', hybrid_cert.table()
    comparison = compare_hybrid(plain_cert, hybrid_cert)
    assert comparison.agreed == comparison.compared == len(plain_cert.judgments)
    # mismatched_output and empty_output are decided in code: 40 + 40 calls not made.
    assert comparison.calls == (model_calls(plain_cert), model_calls(hybrid_cert)) and comparison.calls_saved == 80
    assert all(s == ('PASS', 'PASS') for s in comparison.checks.values())


async def test_a_check_rescues_a_model_that_cannot_do_the_arithmetic() -> None:
    """The model passes any polite reply. Asked the full rubric, it passes other customers' answers;
    with correctness checked in code, the same model is only asked what it can judge."""
    model = Model(checks_answers=False)
    jcs = support_cases()
    hybrid = HybridJudge(CLAUSES, model=model.function_model())
    plain = LLMJudge(rubric=hybrid.full_rubric, model=model.function_model(), include_input=True)
    plain_cert = await certify_judge(plain, jcs, controls=CONTROLS, repeats=1)
    hybrid_cert = await certify_judge(hybrid, jcs, controls=CONTROLS, repeats=1)
    rejection = next(c for c in plain_cert.checks if c.name == 'rejection')
    assert plain_cert.verdict == 'INADMISSIBLE' and 'mismatched_output 0/40 FAIL' in rejection.detail
    assert hybrid_cert.verdict == 'ADMISSIBLE', hybrid_cert.table()
    comparison = compare_hybrid(plain_cert, hybrid_cert)
    assert {role for _, role, _, _ in comparison.disagreements} == {'must_fail:mismatched_output'}


async def test_the_tone_clause_is_still_certified() -> None:
    """Checks cannot hide a model that ignores the clause left to it: rude replies still have to fail."""

    def deaf(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'fine', 'pass': True, 'score': 1.0})])

    cert = await certify_judge(HybridJudge(CLAUSES, model=FunctionModel(deaf)), support_cases(), controls=CONTROLS)
    rejection = next(c for c in cert.checks if c.name == 'rejection')
    assert cert.verdict == 'INADMISSIBLE' and 'rude_tone 0/40 FAIL' in rejection.detail
    assert 'mismatched_output 40/40' in rejection.detail
