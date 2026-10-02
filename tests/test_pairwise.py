"""Pairwise certification on real `PairwiseJudge` calls with scripted models."""

from __future__ import annotations

import re

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

from pydantic_evals_admissibility import PairCase, PairwiseJudge, both_orders, certify_pairwise, first_position_rate

# Even cases are easy (the better answer says 'right'); odd ones are hard (both look plausible).
CASES = [
    PairCase(f'p{i}', f'question {i}', f'right answer {i}' if i % 2 == 0 else f'subtle answer {i}', f'wrong answer {i}')
    for i in range(40)  # 40/40 consistent clears the 0.9 bar; 30/30 cannot (lower bound 0.886)
]


def scripted(decide):  # type: ignore[no-untyped-def]
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(p.content for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart))
        a = re.search(r'<A>\n(.*?)\n</A>', prompt, re.S).group(1)  # type: ignore[union-attr]
        b = re.search(r'<B>\n(.*?)\n</B>', prompt, re.S).group(1)  # type: ignore[union-attr]
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'choice': decide(a, b)})])

    return PairwiseJudge(rubric='Which answer is correct?', model=FunctionModel(model))


def fair(a: str, b: str) -> str:
    return 'B' if b.startswith(('right', 'subtle')) else 'A'


def always_first(a: str, b: str) -> str:
    return 'A'


def first_when_unsure(a: str, b: str) -> str:
    """Right on easy pairs; on hard ones (no 'right' visible) it takes whatever came first."""
    if a.startswith('right'):
        return 'A'
    if b.startswith('right'):
        return 'B'
    return 'A'


async def test_a_fair_judge_is_admissible() -> None:
    cert = await certify_pairwise(scripted(fair), CASES)
    assert cert.verdict == 'ADMISSIBLE', cert.table()
    assert first_position_rate(cert)[0] == 0.5  # type: ignore[index]


async def test_a_judge_that_always_picks_the_first_answer_fails_both_checks() -> None:
    cert = await certify_pairwise(scripted(always_first), CASES)
    assert cert.verdict == 'INADMISSIBLE'
    assert {c.name: c.status for c in cert.checks} == {'accuracy': 'FAIL', 'order_consistency': 'FAIL'}
    consistency = next(c for c in cert.checks if c.name == 'order_consistency')
    assert 'shown first in 40/40' in consistency.detail
    assert first_position_rate(cert)[0] == 1.0  # type: ignore[index]


async def test_position_bias_only_on_hard_pairs_is_caught_and_attributed() -> None:
    cert = await certify_pairwise(scripted(first_when_unsure), CASES)
    consistency = next(c for c in cert.checks if c.name == 'order_consistency')
    assert consistency.successes == 20 and 'shown first in 20/20' in consistency.detail, consistency.detail
    assert cert.verdict == 'INADMISSIBLE'


async def test_both_orders_abstains_where_position_decided() -> None:
    biased = both_orders(scripted(first_when_unsure))
    easy, hard = CASES[0], CASES[1]
    assert await biased(easy.inputs, easy.better, easy.worse) == 'A'
    assert await biased(easy.inputs, easy.worse, easy.better) == 'B'
    assert await biased(hard.inputs, hard.better, hard.worse) is None  # it only ever said "first"
    assert await both_orders(scripted(fair))(hard.inputs, hard.worse, hard.better) == 'B'
