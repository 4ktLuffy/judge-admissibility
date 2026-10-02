"""`AbstainingJudge` and `certify_abstention`, on real Pydantic AI calls with scripted models."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from cited_and_abstain import evidence_removed  # noqa: E402
from evidence_task import FAILURES, RUBRIC, episodes, tool_failed  # noqa: E402

from pydantic_evals_admissibility._abstain import AbstainingJudge, certify_abstention  # noqa: E402
from pydantic_evals_admissibility._cases import JudgeCase  # noqa: E402

GOOD = episodes(20)
BAD = [JudgeCase(f'{c.name}/tool_failed', tool_failed(c.inputs), c.output) for c in GOOD]
MISSING = [JudgeCase(f'{c.name}/evidence_removed', evidence_removed(c.inputs), c.output) for c in GOOD]


def _inputs(messages: list[ModelMessage]) -> dict[str, Any]:
    prompt = ''.join(p.content for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart))
    return json.loads(re.search(r'<Input>\n(.*?)\n</Input>', prompt, re.S).group(1))  # type: ignore[union-attr]


def scripted(policy):  # type: ignore[no-untyped-def]
    """`policy(result or None)` returns (decision, abstain_reason) from the action's tool result."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        action = next(c for c in _inputs(messages)['tool_calls'] if c['tool'] in FAILURES)
        decision, why = policy(action.get('result'))
        args = {'reason': 'scripted', 'decision': decision, 'abstain_reason': why}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return AbstainingJudge(rubric=RUBRIC, model=FunctionModel(model))


def careful(result: dict[str, Any] | None) -> tuple[str, str | None]:
    if result is None:
        return 'abstain', 'missing_evidence'
    return ('pass' if result['status'] != 'failed' else 'fail'), None


def guesses(result: dict[str, Any] | None) -> tuple[str, str | None]:
    return ('pass', None) if result is None else careful(result)


def abstains_on_failures(result: dict[str, Any] | None) -> tuple[str, str | None]:
    """Abstains whenever the news is bad, too: right on missing evidence, but it ducks decisions."""
    if result is None or result['status'] == 'failed':
        return 'abstain', 'missing_evidence'
    return 'pass', None


def wrong_reason(result: dict[str, Any] | None) -> tuple[str, str | None]:
    return ('abstain', 'ambiguous_rubric') if result is None else careful(result)


async def test_a_judge_that_abstains_exactly_on_missing_evidence_is_admissible() -> None:
    cert = await certify_abstention(scripted(careful), GOOD, must_fail=BAD, must_abstain=MISSING)
    assert cert.verdict == 'ADMISSIBLE', cert.table()
    assert cert.check('abstention_recall').successes == 20


async def test_a_judge_that_guesses_on_missing_evidence_fails_recall() -> None:
    cert = await certify_abstention(scripted(guesses), GOOD, must_fail=BAD, must_abstain=MISSING)
    recall = cert.check('abstention_recall')
    assert recall.status == 'FAIL' and 'guessed instead: 20 pass, 0 fail' in recall.detail
    assert cert.check('decided_accuracy').status == 'PASS'


async def test_a_judge_that_ducks_hard_decisions_fails_coverage_and_precision() -> None:
    cert = await certify_abstention(scripted(abstains_on_failures), GOOD, must_fail=BAD, must_abstain=MISSING)
    assert cert.check('coverage').status == 'FAIL', cert.table()  # 20 of 40 answerable decided
    assert cert.check('abstention_precision').successes == 20 and cert.check('abstention_precision').trials == 40
    assert cert.check('abstention_recall').status == 'PASS'
    assert cert.verdict == 'INADMISSIBLE'


async def test_the_wrong_abstain_reason_does_not_count() -> None:
    cert = await certify_abstention(scripted(wrong_reason), GOOD, must_fail=BAD, must_abstain=MISSING)
    recall = cert.check('abstention_recall')
    assert recall.successes == 0 and '20 abstained for another reason' in recall.detail
    loose = await certify_abstention(
        scripted(wrong_reason), GOOD, must_fail=BAD, must_abstain=MISSING, abstain_reason=None
    )
    assert loose.check('abstention_recall').successes == 20


async def test_a_plain_llm_judge_must_guess() -> None:
    def passes_if_not_failed(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        action = next(c for c in _inputs(messages)['tool_calls'] if c['tool'] in FAILURES)
        passed = action.get('result', {}).get('status') != 'failed'
        args = {'reason': 'scripted', 'pass': passed, 'score': 1.0}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    judge = LLMJudge(rubric=RUBRIC, model=FunctionModel(passes_if_not_failed), include_input=True)
    cert = await certify_abstention(judge, GOOD, must_fail=BAD, must_abstain=MISSING)
    assert cert.check('coverage').successes == 40 and cert.check('decided_accuracy').successes == 40
    assert cert.check('abstention_recall').status == 'FAIL'
    assert cert.check('abstention_precision').trials == 0  # it never abstained
    assert cert.verdict == 'INADMISSIBLE'


async def test_errors_are_neither_verdicts_nor_abstentions() -> None:
    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('down')

    cert = await certify_abstention(AbstainingJudge(RUBRIC, FunctionModel(broken)), GOOD, must_abstain=MISSING)
    assert cert.verdict == 'UNVALIDATED'
    assert all(d.decision is None and d.error for d in cert.decisions)
