"""`EvidenceRewrite`: change the evidence, keep the answer, and see whether the judge notices."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from evidence_task import CONTROLS, RUBRIC, episodes  # noqa: E402

from pydantic_evals_admissibility import certify_judge  # noqa: E402


def scripted(decide):  # type: ignore[no-untyped-def]
    """A judge whose model reads the `<Input>` and `<Output>` sections of LLMJudge's real prompt."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(p.content for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart))
        found = re.search(r'<Input>\n(.*?)\n</Input>', prompt, re.S)
        output = re.search(r'<Output>\n(.*?)\n</Output>', prompt, re.S)
        passed = decide(json.loads(found.group(1)) if found else None, output.group(1) if output else '')
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

    return FunctionModel(model)


def grades_the_work(inputs, reply: str) -> bool:  # type: ignore[no-untyped-def]
    """Passes a reply only if the action succeeded and, for refunds, the amount matches."""
    if not reply.strip() or inputs is None:
        return False
    action = inputs['tool_calls'][-1]['result']
    if action['status'] not in ('succeeded', 'cancelled'):
        return False
    return 'refunded' not in action or f'${action["refunded"]}' in reply


def grades_the_claim(inputs, reply: str) -> bool:  # type: ignore[no-untyped-def]
    """Reads only the reply: passes any confident, non-empty one."""
    return bool(reply.strip())


def superstitious(inputs, reply: str) -> bool:  # type: ignore[no-untyped-def]
    """Right about the work, but fails replies whose trace has a span id ending in a digit below 5."""
    ids = [c['span_id'] for c in inputs['tool_calls']] if inputs else []
    return grades_the_work(inputs, reply) and not any(i[-1] in '01234' for i in ids)


async def test_a_judge_that_reads_the_tool_results_is_admissible() -> None:
    judge = LLMJudge(rubric=RUBRIC, model=scripted(grades_the_work), include_input=True)
    cert = await certify_judge(judge, episodes(40), controls=CONTROLS, repeats=1)
    assert cert.verdict == 'ADMISSIBLE', cert.table()


async def test_a_judge_that_grades_only_the_claim_fails_the_evidence_controls() -> None:
    """The same reply, a failed refund: a judge that grades the claim passes it anyway."""
    judge = LLMJudge(rubric=RUBRIC, model=scripted(grades_the_claim), include_input=True)
    cert = await certify_judge(judge, episodes(40), controls=CONTROLS, repeats=1)
    rejection = next(c for c in cert.checks if c.name == 'rejection')
    assert cert.verdict == 'INADMISSIBLE' and 'tool_failed 0/40 FAIL' in rejection.detail, cert.table()
    assert 'empty_output 40/40' in rejection.detail  # not a constant judge: it does fail empty replies


async def test_a_judge_swayed_by_irrelevant_ids_fails_invariance() -> None:
    judge = LLMJudge(rubric=RUBRIC, model=scripted(superstitious), include_input=True)
    cert = await certify_judge(judge, episodes(40), controls=CONTROLS, repeats=1)
    invariance = next(c for c in cert.checks if c.name == 'invariance')
    assert invariance.status != 'PASS' and 'ids_changed' in invariance.detail, cert.table()


def test_controls_never_touch_the_reply() -> None:
    import random

    case = episodes(2)[0]
    for control in CONTROLS[:3]:
        changed = control.make_case(case, [case], random.Random(0))  # type: ignore[attr-defined]
        assert changed is not None and changed.output == case.output and changed.inputs != case.inputs


async def test_the_doctor_says_the_judge_grades_the_claim() -> None:
    from pydantic_evals_admissibility import diagnose

    blind = LLMJudge(rubric=RUBRIC, model=scripted(grades_the_claim))
    cert = await certify_judge(blind, episodes(40), controls=CONTROLS, repeats=1)
    assert any('grades the claim, not the work' in a and 'include_input=True' in a for a in diagnose(cert, blind))

    shown = LLMJudge(rubric=RUBRIC, model=scripted(grades_the_claim), include_input=True)
    cert = await certify_judge(shown, episodes(40), controls=CONTROLS, repeats=1)
    advice = diagnose(cert, shown)
    assert any('shown the evidence and still passes' in a for a in advice), advice
    assert not any("another question's answer" in a for a in advice)  # mismatched_output was not a failed family
