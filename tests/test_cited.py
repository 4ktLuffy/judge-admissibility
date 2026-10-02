"""`CitedJudge` and `verify_citations`, on real Pydantic AI calls with scripted models."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any

from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from evidence_task import FAILURES, RUBRIC, episodes, tool_failed  # noqa: E402

from pydantic_evals_admissibility._cases import JudgeCase  # noqa: E402
from pydantic_evals_admissibility._certify import certify_judge  # noqa: E402
from pydantic_evals_admissibility._cited import (  # noqa: E402
    CitedJudge,
    CitedVerdict,
    certify_citations,
    evidence_items,
    verify_citations,
)

GOOD = episodes(24)
BAD = [JudgeCase(f'{c.name}/tool_failed', tool_failed(c.inputs), c.output, metadata=c.metadata) for c in GOOD]


def action_facts(case: JudgeCase) -> list[str]:
    """The verdict depends on the action's tool call: the refund or the cancellation."""
    return [next(c['tool'] for c in case.inputs['tool_calls'] if c['tool'] in FAILURES)]


def scripted(cite):  # type: ignore[no-untyped-def]
    """A judge right on the verdict; `cite(calls)` decides which ids it cites."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(p.content for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart))
        inputs = json.loads(re.search(r'<Input>\n(.*?)\n</Input>', prompt, re.S).group(1))  # type: ignore[union-attr]
        action = next(c for c in inputs['tool_calls'] if c['tool'] in FAILURES)
        passed = action['result']['status'] in ('succeeded', 'cancelled')
        args = {'reason': 'scripted', 'citations': cite(inputs['tool_calls']), 'pass': passed}
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, args)])

    return CitedJudge(rubric=RUBRIC, model=FunctionModel(model))


def cites_the_action(calls: list[dict[str, Any]]) -> list[str]:
    return [c['span_id'] for c in calls if c['tool'] in FAILURES]


def fabricates(calls: list[dict[str, Any]]) -> list[str]:
    return ['span_0000000000']


def cites_the_lookup(calls: list[dict[str, Any]]) -> list[str]:
    return [c['span_id'] for c in calls if c['tool'] not in FAILURES]


def test_evidence_items_are_found_by_id_at_any_depth() -> None:
    items = evidence_items(GOOD[0].inputs)
    assert len(items) == 2 and all(k.startswith('span_') for k in items)


def test_verify_citations_on_one_verdict() -> None:
    case = GOOD[0]
    action, lookup = cites_the_action(case.inputs['tool_calls'])[0], cites_the_lookup(case.inputs['tool_calls'])[0]
    good = verify_citations(case, CitedVerdict(reason='r', citations=[action], pass_=True), facts=action_facts)
    assert good.valid and good.supported and not good.unknown
    made_up = verify_citations(case, CitedVerdict(reason='r', citations=[action, 'span_x'], pass_=True), facts=['x'])
    assert made_up.valid is False and made_up.unknown == ('span_x',)
    off_topic = verify_citations(case, CitedVerdict(reason='r', citations=[lookup], pass_=True), facts=action_facts)
    assert off_topic.valid and off_topic.supported is False
    silent = verify_citations(case, CitedVerdict(reason='r', citations=[], pass_=True), facts=action_facts)
    assert silent.valid is False and silent.supported is False  # citing nothing is not "all real"
    custom = verify_citations(case, CitedVerdict(reason='r', citations=[lookup], pass_=True), supports=lambda *_: True)
    assert custom.supported is True


async def test_a_judge_citing_the_action_is_admissible() -> None:
    cert = await certify_citations(scripted(cites_the_action), GOOD, must_fail=BAD, facts=action_facts)
    assert cert.verdict == 'ADMISSIBLE', cert.table()


async def test_fabricated_ids_are_caught() -> None:
    cert = await certify_citations(scripted(fabricates), GOOD, must_fail=BAD, facts=action_facts)
    statuses = {c.name: c.status for c in cert.certificate.checks}
    assert statuses['citation_validity'] == 'FAIL' and statuses['acceptance'] == 'PASS', cert.table()
    assert cert.verdict == 'INADMISSIBLE'
    assert all(c.unknown == ('span_0000000000',) for c in cert.citations)


async def test_citing_irrelevant_spans_is_caught() -> None:
    cert = await certify_citations(scripted(cites_the_lookup), GOOD, must_fail=BAD, facts=action_facts)
    statuses = {c.name: c.status for c in cert.certificate.checks}
    assert statuses == {
        'acceptance': 'PASS',
        'rejection': 'PASS',
        'citation_validity': 'PASS',
        'citation_support': 'FAIL',
    }


async def test_it_is_an_evaluator_too() -> None:
    cert = await certify_judge(scripted(cites_the_action), GOOD[:12], controls=(), repeats=1)
    assert cert.checks[0].successes == 12
    assert '[cites: span_' in (cert.judgments[0].reason or '')


async def test_an_erroring_judge_has_no_valid_citations() -> None:
    def broken(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        raise RuntimeError('down')

    cert = await certify_citations(CitedJudge(RUBRIC, FunctionModel(broken)), GOOD[:12], facts=action_facts)
    assert cert.verdict == 'UNVALIDATED'
    assert all(c.valid is None and c.error for c in cert.citations)
