"""How much of a support trace does a judge need? `evidence_budget` on the evidence-task episodes.

    PYTHONPATH=.:bench <venv>/bin/python bench/evidence_budget.py [--codex]

Each episode (`bench/evidence_task.py`) is a customer request, a lookup tool call, an action tool
call (refund or cancel) and the agent's reply. Its components: the customer message, and each
call's span id, arguments and result (tool names always stay). The budget is fitted on the
known-good episodes plus the evidence controls (`tool_failed`, `amount_differs`: must fail;
`ids_changed`: must hold), then checked on episodes it never saw: verdicts on full and on
projected inputs, judged separately.

Offline, four scripted `LLMJudge`s through LLMJudge's real prompt: one that grades the action's
result, one that also (wrongly) reads span ids, one that grades the result but passes the reply
when shown no result, and one that grades only the reply. Each is fitted twice: with the controls,
and on the known-good episodes alone, to show what the controls are for. With `--codex`, a
real `LLMJudge` on Codex `gpt-5.6-luna`, no reasoning, shown the inputs, on 2 episodes (fit) and
2 others (check), within a fixed call budget.
"""

from __future__ import annotations

import asyncio
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any

from evidence_task import CONTROLS, FAILURES, RUBRIC, episodes
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility._evidence_budget import check_budget, evidence_budget

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'evidence_budget.json'
EVIDENCE_CONTROLS = CONTROLS[:3]  # tool_failed, amount_differs, ids_changed; an empty reply needs no evidence
FIELDS = ('span_id', 'arguments', 'result')


def _role(call: dict[str, Any]) -> str:
    return 'action' if call['tool'] in FAILURES else 'lookup'


def parts(inputs: dict[str, Any]) -> list[str]:
    """`customer`, then `lookup.<field>` and `action.<field>` for each field the call has."""
    names = ['customer'] if 'customer' in inputs else []
    for call in inputs['tool_calls']:
        names += [f'{_role(call)}.{f}' for f in FIELDS if f in call]
    return names


def project(inputs: dict[str, Any], kept: frozenset[str]) -> dict[str, Any]:
    out = copy.deepcopy(inputs)
    if 'customer' not in kept:
        out.pop('customer', None)
    for call in out['tool_calls']:
        for f in FIELDS:
            if f'{_role(call)}.{f}' not in kept:
                call.pop(f, None)
    return out


def scripted(decide: Any) -> LLMJudge:
    """An `LLMJudge` whose model reads the `<Input>` (as JSON) and `<Output>` of LLMJudge's real prompt."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(
            p.content
            for m in messages
            for p in getattr(m, 'parts', [])
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )
        found = re.search(r'<Input>\n(.*?)\n</Input>', prompt, re.S)
        output = re.search(r'<Output>\n(.*?)\n</Output>', prompt, re.S)
        passed = decide(json.loads(found.group(1)) if found else {}, output.group(1) if output else '')
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

    return LLMJudge(rubric=RUBRIC, model=FunctionModel(model), include_input=True)


def _action(inputs: dict[str, Any]) -> dict[str, Any]:
    return next((c for c in inputs.get('tool_calls', []) if c['tool'] in FAILURES), {})


def grades_the_work(inputs: dict[str, Any], reply: str) -> bool:
    """Passes a reply only if the action's result says it happened, for the amount the reply states."""
    result = _action(inputs).get('result')
    if not reply.strip() or result is None:
        return False  # no evidence, no pass
    if result.get('status') not in ('succeeded', 'cancelled'):
        return False
    return 'refunded' not in result or f'${result["refunded"]}' in reply


def reads_span_ids(inputs: dict[str, Any], reply: str) -> bool:
    """Right about the work, but fails a trace whose action span id ends in a digit below 5."""
    span = _action(inputs).get('span_id', '')
    return grades_the_work(inputs, reply) and not (span and span[-1] in '01234')


def trusts_the_claim_without_evidence(inputs: dict[str, Any], reply: str) -> bool:
    """Grades the action's result when shown it; shown none, passes the confident reply."""
    return grades_the_work(inputs, reply) if 'result' in _action(inputs) else bool(reply.strip())


def grades_the_claim(inputs: dict[str, Any], reply: str) -> bool:
    """Reads only the reply."""
    return bool(reply.strip())


async def offline() -> dict[str, Any]:
    cases = episodes(24)
    fit, held_out = cases[:12], cases[12:]
    out: dict[str, Any] = {}
    for name, decide in (
        ('grades the action result', grades_the_work),
        ('also reads span ids', reads_span_ids),
        ('grades the result, trusts the claim when no result is shown', trusts_the_claim_without_evidence),
        ('grades only the reply', grades_the_claim),
    ):
        judge = scripted(decide)
        budget = await evidence_budget(judge, fit, EVIDENCE_CONTROLS, parts=parts, project=project)
        check = await check_budget(judge, held_out, EVIDENCE_CONTROLS, kept=budget.kept, project=project)
        # Fitted on the known-good episodes alone, without the controls: what goes wrong.
        bare = await evidence_budget(judge, fit, (), parts=parts, project=project)
        bare_check = await check_budget(judge, held_out, EVIDENCE_CONTROLS, kept=bare.kept, project=project)
        print(f'\n== {name}\n  with controls: {budget.summary()}\n    held out: {check.summary()}')
        print(f'  known-good only: {bare.summary()}\n    held out (with controls): {bare_check.summary()}')
        out[name] = {
            'with_controls': {**budget.to_dict(), 'held_out': check.to_dict()},
            'known_good_only': {**bare.to_dict(), 'held_out': bare_check.to_dict()},
        }
    return out


async def live(max_calls: int = 45) -> dict[str, Any]:
    from codex_judge import TOKENS, codex_model

    cases = episodes(24)
    fit, held_out = cases[:2], cases[2:4]  # one refund and one cancellation in each
    judge = LLMJudge(rubric=RUBRIC, model=codex_model(effort='none'), include_input=True)
    before = len(TOKENS)
    budget = await evidence_budget(
        judge, fit, EVIDENCE_CONTROLS, parts=parts, project=project, max_calls=max_calls, max_concurrency=2
    )
    fit_tokens = TOKENS[before:]
    print(f'\n== Codex gpt-5.6-luna, no reasoning\n  {budget.summary()}')
    for t in budget.trials:
        print(f'    {t.component}: {"dropped" if t.dropped else "kept"} ({t.calls} calls) {t.changed}')
    before = len(TOKENS)
    check = await check_budget(judge, held_out, EVIDENCE_CONTROLS, kept=budget.kept, project=project, max_concurrency=2)
    check_tokens = TOKENS[before:]
    print(f'  held out: {check.summary()}')
    half = len(check_tokens) // 2  # check_budget judges full inputs first, then projected
    return {
        'judge': 'LLMJudge(codex:gpt-5.6-luna, effort none, include_input=True)',
        'fit_cases': [c.name for c in fit],
        'held_out_cases': [c.name for c in held_out],
        **budget.to_dict(),
        'held_out': check.to_dict(),
        'calls': budget.calls + check.calls,
        'tokens': {
            'fit': sum(fit_tokens),
            'held_out_full_inputs': sum(check_tokens[:half]),
            'held_out_projected_inputs': sum(check_tokens[half:]),
            'note': 'tokens Codex reports per call; check_budget judges every full input before any projected one',
        },
    }


async def main() -> None:
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    results['offline'] = await offline()
    OUT.write_text(json.dumps(results, indent=2, default=str))
    if '--codex' in sys.argv and 'codex' not in results:
        results['codex'] = await live()
        OUT.write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    asyncio.run(main())
