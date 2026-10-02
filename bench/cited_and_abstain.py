"""Judges that cite their evidence, and judges that may abstain, on the support episodes.

    PYTHONPATH=.:bench <venv>/bin/python bench/cited_and_abstain.py --backend scripted|codex [--n 16]

Episodes from `bench/evidence_task.py` in three variants: as generated (the reply is true: must
pass), `tool_failed` (the action failed, the reply still claims it: must fail), and
`evidence_removed` (the action's tool result is gone, so nothing says whether it went through:
a judge should abstain with `missing_evidence`).

- Citations: a `CitedJudge` judges the true and the failed variants and cites span ids; each
  verdict's citations are checked for being real and for including the action's tool call (the
  default `facts` rule, the fact being the action tool's name).
- Abstention: an `AbstainingJudge` and a plain `LLMJudge(include_input=True)` judge all three
  variants. The plain judge cannot abstain, so on the removed evidence it must guess.

`--backend scripted` runs scripted judges of known behaviour (no calls); `--backend codex` runs
Codex `gpt-5.6-luna`, no reasoning, 8 calls per episode. Results go to
`results/cited_and_abstain.json`, under the backend's name; a judge already there is skipped,
so a rerun does not pay twice.
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evidence_task import FAILURES, RUBRIC, episodes, tool_failed
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility._abstain import AbstainingJudge, certify_abstention
from pydantic_evals_admissibility._cases import JudgeCase
from pydantic_evals_admissibility._cited import CitedJudge, certify_citations

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'cited_and_abstain.json'


def evidence_removed(inputs: Any) -> Any:
    """The action's tool call with its result deleted: the call was made, its outcome is unknown."""
    changed = copy.deepcopy(inputs)
    del next(c for c in changed['tool_calls'] if c['tool'] in FAILURES)['result']
    return changed


def action_facts(case: JudgeCase) -> list[str]:
    """What a verdict on an episode depends on: the action's tool call (refund or cancellation)."""
    return [next(c['tool'] for c in case.inputs['tool_calls'] if c['tool'] in FAILURES)]


def with_arrival(case: JudgeCase) -> JudgeCase:
    """The refund result also states the arrival window the reply promises.

    Without it, Codex judges (2 of 2 smoke calls, both prompts tried) failed a true refund reply
    because "5-10 business days" is not in the tool results: right by the rubric's letter. The
    same gap in the cancellation replies was fixed in `evidence_task.py` the same way.
    """
    inputs = copy.deepcopy(case.inputs)
    for call in inputs['tool_calls']:
        if call['tool'] == 'issue_refund':
            call['result']['arrival'] = '5-10 business days to the card'
    return JudgeCase(case.name, inputs, case.output, metadata=case.metadata)


def variants(n: int) -> tuple[list[JudgeCase], list[JudgeCase], list[JudgeCase]]:
    good = [with_arrival(c) for c in episodes(n)]
    bad = [JudgeCase(f'{c.name}/tool_failed', tool_failed(c.inputs), c.output, metadata=c.metadata) for c in good]
    missing = [
        JudgeCase(f'{c.name}/evidence_removed', evidence_removed(c.inputs), c.output, metadata=c.metadata) for c in good
    ]
    return good, bad, missing


# --- scripted judges: known behaviour, to show what each check catches -------------------------


def _action(messages: list[ModelMessage]) -> dict[str, Any]:
    parts = [p for m in messages for p in getattr(m, 'parts', []) if isinstance(p, UserPromptPart)]
    prompt = ''.join(p.content for p in parts if isinstance(p.content, str))
    inputs = json.loads(re.search(r'<Input>\n(.*?)\n</Input>', prompt, re.S).group(1))  # type: ignore[union-attr]
    return next(c for c in inputs['tool_calls'] if c['tool'] in FAILURES) | {'_all': inputs['tool_calls']}


def _scripted(respond: Callable[[dict[str, Any]], dict[str, Any]]) -> FunctionModel:
    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        return ModelResponse(parts=[ToolCallPart(info.output_tools[0].name, respond(_action(messages)))])

    return FunctionModel(model)


def _ok(action: dict[str, Any]) -> bool:
    return action['result']['status'] != 'failed'


SCRIPTED_CITERS = {
    'cites the action': lambda a: {'reason': 's', 'citations': [a['span_id']], 'pass': _ok(a)},
    'makes up an id': lambda a: {'reason': 's', 'citations': ['span_00000000ff'], 'pass': _ok(a)},
    'cites the lookup': lambda a: {
        'reason': 's',
        'citations': [c['span_id'] for c in a['_all'] if c['tool'] not in FAILURES],
        'pass': _ok(a),
    },
}
SCRIPTED_ABSTAINERS = {
    'abstains when the result is missing': lambda a: (
        {'reason': 's', 'decision': 'abstain', 'abstain_reason': 'missing_evidence'}
        if 'result' not in a
        else {'reason': 's', 'decision': 'pass' if _ok(a) else 'fail', 'abstain_reason': None}
    ),
    'guesses pass when the result is missing': lambda a: {
        'reason': 's',
        'decision': 'pass' if 'result' not in a or _ok(a) else 'fail',
        'abstain_reason': None,
    },
}


async def run(backend: str, n: int) -> None:
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    done = results.setdefault(backend, {})
    good, bad, missing = variants(n)
    if backend == 'codex':
        from codex_judge import TOKENS, codex_model

        citers = {'CitedJudge, codex no reasoning': CitedJudge(RUBRIC, codex_model())}
        abstainers: dict[str, Any] = {
            'AbstainingJudge, codex no reasoning': AbstainingJudge(RUBRIC, codex_model()),
            'LLMJudge(include_input=True), codex no reasoning': LLMJudge(
                rubric=RUBRIC, model=codex_model(), include_input=True
            ),
        }
        concurrency = 2
    else:
        TOKENS = []
        citers = {k: CitedJudge(RUBRIC, _scripted(f)) for k, f in SCRIPTED_CITERS.items()}
        abstainers = {k: AbstainingJudge(RUBRIC, _scripted(f)) for k, f in SCRIPTED_ABSTAINERS.items()}

        def plain(a: dict[str, Any]) -> dict[str, Any]:
            return {'reason': 's', 'pass': 'result' not in a or _ok(a), 'score': 1.0}

        abstainers['LLMJudge, scripted: passes unless the result says failed'] = LLMJudge(
            rubric=RUBRIC, model=_scripted(plain), include_input=True
        )
        concurrency = 8

    for label, judge in citers.items():
        if label in done:
            continue
        before = len(TOKENS)
        cert = await certify_citations(judge, good, must_fail=bad, facts=action_facts, max_concurrency=concurrency)
        print(f'\n== {label}: {cert.verdict}\n{cert.table()}', flush=True)
        done[label] = {'feature': 'citations', 'episodes': n, **cert.to_dict(), 'codex_calls': len(TOKENS) - before}
        OUT.write_text(json.dumps(results, indent=2, default=str))
    for label, judge in abstainers.items():
        if label in done:
            continue
        before = len(TOKENS)
        cert = await certify_abstention(judge, good, must_fail=bad, must_abstain=missing, max_concurrency=concurrency)
        print(f'\n== {label}: {cert.verdict}\n{cert.table()}', flush=True)
        done[label] = {'feature': 'abstention', 'episodes': n, **cert.to_dict(), 'codex_calls': len(TOKENS) - before}
        OUT.write_text(json.dumps(results, indent=2, default=str))
    if backend == 'codex':
        print(f'\n{len(TOKENS)} Codex calls this run, {sum(TOKENS):,} tokens')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('scripted', 'codex'), default='scripted')
    parser.add_argument('--n', type=int, default=16, help='episodes; each costs 8 judge calls on codex')
    args = parser.parse_args()
    asyncio.run(run(args.backend, args.n))
