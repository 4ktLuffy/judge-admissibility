"""Multi-turn obligations and recovery-aware grading: certify a judge shown the whole conversation.

    PYTHONPATH=.:bench <venv>/bin/python bench/conversation_run.py --backend scripted|codex

Episodes and controls are `bench/conversation_task.py`: the final reply never changes; the
controls insert turns before the customer's last message (consent withdrawn, small talk, an agent
error the agent corrected, the same error left standing). `scripted`: 40 episodes, two repeats,
three scripted judges with known habits (sound; reads only the last turn; fails any trajectory
with an error in it). `codex`: 16 episodes, one repeat, one `LLMJudge` on Codex `gpt-5.6-luna`, no
reasoning, shown the conversation (`include_input=True`): 68 calls.

Each certificate is saved with its recovery profile and, for every control family, the judge's
own reasons, to `results/conversation.json`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from conversation_task import CONTROLS, RUBRIC, episodes, last_turn_only, sound, unforgiving
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility import Certificate, certify_judge, diagnose
from pydantic_evals_admissibility._conversation import recovery_profile

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'conversation.json'


def scripted(decide: Any) -> FunctionModel:
    """A judge model that reads the `<Input>` and `<Output>` sections of LLMJudge's real prompt."""

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(
            p.content
            for m in messages
            for p in getattr(m, 'parts', [])
            if isinstance(p, UserPromptPart) and isinstance(p.content, str)
        )
        found = re.search(r'<Input>\n(.*?)\n</Input>', prompt, re.S)
        output = re.search(r'<Output>\n(.*?)\n</Output>', prompt, re.S)
        passed = decide(json.loads(found.group(1)) if found else None, output.group(1) if output else '')
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

    return FunctionModel(model, model_name=getattr(decide, '__name__', 'scripted'))


def by_family(cert: Certificate) -> dict[str, dict[str, Any]]:
    """Per role: how many judgments passed, and the judge's reasons (every one, they are short)."""
    out: dict[str, dict[str, Any]] = defaultdict(lambda: {'passed': 0, 'failed': 0, 'errors': 0, 'reasons': []})
    for j in cert.judgments:
        role = 'reference' if j.role.startswith('reference#') else j.role
        row = out[role]
        row['passed' if j.passed else 'failed' if j.passed is False else 'errors'] += 1
        row['reasons'].append({'case': j.case, 'passed': j.passed, 'reason': j.reason, 'error': j.error})
    return dict(out)


async def certify(label: str, judge: LLMJudge, n: int, repeats: int, concurrency: int) -> dict[str, Any]:
    cert = await certify_judge(judge, episodes(n), controls=CONTROLS, repeats=repeats, max_concurrency=concurrency)
    profile = recovery_profile(cert)
    print(f'\n== {label}: {cert.verdict} ({cert.calls} calls), recovery: {profile.kind}\n{cert.table()}', flush=True)
    for line in diagnose(cert, judge):
        print('   -', line)
    families = by_family(cert)
    for role, row in families.items():
        print(f'   {role}: passed {row["passed"]}, failed {row["failed"]}, errors {row["errors"]}')
    return {
        **cert.to_dict(judgments=False),
        'recovery': profile.to_dict(),
        'families': families,
        'advice': diagnose(cert, judge),
        'episodes': n,
        'repeats': repeats,
    }


async def main(backend: str) -> None:
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    if backend == 'scripted':
        out = {}
        for decide in (sound, last_turn_only, unforgiving):
            judge = LLMJudge(rubric=RUBRIC, model=scripted(decide), include_input=True)
            out[decide.__name__] = await certify(decide.__name__, judge, 40, 2, 8)
        results['scripted'] = out
    else:
        from codex_judge import TOKENS, codex_model

        judge = LLMJudge(rubric=RUBRIC, model=codex_model(effort='none'), include_input=True)
        label = 'gpt-5.6-luna, no reasoning, shown the conversation'
        results['codex'] = {label: await certify(label, judge, 16, 1, 2)}
        results['codex'][label]['tokens'] = sum(TOKENS)
    OUT.write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('scripted', 'codex'), default='scripted')
    asyncio.run(main(parser.parse_args().backend))
