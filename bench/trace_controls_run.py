"""Does a judge shown the trace grade with hindsight, or miss what the tool calls did? Trace controls.

    PYTHONPATH=.:bench <venv>/bin/python bench/trace_controls_run.py --backend scripted|codex

Two tasks, one `LLMJudge` each, shown the inputs (`include_input=True`), one judgment per case:

- `bench/temporal_task.py`: a decision right at the time, and its decisive fact added two days
  after it (`hindsight_added`, must hold) or one minute before it (`info_was_available`, must fail).
- `bench/retry_task.py`: the right reply, and tool calls that refunded or charged twice
  (`duplicate_side_effect`) or retried on another order or card (`retry_changed_arguments`), both
  must fail; or a timeout retried once with identical arguments (`retried_and_recovered`, must hold).

`--backend scripted` runs an oracle judge through `LLMJudge`'s real prompt, for free. `--backend
codex` runs Codex `gpt-5.6-luna` with no reasoning, two calls at a time, and refuses to start if
the planned calls exceed the budget. Results go to `results/trace_controls.json`; a task already
there is not run again.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import random
import re
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import retry_task
import temporal_task
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility._cases import JudgeCase
from pydantic_evals_admissibility._certify import certify_judge
from pydantic_evals_admissibility._controls import Control
from pydantic_evals_admissibility._diagnose import diagnose

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'trace_controls.json'
BUDGET = 150
EPISODES = 16
TASKS: dict[str, tuple[str, Callable[..., list[JudgeCase]], Sequence[Control], Callable[[Any, str], bool]]] = {
    'temporal': (temporal_task.RUBRIC, temporal_task.episodes, temporal_task.CONTROLS, temporal_task.oracle),
    'retry': (retry_task.RUBRIC, retry_task.episodes, retry_task.CONTROLS, retry_task.oracle),
}


def scripted(decide: Callable[[Any, str], bool]) -> FunctionModel:
    """A model that reads the `<Input>` and `<Output>` sections of `LLMJudge`'s real prompt and decides."""

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

    return FunctionModel(model)


def planned_calls(cases: Sequence[JudgeCase], controls: Sequence[Control]) -> int:
    """Judge calls a one-repeat certificate makes: every case, and every control that applies to it."""
    rng = random.Random(0)
    made = sum(c.make_case(case, cases, rng) is not None for c in controls for case in cases)  # type: ignore[attr-defined]
    return len(cases) + made


def reasons(cert_dict: dict[str, Any]) -> dict[str, list[str]]:
    """The judge's reasons by role, for quoting."""
    out: dict[str, list[str]] = {}
    for j in cert_dict['judgments']:
        out.setdefault(j['role'], []).append(f'{j["case"]} pass={j["passed"]}: {j["reason"] or j["error"]}')
    return out


def _codex() -> FunctionModel:
    from codex_judge import codex_model

    return codex_model()  # gpt-5.6-luna, effort 'none'


async def main(backend: str) -> None:
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() and backend == 'codex' else {}
    tokens: list[int] = []
    if backend == 'codex':
        from codex_judge import TOKENS

        tokens = TOKENS
        todo = [t for t in TASKS if t not in results]
        total = sum(planned_calls(TASKS[t][1](EPISODES), TASKS[t][2]) for t in todo)
        print(f'planned Codex calls: {total} (budget {BUDGET})', flush=True)
        if total > BUDGET:
            raise SystemExit('over budget')
    for task, (rubric, episodes, controls, oracle) in TASKS.items():
        if task in results:
            continue
        cases = episodes(EPISODES)
        model = _codex() if backend == 'codex' else scripted(oracle)
        judge = LLMJudge(rubric=rubric, model=model, include_input=True)
        before = len(tokens)
        cert = await certify_judge(judge, cases, controls=controls, repeats=1, max_concurrency=2)
        advice = diagnose(cert, judge)
        print(f'\n== {task}: {cert.verdict} ({cert.calls} calls)\n{cert.table()}', flush=True)
        for line in advice:
            print('   -', line)
        if backend == 'codex':
            results[task] = {**cert.to_dict(), 'rubric': rubric, 'advice': advice, 'tokens': sum(tokens[before:])}
            OUT.write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--backend', choices=('scripted', 'codex'), default='scripted')
    asyncio.run(main(parser.parse_args().backend))
