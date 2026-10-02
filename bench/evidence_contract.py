"""Does a real judge grade the work, or the agent's claim? Evidence controls on Codex judges.

    PYTHONPATH=.:bench <venv>/bin/python bench/evidence_contract.py

24 support episodes (`bench/evidence_task.py`): every reply claims the refund or cancellation
went through, and in the known-good versions the tool results agree. The controls change only
the tool results (`tool_failed`, `amount_differs`) or irrelevant ids (`ids_changed`); the reply
is never touched. Three `LLMJudge`s on Codex `gpt-5.6-luna`: one shown only the reply
(`include_input=False`, the default), and two shown the request and tool calls, without and with
reasoning.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

from codex_judge import TOKENS, codex_model
from evidence_task import CONTROLS, RUBRIC, episodes
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility import certify_judge, diagnose

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'evidence_contract.json'


async def main() -> None:
    cases = episodes(24)
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}
    judges = {
        'reply only (default include_input=False), no reasoning': LLMJudge(rubric=RUBRIC, model=codex_model()),
        'reply and tool calls, no reasoning': LLMJudge(rubric=RUBRIC, model=codex_model(), include_input=True),
        'reply and tool calls, reasoning high': LLMJudge(
            rubric=RUBRIC, model=codex_model(effort='high', timeout=300), include_input=True
        ),
    }
    for label, judge in judges.items():
        if label in results:
            continue
        before = len(TOKENS)
        cert = await certify_judge(judge, cases, controls=CONTROLS, repeats=2, max_concurrency=4)
        advice = diagnose(cert, judge)
        print(f'\n== {label}: {cert.verdict} ({len(TOKENS) - before} calls)\n{cert.table()}', flush=True)
        for line in advice:
            print('   -', line)
        results[label] = {**cert.to_dict(), 'advice': advice, 'tokens': sum(TOKENS[before:])}
        OUT.write_text(json.dumps(results, indent=2, default=str))


if __name__ == '__main__':
    asyncio.run(main())
