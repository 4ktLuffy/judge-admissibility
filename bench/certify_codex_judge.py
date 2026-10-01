"""Certify Codex (`gpt-5.6-luna`, no reasoning) as an `LLMJudge`, in two configurations.

    PYTHONPATH=.:bench <venv>/bin/python bench/certify_codex_judge.py

20 known-good answers, 2 judgments each, plus three controls per case: about 100 calls per
configuration at ~600 tokens each. Writes `results/codex_judge.json`.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from codex_judge import TOKENS, codex_model
from pydantic_evals.evaluators import LLMJudge

sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))
from test_certify import CASES  # noqa: E402

from pydantic_evals_admissibility import certify_judge  # noqa: E402

RUBRIC = 'The output correctly answers the question.'
CONFIGS = {
    'default (judge sees only the output)': dict(include_input=False),
    'with input (judge sees the question)': dict(include_input=True),
}


async def main() -> None:
    model = codex_model()
    rows = []
    for config, flags in CONFIGS.items():
        TOKENS.clear()
        started = time.monotonic()
        certificate = await certify_judge(
            LLMJudge(rubric=RUBRIC, model=model, **flags), CASES[:20], repeats=2, max_concurrency=4
        )
        elapsed = time.monotonic() - started
        print(f'\n=== {config} | {elapsed:.0f}s | {len(TOKENS)} calls, {sum(TOKENS):,} tokens')
        print(certificate.table())
        rows.append({'config': config, 'seconds': round(elapsed), 'calls': len(TOKENS), 'tokens': sum(TOKENS), **certificate.to_dict()})
        (Path(__file__).parent.parent / 'results' / 'codex_judge.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
