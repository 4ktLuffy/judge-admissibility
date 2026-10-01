"""Certify two local models as `LLMJudge`s, in the three ways `LLMJudge` can be configured.

    PYTHONPATH=. <venv>/bin/python bench/certify_local_judges.py

Needs Ollama on localhost:11434 with the models below pulled. Writes
`results/local_judges.json` and prints one certificate per configuration.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.ollama import OllamaProvider
from pydantic_evals.evaluators import LLMJudge

sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))
from test_certify import CASES  # noqa: E402  (the same 30 known-good answers the unit tests use)

from pydantic_evals_admissibility import certify_judge  # noqa: E402

MODELS = ['qwen2.5:7b', 'granite4:7b-a1b-h']
RUBRIC = 'The output correctly answers the question.'
CONFIGS = {
    'default (output only)': dict(include_input=False, include_expected_output=False),
    'with input': dict(include_input=True, include_expected_output=False),
    'with input and expected': dict(include_input=True, include_expected_output=True),
}


async def main() -> None:
    rows = []
    for model_name in MODELS:
        model = OpenAIChatModel(model_name, provider=OllamaProvider(base_url='http://localhost:11434/v1'))
        for config, flags in CONFIGS.items():
            started = time.monotonic()
            certificate = await certify_judge(
                LLMJudge(rubric=RUBRIC, model=model, **flags), CASES, repeats=3, max_concurrency=4
            )
            elapsed = time.monotonic() - started
            print(f'\n=== {model_name} | {config} | {elapsed:.0f}s')
            print(certificate.table())
            rows.append({'model': model_name, 'config': config, 'seconds': round(elapsed), **certificate.to_dict()})
            out = Path(__file__).parent.parent / 'results' / 'local_judges.json'
            out.write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    asyncio.run(main())
