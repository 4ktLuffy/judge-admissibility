"""One call on pydantic-ai's real example dataset: `certify_dataset(Dataset.from_file(...))`.

    PYTHONPATH=.:bench <venv>/bin/python bench/certify_pydantic_dataset.py [--rediagnose]

Loads `examples/pydantic_ai_examples/evals/datasets/time_range_v2.yaml` exactly as the example
does, with its own custom evaluator types, and certifies every judge in it on Codex
`gpt-5.6-luna` (reasoning high) in place of the judge's configured model. The example package
builds an OpenAI agent on import, so a placeholder key is set; no OpenAI call is made.

`--rediagnose` makes no judge calls: it rebuilds the saved certificates and runs the current
doctor on them again, so a change to `diagnose` can be checked against the same verdicts.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
from pathlib import Path

os.environ.setdefault('OPENAI_API_KEY', 'placeholder-no-calls-are-made')

from codex_judge import TOKENS, codex_model  # noqa: E402
from pydantic_ai_examples.evals.custom_evaluators import CUSTOM_EVALUATOR_TYPES  # noqa: E402
from pydantic_ai_examples.evals.models import TimeRangeInputs, TimeRangeResponse  # noqa: E402
from pydantic_evals import Dataset  # noqa: E402
from pydantic_evals.evaluators import LLMJudge  # noqa: E402

from pydantic_evals_admissibility import Certificate, certify_dataset, dataset_report, diagnose  # noqa: E402

ROOT = Path(__file__).parent.parent
UPSTREAM = ROOT.parent / 'pydantic-durability' / 'upstream'
PATH = UPSTREAM / 'examples/pydantic_ai_examples/evals/datasets/time_range_v2.yaml'
SAVED = ROOT / 'results' / 'certify_pydantic_dataset.json'


def rediagnose(dataset: Dataset) -> None:
    saved = json.loads(SAVED.read_text())
    judges = {f'dataset: {e.rubric[:60]}': e for e in dataset.evaluators if isinstance(e, LLMJudge)}
    for entry in saved['judges']:
        if 'checks' not in entry:
            continue
        cert = Certificate.from_dict(entry)
        entry['advice'] = diagnose(cert, judges.get(entry['label']))
        print(f'{entry["label"]}: {cert.verdict}\n' + '\n'.join(f'  - {a}' for a in entry['advice']))
    SAVED.write_text(json.dumps(saved, indent=2, default=str))


async def main() -> None:
    commit = subprocess.run(
        ['git', 'rev-parse', '--short', 'HEAD'], cwd=UPSTREAM, capture_output=True, text=True
    ).stdout.strip()
    dataset = Dataset[TimeRangeInputs, TimeRangeResponse, None].from_file(
        PATH, custom_evaluator_types=CUSTOM_EVALUATOR_TYPES
    )
    if '--rediagnose' in sys.argv:
        return rediagnose(dataset)
    results = await certify_dataset(
        dataset, model=codex_model(effort='high', timeout=300), repeats=3, max_concurrency=4
    )
    report = dataset_report(results)
    print(f'pydantic-ai {commit}, {PATH.name}: {len(dataset.cases)} cases, {len(TOKENS)} judge calls\n\n{report}')
    SAVED.write_text(json.dumps({
        'commit': commit, 'dataset': PATH.name, 'calls': len(TOKENS),
        'judges': [{'label': r.label, 'kind': r.kind, 'cases': r.cases, 'skipped': r.skipped, 'advice': r.advice,
                    **(r.certificate.to_dict() if r.certificate else {})} for r in results],
    }, indent=2, default=str))  # fmt: skip


if __name__ == '__main__':
    asyncio.run(main())
