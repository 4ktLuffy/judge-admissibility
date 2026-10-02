"""The same certificates with the verdict asked for before the reason, and after it.

    PYTHONPATH=.:bench <venv>/bin/python bench/field_order.py

Until it was caught, `codex_judge.py` sent Codex the judge's output schema with its keys sorted,
so `pass` came before `reason` (and `choice` before `reason` for comparison judges). Structured
output is written in schema order, so the judge decided first and explained afterwards.
`LLMJudge`'s own schema puts `reason` first. The verdict-first results are kept in
`results/verdict_first/`; this prints them next to the re-runs with the order fixed. Same cases,
same cached agent replies, same models; only the field order differs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from support_slices import kind
from support_task import cases

from pydantic_evals_admissibility import Certificate, recertify

ROOT = Path(__file__).parent.parent
OLD, NEW = ROOT / 'results' / 'verdict_first', ROOT / 'results'


def certificates(name: str, root: Path) -> dict[str, dict[str, Any]]:
    path = root / name
    if not path.exists():
        return {}
    data = json.loads(path.read_text())
    if name == 'support_eval.json':
        found = dict(data.get('certificates', {}))
        found.update({k: v for k, v in data.items() if k.startswith('pairwise')})
        return found
    if name == 'pydantic_example_judge.json':
        return data['judges']
    if name == 'certify_pydantic_dataset.json':
        return {j['label'][:40]: j for j in data['judges'] if 'checks' in j}
    return {'comparison judge, no reasoning': data}


def summary(entry: dict[str, Any], slices: dict[str, str] | None) -> dict[str, Any]:
    checks = {c['name']: f'{c["successes"]}/{c["trials"]}' for c in entry['checks']}
    out: dict[str, Any] = {'verdict': entry['verdict'], **checks}
    if 'first_position_rate' in entry and entry['first_position_rate']:
        out['first shown chosen'] = round(entry['first_position_rate'][0], 2)
    if slices and 'judgments' in entry and any(c['name'] == 'acceptance' for c in entry['checks']):
        sliced = recertify(Certificate.from_dict(entry), slices=slices)
        out['slices'] = next(f'{c.status}: {c.detail}' for c in sliced.checks if c.name == 'slices')
        out['verdict by slice'] = sliced.verdict
    return out


def main() -> None:
    support_slices = {c.name: kind(c) for c in cases()}
    report: dict[str, Any] = {}
    for name in ('support_eval.json', 'pairwise_codex.json', 'pairwise_codex.terse.json',
                 'pydantic_example_judge.json', 'certify_pydantic_dataset.json'):  # fmt: skip
        slices = support_slices if name == 'support_eval.json' else None
        old, new = certificates(name, OLD), certificates(name, NEW)
        for label in old:
            row = {'verdict first': summary(old[label], slices)}
            if label in new:
                row['reason first'] = summary(new[label], slices)
            report[f'{name}: {label}'] = row
            print(f'\n{name}: {label}')
            for order, s in row.items():
                print(f'  {order:<14} ' + '; '.join(f'{k} {v}' for k, v in s.items()))
    (ROOT / 'results' / 'field_order.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
