"""Every saved certificate in `results/`, decided again under the package's current rules.

    PYTHONPATH=.:bench <venv>/bin/python bench/recertify_all.py

No judge is called: the saved verdicts are rebuilt with `Certificate.from_dict` and decided again
with `recertify` / `recertify_pairwise`. Run after a change to how certificates are decided, so
the README's numbers can be checked against the rules it describes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic_evals_admissibility import Certificate, diagnose, recertify, recertify_pairwise

ROOT = Path(__file__).parent.parent


def entries() -> list[tuple[str, dict[str, Any]]]:
    """Every saved certificate that kept its verdicts; older files without them cannot be redecided."""
    out: list[tuple[str, dict[str, Any]]] = []

    def visit(name: str, value: Any) -> None:
        if isinstance(value, dict):
            if 'checks' in value and 'judgments' in value:
                out.append((name, value))
                return
            for key, inner in value.items():
                visit(f'{name} {inner.get("label", key) if isinstance(inner, dict) else key}', inner)
        elif isinstance(value, list):
            for i, inner in enumerate(value):
                visit(f'{name} {inner.get("config", inner.get("label", i)) if isinstance(inner, dict) else i}', inner)

    for path in sorted((ROOT / 'results').glob('*.json')):
        if path.name not in ('recertify_all.json', 'field_order.json', 'support_slices.json'):
            visit(path.name, json.loads(path.read_text()))
    return out


def main() -> None:
    report = {}
    for name, entry in entries():
        before = Certificate.from_dict(entry)
        pairwise = any(c.name == 'accuracy' for c in before.checks)
        after = recertify_pairwise(before) if pairwise else recertify(before)
        report[name] = {'before': before.verdict, **after.to_dict(judgments=False), 'advice': diagnose(after)}
        print(f'== {name}: {before.verdict} -> {after.verdict}\n{after.table()}\n')
    (ROOT / 'results' / 'recertify_all.json').write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
