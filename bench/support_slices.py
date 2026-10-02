"""Re-decide the support task's saved certificates by kind of case, without calling a judge.

    PYTHONPATH=.:bench <venv>/bin/python bench/support_slices.py

Reads `results/support_eval.json`, rebuilds each certificate with `Certificate.from_dict`, and
`recertify`s it with each case's kind and correct answer as its slice (`return: no`, `refund`, ...).
The verdicts are the ones already paid for; only the decision is made again.
"""

from __future__ import annotations

import json
from pathlib import Path

from support_task import cases

from pydantic_evals_admissibility import Certificate, diagnose, recertify

ROOT = Path(__file__).parent.parent


def kind(case) -> str:  # type: ignore[no-untyped-def]
    """Yes/no questions split by their correct answer; amounts are one slice per kind."""
    return f'{case.kind}: {case.answer}' if case.answer in ('yes', 'no') else case.kind


def main() -> None:
    slices = {c.name: kind(c) for c in cases()}
    saved = json.loads((ROOT / 'results' / 'support_eval.json').read_text())
    out = {}
    for label, entry in saved['certificates'].items():
        before = Certificate.from_dict(entry)
        after = recertify(before, slices=slices)
        advice = diagnose(after)
        print(f'== {label}: {before.verdict} overall -> {after.verdict} by slice\n{after.table()}')
        for a in advice:
            print('   -', a)
        print()
        out[label] = {'overall': before.verdict, 'by_slice': after.verdict, **after.to_dict(judgments=False)}
        out[label]['advice'] = advice
    (ROOT / 'results' / 'support_slices.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
