"""Which saved judges are qualified for what: `qualify` over every certificate in `results/`.

    PYTHONPATH=.:bench <venv>/bin/python bench/qualify_demo.py

No judge is called. Each saved certificate is rebuilt with `Certificate.from_dict`, decided again
under the current rules (as `bench/recertify_all.py` does), and asked whether it qualifies its
judge to report, gate, promote, steer, and steer while reading traces. The judge is taken to run
as certified (its recorded identity is passed as the judge); a certificate that recorded none
cannot show what it covers. Freshness uses the date the results file was last committed (its
modification time when it is not committed yet) and a 30-day limit, since a `Certificate`
records no date of its own.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from recertify_all import entries

from pydantic_evals_admissibility import Certificate, recertify, recertify_pairwise
from pydantic_evals_admissibility._qualify import qualify

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'qualify_demo.json'
MAX_AGE_DAYS = 30
USES: list[tuple[str, str, dict[str, Any]]] = [
    ('report', 'report', {}),
    ('gate', 'gate', {}),
    ('promote', 'promote', {}),
    ('steer', 'steer', {}),
    ('steer+traces', 'steer', {'sees_traces': True}),
]


def issued(name: str) -> tuple[datetime | None, str]:
    """When the results file was last committed, or else last written, and which of the two."""
    try:
        out = subprocess.run(
            ['git', 'log', '-1', '--format=%cI', '--', f'results/{name}'],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        out = ''
    if out:
        return datetime.fromisoformat(out), 'git commit'
    path = ROOT / 'results' / name
    if path.exists():
        return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc), 'file modified (not committed)'
    return None, 'unknown'


SHORT = {
    'coverage': 'cov',
    'verdict': 'ver',
    'checks': 'chk',
    'slices': 'sli',
    'context_slice': 'ctx',
    'evidence': 'evi',
    'freshness': 'age',
}


def main() -> None:
    now = datetime.now(timezone.utc)
    report: dict[str, Any] = {'checked_at': now.isoformat(), 'max_age_days': MAX_AGE_DAYS, 'judges': {}}
    header = f'{"certificate":<62} {"verdict":<13}' + ''.join(f'{label:<17}' for label, _, _ in USES)
    print(header)
    print('-' * len(header))
    notes: list[str] = []
    for name, entry in entries():
        saved = Certificate.from_dict(entry)
        pairwise = any(c.name == 'accuracy' for c in saved.checks)
        certificate = recertify_pairwise(saved) if pairwise else recertify(saved)
        when, source = issued(name.split(' ')[0])
        row: dict[str, Any] = {
            'verdict': certificate.verdict,
            'issued_at': when and when.isoformat(),
            'issued_at_source': source,
            'has_identity': certificate.identity is not None,
            'uses': {},
        }
        reasons: dict[str, str] = {}
        cells = []
        for label, decision, context in USES:
            q = qualify(
                certificate.identity,
                certificate,
                decision=decision,  # type: ignore[arg-type]
                context=context,
                max_age_days=MAX_AGE_DAYS,
                issued_at=when,
                now=now,
            )
            failed = [r.name for r in q.rules if r.outcome == 'fail']
            # What re-certifying with the identity recorded would unlock: blocked by coverage alone.
            only_coverage = failed == ['coverage']
            row['uses'][label] = {**q.to_dict(), 'blocked_only_by_coverage': only_coverage}
            cells.append(
                ('yes*' if q.warnings else 'yes') if q.qualified else f'no({",".join(SHORT[f] for f in failed)})'
            )
            for r in q.rules:
                if r.outcome == 'fail' and r.name not in reasons:
                    reasons[r.name] = r.detail
        report['judges'][name] = row
        print(f'{name[:62]:<62} {certificate.verdict:<13}' + ''.join(f'{c:<17}' for c in cells))
        notes.append(f'{name}: ' + ('; '.join(f'{k}: {v}' for k, v in reasons.items()) or 'qualified for every use'))
    print(
        '\nyes* = qualified with a warning; no(...) = the rules that failed: '
        + ', '.join(f'{v}={k}' for k, v in SHORT.items())
    )
    print('\nFirst reason per failing rule, by certificate:')
    for note in notes:
        print(f'- {note}')
    counts = {label: sum(row['uses'][label]['qualified'] for row in report['judges'].values()) for label, _, _ in USES}
    unlock = {
        label: sum(row['uses'][label]['blocked_only_by_coverage'] for row in report['judges'].values())
        for label, _, _ in USES
    }
    report['summary'] = {'certificates': len(report['judges']), 'qualified': counts, 'blocked_only_by_coverage': unlock}
    print(f'\n{len(report["judges"])} certificates. Qualified per use: {counts}')
    print(f'Blocked only because no identity was recorded: {unlock}')
    OUT.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
