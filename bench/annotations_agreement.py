"""Judge agreement from a Logfire-style annotation export, on saved real verdicts. No model calls.

    PYTHONPATH=.:bench <venv>/bin/python bench/annotations_agreement.py

What is real: the 80 support-agent replies (Codex `gpt-5.6-luna`), and each certified Codex
judge's pass/fail verdict on every one of them, all read from `results/support_eval.json`
(written by `bench/support_eval.py`). Ground truth is real too: `support_task.is_correct`
checks each reply against the answer computed from the policy, and this script re-checks it
against the labels the certificates were given.

What is constructed: the annotation export. No person reviewed these replies. Ground truth
stands in for the reviewers: each reply becomes one export row with verdict pass (correct) or
fail (wrong), reviewer `ground-truth@bench.invalid`, a comment that says so, and the computed
answer as the expected output of a fail. Trace and span ids are made up (hashes of the case and
reply number), because these replies were not traced in Logfire. The keys are the defaults of
`AnnotationFormat`: Logfire documents which fields an export row has, not their key names. With
one stand-in reviewer per run, the export has no neutral verdicts and no conflicts, so those
rules are not exercised here (the tests exercise them).

For each certificate, `annotation_agreement` is run twice: alone (the whole 5% FAIL budget), and
with the certificate's own share of the budget, which must reproduce exactly (kappa, interval,
status, detail) the `human_agreement` check that `recertify` gives the saved judgments. The
checks as saved were decided by an older rule (an interval on raw agreement, not the
case-bootstrap kappa interval); they are recorded alongside, and only their kappa is compared. Writes the export to
`tests/fixtures/annotations/support_eval_truth.jsonl` and the results to
`results/annotations_agreement.json`.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from support_task import cases, is_correct

from pydantic_evals_admissibility._annotations import annotation_agreement, read_annotations, verdicts_from_certificate
from pydantic_evals_admissibility._certify import Certificate, recertify

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / 'results' / 'support_eval.json'
EXPORT = ROOT / 'tests' / 'fixtures' / 'annotations' / 'support_eval_truth.jsonl'
OUT = ROOT / 'results' / 'annotations_agreement.json'
REVIEWER = 'ground-truth@bench.invalid'


def ids(run: str) -> tuple[str, str]:
    digest = hashlib.sha256(run.encode()).hexdigest()
    return digest[:32], digest[32:48]


def export_rows(cert: Certificate) -> list[dict[str, Any]]:
    """One export row per labelled reply, the label checked again against `is_correct`."""
    by_name = {c.name: c for c in cases()}
    seen: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for j in cert.judgments:
        if not j.role.startswith('human:'):
            continue
        i = seen.get(j.case, 0)
        seen[j.case] = i + 1
        case = by_name[j.case]
        passed = j.role == 'human:1'
        if passed != is_correct(case, j.output):
            raise SystemExit(f'{j.case}#{i}: the saved label does not match is_correct')
        trace_id, span_id = ids(f'{j.case}#{i}')
        answer = case.answer if case.answer in ('yes', 'no') else f'${case.answer}'
        rows.append(
            {
                'trace_id': trace_id,
                'span_id': span_id,
                'agent_name': 'support',
                'verdict': 'pass' if passed else 'fail',
                'category': None,
                'expected_output': None if passed else f'Answer: {answer}',
                'comment': 'ground truth (support_task.is_correct), standing in for a reviewer',
                'tags': ['ground-truth'],
                'reviewer_email': REVIEWER,
            }
        )
    return rows


def certificate_alpha(cert: Certificate) -> float:
    """The FAIL budget the certificate gave its agreement check: 5% over its tests and looks (`_assess`)."""
    tests = 1
    for check in cert.checks:
        if check.name in ('rejection', 'invariance'):
            tests += len(check.families)
        elif check.name in ('stability', 'human_agreement'):
            tests += 1
        elif check.name == 'slices':
            raise SystemExit('a sliced certificate: count its slices too')
    return 0.05 / (tests * cert.looks)


def main() -> None:
    saved = json.loads(SOURCE.read_text())['certificates']
    certs = {label: Certificate.from_dict(data) for label, data in saved.items()}

    # The full certificates labelled the same 80 replies; the one stopped early, a prefix of them.
    full = next(cert for cert in certs.values() if sum(j.role.startswith('human:') for j in cert.judgments) == 80)
    rows = export_rows(full)
    EXPORT.write_text(''.join(json.dumps(row) + '\n' for row in rows))
    annotations = read_annotations(EXPORT)
    run_ids = {key: ids(key)[1] for key in verdicts_from_certificate(full)}  # case#i to the span id in the export
    print(f'export: {EXPORT.relative_to(ROOT)}, {annotations.summary()}\n')

    results: dict[str, Any] = {
        'source': str(SOURCE.relative_to(ROOT)),
        'export': str(EXPORT.relative_to(ROOT)),
        'what_is_real': 'replies and judge verdicts (Codex, saved in support_eval.json); ground truth from is_correct',
        'what_is_constructed': 'the annotation export: ground truth stands in for reviewers; ids are hashes; '
        'key names are AnnotationFormat defaults, since Logfire does not document them',
        'annotations': annotations.summary(),
        'judges': {},
    }
    for label, cert in certs.items():
        by_key = verdicts_from_certificate(cert)
        verdicts = {run_ids[key]: passed for key, passed in by_key.items()}
        case_of = {run_ids[key]: key.rsplit('#', 1)[0] for key in by_key}
        alone = annotation_agreement(annotations, verdicts, case_of=case_of)
        alpha = certificate_alpha(cert)
        shared = annotation_agreement(annotations, verdicts, case_of=case_of, alpha_fail=alpha)
        saved_check = next(c for c in cert.checks if c.name == 'human_agreement')
        current = next(c for c in recertify(cert).checks if c.name == 'human_agreement')
        same = (
            shared.check == current
            and shared.status == current.status
            and f'kappa={shared.kappa:.2f}' in saved_check.detail
        )
        print(f'== {label} ({cert.judge}, certificate {cert.verdict})\n{alone.table()}')
        print(
            f"  with the certificate's FAIL budget ({alpha:.5f}): {shared.status}, "
            f'{"identical to" if same else "DIFFERENT FROM"} recertify(certificate) ({current.status}); '
            f'as saved, under the older rule: {saved_check.status} {saved_check.detail!r}\n'
        )
        if not same:
            raise SystemExit(f'{label}: annotation_agreement does not reproduce the certificate')
        results['judges'][label] = {
            'judge': cert.judge,
            'certificate_verdict': cert.verdict,
            'labelled_replies_judged': len(by_key),
            'alone': alone.to_dict(),
            'with_certificate_budget': {
                'alpha_fail': alpha,
                'status': shared.status,
                'identical_to_recertify': same,
            },
            'saved_check_older_rule': {
                'status': saved_check.status,
                'interval': list(saved_check.interval),
                'detail': saved_check.detail,
                'note': 'decided when the certificate was made, on a raw-agreement interval; recertify re-decides '
                'it with the current case-bootstrap kappa interval',
            },
        }
    OUT.write_text(json.dumps(results, indent=2) + '\n')
    print(f'wrote {OUT.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
