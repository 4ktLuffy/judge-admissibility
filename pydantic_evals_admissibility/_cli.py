"""`judge-admissibility`: certify the judges in a pydantic-evals dataset file, from a shell or CI.

    judge-admissibility certify cases.yaml --model openai:gpt-5 --json certificates.json
    judge-admissibility report certificates.json

`certify` reads the file the way `Dataset.from_file` does but keeps only its `LLMJudge`
evaluators, so the dataset's own custom evaluators need not be importable. It certifies every
judge with `certify_dataset`, prints each certificate with the doctor's advice, and exits 1
unless every judge it certified is ADMISSIBLE, so it can gate a CI job. `report` prints a saved
result again, with the current doctor, without calling a judge.

Exit codes: 0 every judge is certified ADMISSIBLE; 1 one is not, or one was skipped (unless
`--allow-skipped`); 2 no judge could be certified.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from ._dataset import DatasetJudgeCertificate, certify_dataset, dataset_report
from ._diagnose import diagnose


def _name(spec: Any) -> str | None:
    """The evaluator name in a serialized spec: `Name`, `{Name: arg}` or `{Name: {kwargs}}`."""
    if isinstance(spec, str):
        return spec
    if isinstance(spec, dict) and len(spec) == 1:
        return next(iter(spec))
    return None


def only_judges(data: dict[str, Any]) -> dict[str, Any]:
    """The serialized dataset with every evaluator but `LLMJudge` removed."""

    def keep(specs: Any) -> list[Any]:
        return [s for s in specs or [] if _name(s) == 'LLMJudge']

    cases = [{**case, 'evaluators': keep(case.get('evaluators'))} for case in data.get('cases', [])]
    out = {k: v for k, v in data.items() if k not in ('report_evaluators', '$schema')}
    return {**out, 'cases': cases, 'evaluators': keep(data.get('evaluators'))}


def load(path: Path) -> Any:
    from pydantic_evals import Dataset

    text = path.read_text()
    if path.suffix in ('.yaml', '.yml'):
        import yaml

        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    # Untyped: inputs and outputs load as plain data, which is what the judge is shown anyway.
    return Dataset[Any, Any, Any].from_dict(only_judges(data), default_name=path.stem)


def _exit_code(results: Sequence[DatasetJudgeCertificate], allow_skipped: bool = False) -> int:
    certified = [r.certificate for r in results if r.certificate is not None]
    if not certified:
        return 2
    if not allow_skipped and len(certified) < len(results):
        return 1  # a CI gate over the whole dataset must not pass on the judges it could check
    return 0 if all(c.admissible for c in certified) else 1


async def _certify(args: argparse.Namespace) -> int:
    dataset = load(Path(args.dataset))
    results = await certify_dataset(
        dataset,
        model=args.model,
        repeats=args.repeats,
        min_cases=args.min_cases,
        batch_size=args.batch_size,
        max_concurrency=args.max_concurrency,
    )
    if not results:
        print(f'{args.dataset}: no LLMJudge evaluators to certify')
        return 2
    print(dataset_report(results))
    if args.json:
        Path(args.json).write_text(json.dumps({'judges': [r.to_dict() for r in results]}, indent=2, default=str))
    return _exit_code(results, args.allow_skipped)


def _report(args: argparse.Namespace) -> int:
    saved = json.loads(Path(args.saved).read_text())
    results = [DatasetJudgeCertificate.from_dict(entry) for entry in saved['judges']]
    for r in results:
        if r.certificate is not None:
            judge = SimpleNamespace(
                rubric=r.rubric, include_input=r.include_input, include_expected_output=r.include_expected_output
            )
            r.advice = diagnose(r.certificate, judge)
    print(dataset_report(results))
    return _exit_code(results)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog='judge-admissibility', description='Certify the LLM judges in a pydantic-evals dataset file.'
    )
    commands = parser.add_subparsers(dest='command', required=True)

    certify = commands.add_parser('certify', help='certify every LLMJudge in a dataset file')
    certify.add_argument('dataset', help='a pydantic-evals dataset, .yaml or .json')
    certify.add_argument('--model', help="certify on this model instead of each judge's own, e.g. openai:gpt-5")
    certify.add_argument('--repeats', type=int, default=2, help='judgments per known-good answer (default 2)')
    certify.add_argument('--min-cases', type=int, default=10, help='skip judges with fewer cases (default 10)')
    certify.add_argument('--batch-size', type=int, help='judge this many cases at a time; stop early on failure')
    certify.add_argument('--max-concurrency', type=int, default=8, help='judgments in flight (default 8)')
    certify.add_argument('--json', help='also write the certificates, with every verdict, to this file')
    certify.add_argument(
        '--allow-skipped', action='store_true', help='exit 0 even if some judges could not be certified'
    )

    report = commands.add_parser('report', help='print saved certificates again, without calling a judge')
    report.add_argument('saved', help='a file written by `certify --json`')

    args = parser.parse_args(argv)
    if args.command == 'certify':
        return asyncio.run(_certify(args))
    return _report(args)


def main_exit() -> None:
    """The `judge-admissibility` console script."""
    sys.exit(main())


if __name__ == '__main__':
    main_exit()
