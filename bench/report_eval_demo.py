"""A judge's certificate inside the pydantic-evals report its scores appear in.

    PYTHONPATH=. <venv>/bin/python bench/report_eval_demo.py

No model is called: the judges are real `LLMJudge`s on scripted `FunctionModel`s (`tests/judges.py`),
each given a model name of its own (`oracle`, `yes-man`), as a real model has, so the report records
which model produced each result. One dataset of 20 arithmetic questions, a task that gets 3 of
them wrong, and four runs:

1. a sound judge, certified before the run with `certify_for_report`: coverage `yes`;
2. a judge that passes everything, certified during the run by `CertifyJudgeReport`;
3. the sound judge's certificate, attached to a judge whose rubric has since changed: `no`;
4. the sound judge on a `FunctionModel` with no name of its own, certified, and its certificate
   attached to a report scored by an unnamed pass-everything judge: the report records both
   models as `function:function:model:`, so coverage is `unverifiable` and the 20/20 passes are
   NOT EVIDENCE, rather than `yes`.

Each report is rendered with `report.render()`; the analyses are written to `results/report_eval_demo.json`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import json
import sys
from pathlib import Path
from typing import Any

from pydantic_ai.models.function import FunctionModel
from pydantic_evals import Case, Dataset
from pydantic_evals.evaluators import LLMJudge
from pydantic_evals.reporting import EvaluationReport

sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))

from judges import judge, oracle, yes_man  # noqa: E402  # pyright: ignore[reportMissingImports]

from pydantic_evals_admissibility._report_eval import (  # noqa: E402
    CertifiedJudgeReport,
    CertifyJudgeReport,
    certify_for_report,
)

OUT = Path(__file__).parent.parent / 'results' / 'report_eval_demo.json'
WRONG = {3, 8, 15}


def dataset(llm_judge: Any) -> Dataset[dict[str, int], str, Any]:
    cases = [Case(name=f'q{i}', inputs={'a': i, 'b': 7}, expected_output=f'The sum is {i + 7}.') for i in range(20)]
    return Dataset(name='sums', cases=cases, evaluators=[llm_judge])


def model_named(llm_judge: LLMJudge, name: str | None) -> LLMJudge:
    """The scripted judge on a FunctionModel named `name` (None: no name of its own)."""
    assert isinstance(llm_judge.model, FunctionModel) and llm_judge.model.function is not None
    return dataclasses.replace(llm_judge, model=FunctionModel(llm_judge.model.function, model_name=name))


def task(inputs: dict[str, int]) -> str:
    a, b = inputs['a'], inputs['b']
    return f'The sum is {a + b + (1 if a in WRONG else 0)}.'


def show(title: str, report: EvaluationReport[Any, Any, Any]) -> dict[str, Any]:
    print(f'\n===== {title} =====')
    print(report.render(width=150, include_averages=True))
    return {
        'title': title,
        'judge_pass_rate': sum(c.assertions['LLMJudge'].value for c in report.cases) / len(report.cases),
        'analyses': [a.model_dump() for a in report.analyses],
        'report_evaluator_failures': [f.error_message for f in report.report_evaluator_failures],
    }


async def main() -> None:
    runs: list[dict[str, Any]] = []

    sound = dataset(model_named(judge(oracle), 'oracle'))
    attach = await certify_for_report(sound)
    sound.report_evaluators.append(attach)
    runs.append(show('sound judge, certified before the run', await sound.evaluate(task, progress=False)))

    broken_judge = model_named(judge(yes_man), 'yes-man')
    broken = dataset(broken_judge)
    broken.report_evaluators.append(CertifyJudgeReport(broken_judge))
    runs.append(
        show('judge that passes everything, certified during the run', await broken.evaluate(task, progress=False))
    )

    changed_judge = dataclasses.replace(attach.judge, rubric='The output is a correct sum, stated politely.')
    changed = dataset(changed_judge)
    changed.report_evaluators.append(CertifiedJudgeReport(attach.certified, judge=changed_judge))
    runs.append(
        show('certificate of the sound judge, rubric since changed', await changed.evaluate(task, progress=False))
    )

    unnamed_sound = model_named(judge(oracle), None)
    unnamed_cert = (await certify_for_report(dataset(unnamed_sound))).certified
    impostor = dataset(model_named(judge(yes_man), None))
    impostor.report_evaluators.append(CertifiedJudgeReport(unnamed_cert, judge=unnamed_sound))
    runs.append(
        show(
            'unnamed FunctionModel: the sound judge certified, a pass-everything judge scored the report',
            await impostor.evaluate(task, progress=False),
        )
    )

    OUT.write_text(json.dumps({'wrong_answers': sorted(WRONG), 'cases': 20, 'runs': runs}, indent=2) + '\n')
    print(f'\nwrote {OUT.relative_to(OUT.parent.parent)}')


if __name__ == '__main__':
    asyncio.run(main())
