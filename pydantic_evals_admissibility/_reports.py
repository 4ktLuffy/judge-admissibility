"""Gate a change using two ordinary pydantic-evals reports.

`Dataset.evaluate(task, repeat=N)` already produces everything the gate needs: per-case
assertion results, repeated. `compare_reports` reads one assertion from each report, pairs the
cases by their source case, and decides.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from pydantic_evals.reporting import EvaluationReport

from ._certify import Certificate
from ._gate import GateResult, GateRules, decide


def outcomes(report: EvaluationReport[Any, Any, Any], assertion: str) -> dict[str, list[bool]]:
    """Per source case, the value of `assertion` on each run.

    A run whose task raised, or whose evaluator did not produce `assertion`, counts as a fail:
    it is not evidence the case passed, and dropping it would let a crashing candidate look
    better than one that answered and was wrong.
    """
    out: dict[str, list[bool]] = defaultdict(list)
    for case in report.cases:
        key = case.source_case_name or case.name
        result = case.assertions.get(assertion)
        out[key].append(bool(result.value) if result is not None else False)
    for failure in report.failures:
        out[failure.source_case_name or failure.name].append(False)
    return dict(out)


def compare_reports(
    baseline: EvaluationReport[Any, Any, Any],
    candidate: EvaluationReport[Any, Any, Any],
    *,
    assertion: str,
    certificate: Certificate | None = None,
    rules: GateRules | None = None,
    judge: Any = None,
) -> GateResult:
    """Decide whether `candidate` should replace `baseline`, from one assertion in both reports.

    Args:
        baseline: The report for the current version.
        candidate: The report for the proposed version, on the same dataset.
        assertion: The assertion to compare, as named in the reports (for an `LLMJudge`, its
            evaluation name).
        certificate: The certificate of the judge behind `assertion`. Without an ADMISSIBLE
            certificate the gate refuses; leave it out only for deterministic evaluators.
        rules: The thresholds; use `rules.for_candidates(k)` when choosing among `k` candidates.
        judge: The judge behind `assertion`; given, a certificate for another configuration of it
            is refused.
    """
    return decide(
        outcomes(baseline, assertion), outcomes(candidate, assertion), certificate=certificate, rules=rules, judge=judge
    )
