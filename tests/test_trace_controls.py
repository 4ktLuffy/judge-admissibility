"""Trace controls: a fact timestamped after the decision, or a tool call that did harm on the way."""

from __future__ import annotations

import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from pydantic_evals.evaluators import LLMJudge

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
import retry_task  # noqa: E402
import temporal_task  # noqa: E402
from trace_controls_run import planned_calls, scripted  # noqa: E402

from pydantic_evals_admissibility._certify import Certificate, certify_judge  # noqa: E402
from pydantic_evals_admissibility._trace_controls import duplicate_call, hindsight_pair  # noqa: E402


def _row(cert: Certificate, name: str) -> Any:
    return next(c for c in cert.checks if c.name == name)


async def _certify(task: Any, decide: Any, n: int = 40) -> Certificate:
    judge = LLMJudge(rubric=task.RUBRIC, model=scripted(decide), include_input=True)
    return await certify_judge(judge, task.episodes(n), controls=task.CONTROLS, repeats=1)


def with_hindsight(inputs: Any, output: str) -> bool:
    """Fails a decision whenever the trace holds its decisive fact, whenever that fact arrived."""
    return bool(output.strip()) and not any(e['source'] in temporal_task.FACT_SOURCES for e in inputs['events'])


def ignores_the_fact(inputs: Any, output: str) -> bool:
    """Passes any decision: never reads the timeline."""
    return bool(output.strip())


def reads_the_reply(inputs: Any, output: str) -> bool:
    """Passes any reply that claims success: never reads the tool calls."""
    return bool(output.strip())


def no_failures_allowed(inputs: Any, output: str) -> bool:
    """Right about harm, but also fails any trace with a failed tool call, harmless or not."""
    failed = any(c['result'].get('status') == 'error' for c in inputs['tool_calls'])
    return retry_task.oracle(inputs, output) and not failed


async def test_an_oracle_judge_is_admissible_on_both_tasks() -> None:
    for task in (temporal_task, retry_task):
        cert = await _certify(task, task.oracle)
        assert cert.verdict == 'ADMISSIBLE', cert.table()


async def test_a_hindsight_judge_fails_the_must_hold_family() -> None:
    cert = await _certify(temporal_task, with_hindsight)
    assert cert.verdict == 'INADMISSIBLE', cert.table()
    assert 'hindsight_added 0/40 FAIL' in _row(cert, 'invariance').detail
    assert 'info_was_available 40/40' in _row(cert, 'rejection').detail  # it does use the fact


async def test_a_judge_that_ignores_the_timeline_fails_the_must_fail_family() -> None:
    cert = await _certify(temporal_task, ignores_the_fact)
    assert 'info_was_available 0/40 FAIL' in _row(cert, 'rejection').detail, cert.table()
    assert _row(cert, 'invariance').status == 'PASS'  # hindsight cannot catch a judge that reads nothing


async def test_a_judge_that_reads_only_the_reply_misses_duplicate_side_effects() -> None:
    cert = await _certify(retry_task, reads_the_reply)
    rejection = _row(cert, 'rejection')
    assert cert.verdict == 'INADMISSIBLE', cert.table()
    assert 'duplicate_side_effect 0/40 FAIL' in rejection.detail
    assert 'retry_changed_arguments 0/40 FAIL' in rejection.detail


async def test_the_recovered_retry_catches_an_overly_strict_judge() -> None:
    cert = await _certify(retry_task, no_failures_allowed)
    assert _row(cert, 'rejection').status == 'PASS', cert.table()
    assert 'retried_and_recovered 0/40 FAIL' in _row(cert, 'invariance').detail


def test_controls_change_the_trace_and_never_the_answer() -> None:
    for task in (temporal_task, retry_task):
        cases = task.episodes(8)
        for control in task.CONTROLS:
            for case in cases:
                changed = control.make_case(case, cases, random.Random(0))
                assert changed is not None and changed.output == case.output and changed.inputs != case.inputs


def test_the_hindsight_pair_differs_only_in_the_timestamp() -> None:
    cases = temporal_task.episodes(8)
    after, before = temporal_task.CONTROLS
    for case in cases:
        late = after.make_case(case, cases, random.Random(0))
        early = before.make_case(case, cases, random.Random(0))
        assert late is not None and early is not None
        decided = datetime.fromisoformat(case.inputs['decided_at'])
        [added_late] = [e for e in late.inputs['events'] if e not in case.inputs['events']]
        [added_early] = [e for e in early.inputs['events'] if e not in case.inputs['events']]
        assert {**added_late, 'time': ''} == {**added_early, 'time': ''}
        assert datetime.fromisoformat(added_late['time']) > decided > datetime.fromisoformat(added_early['time'])
        for events in (late.inputs['events'], early.inputs['events']):
            times = [datetime.fromisoformat(e['time']) for e in events]
            assert times == sorted(times)


def test_rewrites_skip_cases_they_do_not_apply_to() -> None:
    late, early = hindsight_pair(lambda inputs: None)
    duplicate = duplicate_call('issue_refund', lambda result: result)
    case = retry_task.episodes(2)[1]  # an upgrade: no refund call to duplicate
    assert duplicate.make_case(case, [case], random.Random(0)) is None
    assert late.make_case(temporal_task.episodes(1)[0], [], random.Random(0)) is None
    assert early.rewrite('not a mapping') is None


def test_the_codex_run_fits_its_budget() -> None:
    total = sum(planned_calls(task.episodes(16), task.CONTROLS) for task in (temporal_task, retry_task))
    assert total == 16 + 2 * 16 + 16 + 3 * 16 <= 150
