"""`evidence_budget`: the smallest projection of a trace that keeps every verdict that matters."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from evidence_budget import EVIDENCE_CONTROLS, grades_the_work, parts, project, scripted  # noqa: E402
from evidence_task import episodes  # noqa: E402

from pydantic_evals_admissibility._evidence_budget import check_budget, evidence_budget  # noqa: E402

CASES = episodes(8)
ALL = ('customer', 'lookup.span_id', 'lookup.result', 'action.span_id', 'action.arguments', 'action.result')


def needs_everything(inputs: dict[str, Any], reply: str) -> bool:
    """Right about the work, but fails any trace with a component missing."""
    complete = set(parts(inputs)) == set(parts(CASES[0].inputs)) | set(parts(CASES[1].inputs))
    return complete and grades_the_work(inputs, reply)


def trusts_the_claim_without_evidence(inputs: dict[str, Any], reply: str) -> bool:
    """Grades the action's result when shown it; shown none, passes the confident reply."""
    action = next(c for c in inputs['tool_calls'] if c['tool'] in ('issue_refund', 'cancel_subscription'))
    return grades_the_work(inputs, reply) if 'result' in action else bool(reply.strip())


def test_parts_and_project_name_every_component() -> None:
    names = set(parts(CASES[0].inputs)) | set(parts(CASES[1].inputs))
    assert names == set(ALL)
    projected = project(CASES[0].inputs, frozenset({'action.result'}))
    assert 'customer' not in projected and all(set(c) <= {'tool', 'result'} for c in projected['tool_calls'])
    assert projected['tool_calls'][0] == {'tool': 'lookup_order'}  # the tool name always stays


async def test_a_judge_that_needs_only_the_action_result_keeps_only_that() -> None:
    budget = await evidence_budget(scripted(grades_the_work), CASES, EVIDENCE_CONTROLS, parts=parts, project=project)
    assert budget.kept == ('action.result',), budget.summary()
    assert budget.complete and set(budget.dropped) == set(ALL) - {'action.result'}
    assert 0.5 < budget.saved < 1
    kept_trial = next(t for t in budget.trials if t.component == 'action.result')
    assert not kept_trial.dropped and kept_trial.changed and kept_trial.calls < budget.judgments  # stopped early
    check = await check_budget(
        scripted(grades_the_work), episodes(16)[8:], EVIDENCE_CONTROLS, kept=budget.kept, project=project
    )
    assert check.agree == check.judgments and not check.disagreements


async def test_a_judge_whose_verdicts_move_when_anything_is_dropped_keeps_everything() -> None:
    budget = await evidence_budget(scripted(needs_everything), CASES, EVIDENCE_CONTROLS, parts=parts, project=project)
    assert set(budget.kept) == set(ALL) and budget.saved == 0, budget.summary()
    assert all(t.changed and not t.dropped for t in budget.trials)


async def test_without_controls_the_budget_drops_the_evidence_the_verdict_should_rest_on() -> None:
    """On known-good episodes alone, a judge that trusts the claim keeps its verdicts with no evidence."""
    judge = scripted(trusts_the_claim_without_evidence)
    bare = await evidence_budget(judge, CASES, (), parts=parts, project=project)
    assert 'action.result' not in bare.kept, bare.summary()
    with_controls = await evidence_budget(judge, CASES, EVIDENCE_CONTROLS, parts=parts, project=project)
    assert with_controls.kept == ('action.result',), with_controls.summary()
    check = await check_budget(judge, episodes(16)[8:], EVIDENCE_CONTROLS, kept=bare.kept, project=project)
    assert check.agree < check.judgments  # the bare budget moves the controls' verdicts


async def test_the_call_budget_is_respected_and_reported() -> None:
    budget = await evidence_budget(
        scripted(grades_the_work), CASES, EVIDENCE_CONTROLS, parts=parts, project=project, max_calls=30
    )
    assert budget.calls <= 30 and not budget.complete
    with pytest.raises(ValueError):
        await evidence_budget(scripted(grades_the_work), CASES, EVIDENCE_CONTROLS, parts=parts, project=project,
                              max_calls=3)  # fmt: skip


async def test_an_order_naming_an_unknown_component_is_refused() -> None:
    with pytest.raises(ValueError, match='never returns'):
        await evidence_budget(scripted(grades_the_work), CASES, (), parts=parts, project=project, order=['nope'])
