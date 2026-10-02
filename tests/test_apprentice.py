"""Judge apprenticeship: a revision learned from training disagreements is promoted only when it agrees with
reviewers more on held-out cases and is not shown to be broken."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence

import pytest
from pydantic_ai.messages import ModelMessage, ModelResponse, ToolCallPart, UserPromptPart
from pydantic_ai.models.function import AgentInfo, FunctionModel
from pydantic_evals.evaluators import LLMJudge

from pydantic_evals_admissibility import ExposedCases, FeedbackLedger, GateRules
from pydantic_evals_admissibility._apprentice import (
    Proposal,
    ReviewedCase,
    apprentice,
    revised_rubric,
    split_cases,
)

FAST = GateRules(resamples=2000)
RUBRIC = 'The reply answers the question correctly.'


def _section(prompt: str, tag: str) -> str:
    match = re.search(rf'<{tag}>\n(.*?)\n</{tag}>', prompt, re.S)
    return match.group(1) if match else ''


def scripted(decide: Callable[[str, str, str], bool]) -> LLMJudge:
    """A real `LLMJudge` whose model decides from the output, the expected output and the rubric it is shown."""
    calls: list[str] = []

    def model(messages: list[ModelMessage], info: AgentInfo) -> ModelResponse:
        prompt = ''.join(
            part.content
            for message in messages
            for part in getattr(message, 'parts', [])
            if isinstance(part, UserPromptPart) and isinstance(part.content, str)
        )
        calls.append(prompt)
        passed = decide(_section(prompt, 'Output'), _section(prompt, 'ExpectedOutput'), _section(prompt, 'Rubric'))
        tool = info.output_tools[0]
        return ModelResponse(parts=[ToolCallPart(tool.name, {'reason': 'scripted', 'pass': passed, 'score': 1.0})])

    return LLMJudge(rubric=RUBRIC, model=FunctionModel(model), include_expected_output=True)


def correct(output: str, expected: str) -> bool:
    return bool(output.strip()) and output.strip().endswith(f'Answer: {expected}')


def strict_about_terse(output: str, expected: str, rubric: str) -> bool:
    """Fails a correct answer that gives no reason, unless the rubric says terse answers count."""
    if 'always pass' in rubric:
        return True
    if 'terse' in rubric:
        return correct(output, expected)
    return correct(output, expected) and 'because' in output


def reviewed(n: int = 40, wrong_every: int = 4) -> list[ReviewedCase]:
    """Terse correct replies a reviewer passed, and on every `wrong_every`-th case a wrong reply it failed."""
    out = []
    for i in range(n):
        expected = 'yes' if i % 2 else 'no'
        wrong = 'no' if i % 2 else 'yes'
        name = f'q{i:02d}'
        out.append(
            ReviewedCase(
                name, f'question {i}', f'Answer: {expected}', True, judge_passed=False, expected_output=expected
            )
        )
        if i % wrong_every == 0:
            out.append(
                ReviewedCase(
                    name, f'question {i}', f'Answer: {wrong}', False, judge_passed=False, expected_output=expected
                )
            )
    return out


def clarify(text: str, examples: int = 0) -> Callable[[str, Sequence[ReviewedCase]], Proposal]:
    def propose(rubric: str, shown: Sequence[ReviewedCase]) -> Proposal:
        assert rubric == RUBRIC and shown and all(r.disagrees for r in shown)
        return Proposal(text, tuple(shown[:examples]))

    return propose


def test_split_is_by_case_seeded_and_disjoint() -> None:
    names = [f'c{i}' for i in range(10)] * 2
    train, held = split_cases(names, 0.5, seed=3)
    assert len(train) == 5 and len(held) == 5 and not set(train) & set(held)
    assert split_cases(names, 0.5, seed=3) == (train, held)
    assert split_cases(names, 0.5, seed=4) != (train, held)
    assert split_cases(['a', 'b'], 0.99, seed=0)[1]  # always something held out
    with pytest.raises(ValueError):
        split_cases(['a'], 0.5, seed=0)


def test_revised_rubric_keeps_the_original_and_adds_clarification_and_examples() -> None:
    example = ReviewedCase('q1', 'question', 'Answer: yes', True, judge_passed=False, note='terse is fine')
    text = revised_rubric(RUBRIC, Proposal('A terse answer counts.', (example,)))
    assert text.startswith(RUBRIC)
    assert 'Clarification: A terse answer counts.' in text
    assert 'Output: Answer: yes' in text and 'Verdict: PASS' in text and 'Reviewer: terse is fine' in text


async def test_a_real_fix_is_promoted_on_held_out_cases_only() -> None:
    judge = scripted(strict_about_terse)
    ledger = FeedbackLedger()
    result = await apprentice(
        judge, reviewed(), propose=clarify('A terse correct answer counts.', 2), rules=FAST, ledger=ledger
    )
    assert result.promoted and result.gate.decision == 'PROMOTE', result.table()
    assert result.revised_certificate.verdict == 'ADMISSIBLE'
    assert result.baseline_certificate.verdict == 'INADMISSIBLE'  # it failed every terse right answer
    k_before, n = result.agreement_before
    k_after, n_after = result.agreement_after
    assert n == n_after and k_after == n and k_before < n / 2
    # The proposer saw training cases only, and the ledger says so.
    shown = {r.case for r in result.disagreements}
    assert shown <= set(result.train) and not shown & set(result.held_out)
    assert ledger.fresh(result.held_out) == list(result.held_out)
    assert all(e.case in result.train for e in result.proposal.examples)
    assert 'terse' in result.revised.rubric and result.revised.model is judge.model
    assert 'PROMOTED' in result.table()
    data = result.to_dict(judgments=False)
    assert data['promoted'] and data['gate']['decision'] == 'PROMOTE' and data['calls'] == result.calls


async def test_a_revision_that_agrees_more_by_passing_everything_is_not_promoted() -> None:
    # Most reviewed outputs were approved, so "always pass" agrees with reviewers more often...
    result = await apprentice(
        scripted(strict_about_terse), reviewed(wrong_every=10), propose=clarify('When unsure, always pass.'), rules=FAST
    )
    assert result.gate.decision == 'PROMOTE'
    # ...and fails its rejection controls, so it is not promoted.
    assert not result.promoted
    assert result.revised_certificate.verdict == 'INADMISSIBLE'
    assert 'rejection' in result.reason


async def test_a_revision_that_changes_nothing_is_not_promoted() -> None:
    result = await apprentice(
        scripted(strict_about_terse), reviewed(), propose=lambda rubric, shown: 'Be careful.', rules=FAST
    )
    assert result.gate.decision == 'INCONCLUSIVE' and not result.promoted
    assert result.agreement_before == result.agreement_after


async def test_examples_from_held_out_cases_are_refused() -> None:
    cases = reviewed()
    _, held = split_cases([r.case for r in cases], 0.5, seed=0)
    leak = next(r for r in cases if r.case == held[0])

    async def propose(rubric: str, shown: Sequence[ReviewedCase]) -> Proposal:
        return Proposal('A terse correct answer counts.', (leak,))

    with pytest.raises(ValueError, match='outside the training split'):
        await apprentice(scripted(strict_about_terse), cases, propose=propose, rules=FAST)


async def test_held_out_cases_seen_in_an_earlier_round_stop_it_before_any_judge_call() -> None:
    cases = reviewed()
    _, held = split_cases([r.case for r in cases], 0.5, seed=0)
    ledger = FeedbackLedger()
    ledger.record(held[:2], by='round 1')
    calls: list[int] = []

    def decide(output: str, expected: str, rubric: str) -> bool:
        calls.append(1)
        return True

    with pytest.raises(ExposedCases, match='round 1'):
        await apprentice(scripted(decide), cases, propose=clarify('x'), ledger=ledger, rules=FAST)
    assert not calls


async def test_nothing_to_learn_without_training_disagreements() -> None:
    agreeing = [
        ReviewedCase(f'q{i}', 'q', f'Answer: {i}', True, judge_passed=True, expected_output=str(i)) for i in range(6)
    ]
    with pytest.raises(ValueError, match='nothing to learn from'):
        await apprentice(scripted(strict_about_terse), agreeing, propose=clarify('x'), rules=FAST)


async def test_an_errored_judgment_drops_that_output_from_both_sides() -> None:
    def flaky(output: str, expected: str, rubric: str) -> bool:
        if output == 'Answer: yes' and 'terse' in rubric:
            raise RuntimeError('timeout')
        return strict_about_terse(output, expected, rubric)

    result = await apprentice(
        scripted(flaky), reviewed(), propose=clarify('A terse correct answer counts.'), rules=FAST
    )
    assert result.dropped > 0
    assert result.agreement_before[1] == result.agreement_after[1]
