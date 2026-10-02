"""Controls have to match the rubric: a sound style judge fails the default correctness controls."""

from __future__ import annotations

from judges import judge

from pydantic_evals_admissibility import JudgeCase, MismatchedOutput, Rewrite, WhitespaceReformat, certify_judge

CASES = [JudgeCase(f'c{i}', f'request {i}', f'You asked about item {i}; here is what we found.') for i in range(40)]


def friendly(output: str, expected: str) -> bool:
    """A sound judge of the rubric "friendly, second person": content is irrelevant to it."""
    return output.strip().startswith('You ')


STYLE_CONTROLS = (
    Rewrite(lambda o: f'ERR_OK status=0x00 len={len(o)}', 'debug_string', 'must_fail'),
    Rewrite(lambda o: '', 'empty', 'must_fail'),
    MismatchedOutput(kind='must_hold'),
    WhitespaceReformat(),
)


async def test_a_sound_style_judge_is_admissible_with_controls_for_its_rubric() -> None:
    cert = await certify_judge(judge(friendly), CASES, controls=STYLE_CONTROLS)
    assert cert.verdict == 'ADMISSIBLE', cert.table()


async def test_the_default_correctness_controls_would_wrongly_fail_it() -> None:
    """Another case's friendly reply is still friendly; treating it as must-fail blames the judge."""
    cert = await certify_judge(judge(friendly), CASES)
    rejection = next(c for c in cert.checks if c.name == 'rejection')
    assert cert.verdict == 'INADMISSIBLE' and 'mismatched_output 0/40' in rejection.detail


async def test_a_style_judge_swayed_by_content_fails_the_must_hold_control() -> None:
    def content_sensitive(output: str, expected: str) -> bool:
        return output.strip().startswith('You ') and 'item 1' in output  # grades content, not style

    cert = await certify_judge(judge(content_sensitive), CASES, controls=STYLE_CONTROLS)
    assert cert.verdict == 'INADMISSIBLE'


async def test_the_doctor_points_at_the_controls_not_the_judge() -> None:
    from pydantic_evals_admissibility import diagnose

    advice = diagnose(await certify_judge(judge(friendly), CASES))
    assert any("MismatchedOutput(kind='must_hold')" in a for a in advice), advice


def test_the_controls_hint_stays_silent_for_a_correctness_rubric() -> None:
    """Found on the real support-task certificate: the hint misfired on a correctness rubric."""
    from pydantic_evals.evaluators import LLMJudge

    from pydantic_evals_admissibility import Certificate, Check, diagnose

    checks = (Check('rejection', 'FAIL', 10, 20, (0.26, 0.74), 0.8, 'mismatched_output 0/10, empty_output 10/10'),)
    cert = Certificate('INADMISSIBLE', checks, ())
    correctness = LLMJudge(rubric="The reply correctly answers the customer's question according to the policy.")
    style = LLMJudge(rubric='The explanation is in a second-person or friendly style.')
    assert not any('must_hold' in a for a in diagnose(cert, correctness))
    assert any('must_hold' in a for a in diagnose(cert, style))
    assert any('must_hold' in a for a in diagnose(cert))  # no rubric to read: offered as a maybe
