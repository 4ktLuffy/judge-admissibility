"""Conversation controls: obligations from earlier turns, and credit for an agent's own repair."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from typing import Any

from pydantic_evals.evaluators import LLMJudge

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from conversation_run import scripted  # noqa: E402
from conversation_task import CONTROLS, RUBRIC, episodes, last_turn_only, sound, unforgiving  # noqa: E402

from pydantic_evals_admissibility import certify_judge  # noqa: E402
from pydantic_evals_admissibility._conversation import insert_turns, recovery_profile  # noqa: E402


def judge(decide: Any) -> LLMJudge:
    return LLMJudge(rubric=RUBRIC, model=scripted(decide), include_input=True)


def test_inserted_turns_go_before_the_last_message_and_the_reply_never_changes() -> None:
    cases = episodes(8)
    for control in CONTROLS:
        for case in cases:
            changed = control.make_case(case, cases, random.Random(0))
            if changed is None:
                continue
            before, after = case.inputs['conversation'], changed.inputs['conversation']
            assert changed.output == case.output and after[-1] == before[-1] and len(after) > len(before)
            assert after[: len(before) - 1] == before[:-1]  # nothing earlier is touched

    assert [c.inputs for c in cases] == [c.inputs for c in episodes(8)]  # the originals are not mutated


def test_controls_apply_only_where_they_mean_something() -> None:
    cases = episodes(16)
    applied = {
        c.name: sum(c.make_case(e, cases, random.Random(0)) is not None for e in cases)
        for c in CONTROLS  # type: ignore[attr-defined]
    }
    # consent cannot be withdrawn twice; the error controls skip episodes with an error of their own
    assert applied == {
        'consent_revoked': 12,
        'obligation_irrelevant_turn': 16,
        'error_then_repaired': 12,
        'error_unrepaired': 12,
    }


def test_error_controls_insert_no_customer_turn_and_differ_only_by_the_repair() -> None:
    """The error is volunteered, so nothing new needs answering; the repair is the one difference."""
    cases = episodes(16)
    by_name = {c.name: c for c in CONTROLS}  # type: ignore[attr-defined]
    for case in cases:
        made = [
            by_name[n].make_case(case, cases, random.Random(0)) for n in ('error_then_repaired', 'error_unrepaired')
        ]
        if made[0] is None:
            assert made[1] is None
            continue
        repaired, unrepaired = (m.inputs['conversation'][len(case.inputs['conversation']) - 1 : -1] for m in made)
        assert all(t['role'] == 'agent' for t in repaired + unrepaired)
        assert repaired[:-1] == unrepaired and len(repaired) == 2 and 'Correction' in repaired[-1]['text']
        assert not any('?' in t['text'] for t in repaired)


def test_insert_turns_skips_inputs_without_a_conversation() -> None:
    control = insert_turns(lambda inputs: [{'role': 'agent', 'text': 'hi'}], 'x', 'must_hold')
    assert control.rewrite('just a string') is None
    assert control.rewrite({'conversation': []}) == {'conversation': [{'role': 'agent', 'text': 'hi'}]}
    assert insert_turns(lambda inputs: None, 'x', 'must_hold').rewrite({'conversation': [{}]}) is None


async def test_a_judge_that_holds_the_reply_to_the_conversation_is_admissible_and_recovery_aware() -> None:
    cert = await certify_judge(judge(sound), episodes(40), controls=CONTROLS, repeats=1)
    assert cert.verdict == 'ADMISSIBLE', cert.table()
    assert recovery_profile(cert).kind == 'recovery-aware'


async def test_a_judge_that_reads_only_the_last_turn_misses_revoked_consent() -> None:
    cert = await certify_judge(judge(last_turn_only), episodes(40), controls=CONTROLS, repeats=1)
    rejection = next(c for c in cert.checks if c.name == 'rejection')
    assert cert.verdict == 'INADMISSIBLE' and 'consent_revoked 0/30 FAIL' in rejection.detail, cert.table()
    # It passes every known-good reply, so acceptance cannot see the problem; only the controls do.
    assert next(c for c in cert.checks if c.name == 'acceptance').status == 'PASS'
    assert recovery_profile(cert).kind == 'blind to unrepaired errors'


async def test_an_unforgiving_judge_fails_repaired_trajectories() -> None:
    cert = await certify_judge(judge(unforgiving), episodes(40), controls=CONTROLS, repeats=1)
    invariance = next(c for c in cert.checks if c.name == 'invariance')
    assert cert.verdict == 'INADMISSIBLE' and 'error_then_repaired 0/30 FAIL' in invariance.detail, cert.table()
    assert 'obligation_irrelevant_turn 40/40' in invariance.detail  # it is not swayed by any inserted turn
    profile = recovery_profile(cert)
    assert profile.kind == 'unforgiving' and profile.catches_unrepaired.status == 'PASS'


async def test_a_judge_that_punishes_the_word_correction_is_backwards() -> None:
    def backwards(inputs: Any, reply: str) -> bool:
        return not any('Correction' in t['text'] for t in inputs['conversation'][4:])

    profile = recovery_profile(await certify_judge(judge(backwards), episodes(40), controls=CONTROLS, repeats=1))
    assert profile.kind == 'backwards', profile


async def test_repaired_trajectories_count_only_where_the_original_passed() -> None:
    """A judge that fails some episodes for an unrelated reason is not called unforgiving for them."""

    def dislikes_address_changes(inputs: Any, reply: str) -> bool:
        return sound(inputs, reply) and not any('change of plan' in t['text'] for t in inputs['conversation'])

    cert = await certify_judge(judge(dislikes_address_changes), episodes(40), controls=CONTROLS, repeats=1)
    profile = recovery_profile(cert)
    assert profile.forgives_repaired.trials == 20 and profile.forgives_repaired.successes == 20
    assert profile.kind == 'recovery-aware'
