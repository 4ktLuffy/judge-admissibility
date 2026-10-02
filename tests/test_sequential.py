"""Sequential certification: same verdict as judging everything, fewer calls for a broken judge."""

from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest
from judges import judge

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from sequential_check import CASES, leaky  # noqa: E402

from pydantic_evals_admissibility import certify_judge


@pytest.mark.parametrize(
    ('pass_wrong', 'fail_right'),
    [(0.0, 0.02), (0.6, 0.02), (0.25, 0.02), (0.10, 0.02)],
    ids=['sound', 'broken', 'borderline-bad', 'borderline-good'],
)
async def test_same_verdict_as_judging_everything(pass_wrong: float, fail_right: float) -> None:
    for seed in range(5):
        full = await certify_judge(judge(leaky(pass_wrong, fail_right, seed)), CASES, repeats=2, seed=seed)
        seq = await certify_judge(
            judge(leaky(pass_wrong, fail_right, seed)), CASES, repeats=2, seed=seed, batch_size=15
        )
        assert seq.verdict == full.verdict, (seed, full.table(), seq.table())


async def test_a_broken_judge_is_stopped_after_the_first_batch() -> None:
    cert = await certify_judge(judge(leaky(0.6, 0.02, 0)), CASES, repeats=2, batch_size=15)
    assert cert.verdict == 'INADMISSIBLE'
    assert cert.calls is not None and cert.planned is not None
    assert cert.calls <= cert.planned // 4 + 1, (cert.calls, cert.planned)


async def test_a_sound_judge_is_never_stopped_early() -> None:
    cert = await certify_judge(judge(leaky(0.0, 0.0, 0)), CASES, repeats=2, batch_size=15)
    assert cert.verdict == 'ADMISSIBLE' and cert.calls == cert.planned


def test_scripted_judge_is_deterministic_per_answer() -> None:
    a, b = leaky(0.5, 0.0, 3), leaky(0.5, 0.0, 3)
    outs = [f'answer {random.Random(i).randint(0, 9)}' for i in range(20)]
    assert [a(o, 'answer 1') for o in outs] == [b(o, 'answer 1') for o in outs]
