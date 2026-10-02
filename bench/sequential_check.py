"""Does stopping early save calls without certifying worse judges? Scripted judges, many seeds.

PYTHONPATH=.:tests <venv>/bin/python bench/sequential_check.py
"""

from __future__ import annotations

import asyncio
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'tests'))
from judges import judge  # noqa: E402

from pydantic_evals_admissibility import JudgeCase, certify_judge  # noqa: E402

CASES = [JudgeCase(f'q{i}', f'question {i}', f'answer {i}', expected_output=f'answer {i}') for i in range(60)]


def leaky(pass_wrong: float, fail_right: float, seed: int):  # type: ignore[no-untyped-def]
    """Right answers pass unless it slips (`fail_right`); wrong ones pass with `pass_wrong`.

    The slip for each (answer, n-th time asked) is fixed by a hash, not drawn in call order, so a
    full and a sequential certificate of the same judge see identical verdicts and any difference
    between them is the procedure's, not chance's.
    """
    asked: dict[tuple[str, str], int] = {}

    def decide(output: str, expected: str) -> bool:
        n = asked[(output, expected)] = asked.get((output, expected), -1) + 1
        u = random.Random(f'{seed}|{output}|{expected}|{n}').random()
        right = output.strip() == expected.strip()
        return u >= fail_right if right else u < pass_wrong

    return decide


async def run(label: str, pass_wrong: float, fail_right: float, seeds: int) -> None:
    full_admit = seq_admit = agree = 0
    calls_full = calls_seq = 0
    for s in range(seeds):
        full = await certify_judge(judge(leaky(pass_wrong, fail_right, s)), CASES, repeats=2, seed=s)
        seq = await certify_judge(judge(leaky(pass_wrong, fail_right, s)), CASES, repeats=2, seed=s, batch_size=15)
        full_admit += full.admissible
        seq_admit += seq.admissible
        agree += full.verdict == seq.verdict or (seq.verdict == 'UNVALIDATED' and full.verdict == 'UNVALIDATED')
        calls_full += full.calls or 0
        calls_seq += seq.calls or 0
    print(f'{label:46} ADMISSIBLE full {full_admit:3}/{seeds}  sequential {seq_admit:3}/{seeds}  '
          f'same verdict {agree:3}/{seeds}  calls {calls_seq / calls_full:.0%} of full')  # fmt: skip


async def main() -> None:
    await run('sound (passes 0% wrong, fails 2% right)', 0.0, 0.02, 100)
    await run('broken (passes 60% of wrong answers)', 0.6, 0.02, 100)
    await run('borderline bad (passes 25% wrong; bar is 20%)', 0.25, 0.02, 100)
    await run('borderline good (passes 10% wrong)', 0.10, 0.02, 100)


if __name__ == '__main__':
    asyncio.run(main())
