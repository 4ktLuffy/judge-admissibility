"""Is Codex position-biased as a comparison judge? Real right and wrong replies, both orders.

    PYTHONPATH=.:bench <venv>/bin/python bench/pairwise_codex.py

For every question where the cached runs produced both a correct and an incorrect reply (ground
truth decides which), one pair: the first correct and the first incorrect reply found. Each pair
is judged once each way round by a `PairwiseJudge` on Codex `gpt-5.6-luna`. Writes
`results/pairwise_codex.json`.
"""

from __future__ import annotations

import asyncio
import json

from codex_judge import TOKENS, codex_model
from optimize import BASELINE_PROMPT, CACHE_PATH, REPEATS, ROOT, Cache, split
from task import is_correct

from pydantic_evals_admissibility import PairCase, PairwiseJudge, certify_pairwise, first_position_rate


def pairs() -> list[PairCase]:
    cache = json.loads(CACHE_PATH.read_text())
    run = json.loads((ROOT / 'results' / 'optimize.json').read_text())
    prompts = [BASELINE_PROMPT] + [c['prompt'] for c in run['candidates']]
    train, test = split()
    out = []
    for q in train + test:
        right = wrong = None
        for prompt in prompts:
            for r in range(REPEATS):
                reply = cache.get(Cache.key('agent', prompt, q.name, q.question, str(r)))
                if reply is None:
                    continue
                if is_correct(q, reply):
                    right = right or reply
                else:
                    wrong = wrong or reply
        if right and wrong:
            out.append(PairCase(q.name, q.question, right, wrong))
    return out


def terse(cases: list[PairCase]) -> list[PairCase]:
    """The same pairs with only the final answer line, so length and working give nothing away."""
    from task import final_answer

    return [
        PairCase(c.name, c.inputs, f'Answer: {final_answer(c.better)}', f'Answer: {final_answer(c.worse)}')
        for c in cases
        if final_answer(c.better) and final_answer(c.worse) and final_answer(c.better) != final_answer(c.worse)
    ]


async def main() -> None:
    import sys

    cases = pairs()
    if '--terse' in sys.argv:
        cases = terse(cases)
    print(f'{len(cases)} questions with both a right and a wrong reply')
    judge = PairwiseJudge(rubric='Which answer correctly answers the question?', model=codex_model())
    cert = await certify_pairwise(judge, cases, max_concurrency=4)
    print(cert.table())
    rate = first_position_rate(cert)
    print(
        f'chose the answer shown first in {rate[0]:.2f} of presentations, 95% interval [{rate[1][0]:.2f}, {rate[1][1]:.2f}]'
        if rate
        else ''
    )
    print(f'{len(TOKENS)} calls, {sum(TOKENS):,} tokens')
    name = 'pairwise_codex.terse.json' if '--terse' in sys.argv else 'pairwise_codex.json'
    (ROOT / 'results' / name).write_text(json.dumps({
        **cert.to_dict(), 'pairs': len(cases), 'first_position_rate': rate, 'calls': len(TOKENS),
    }, indent=2, default=str))  # fmt: skip


if __name__ == '__main__':
    asyncio.run(main())
