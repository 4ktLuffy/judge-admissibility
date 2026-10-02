"""Measure the agent's starting prompt on the 40 questions, against ground truth only.

    PYTHONPATH=.:bench <venv>/bin/python bench/baseline.py [--smoke]

Each question runs `REPEATS` times, because the run-to-run spread is what any later "improvement"
has to beat. Writes `results/task_baseline.json` with every reply.
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

from codex_judge import TOKENS, codex_text_model
from pydantic_ai import Agent
from task import is_correct, questions

BASELINE_PROMPT = 'Answer the question. End your reply with a line "Answer: <answer>".'
REPEATS = 2


async def run(prompt: str, qs, repeats: int, concurrency: int = 4) -> list[dict]:  # type: ignore[no-untyped-def]
    agent = Agent(codex_text_model(), instructions=prompt)
    limit = asyncio.Semaphore(concurrency)

    async def one(q, r):  # type: ignore[no-untyped-def]
        async with limit:
            try:
                reply = (await agent.run(q.question)).output
                error = None
            except Exception as exc:  # a failed call is recorded, not retried into a better number
                reply, error = '', f'{type(exc).__name__}: {exc}'[:200]
        return {'question': q.name, 'kind': q.kind, 'repeat': r, 'reply': reply, 'error': error,
                'correct': is_correct(q, reply)}  # fmt: skip

    return await asyncio.gather(*(one(q, r) for q in qs for r in range(repeats)))


async def main() -> None:
    smoke = '--smoke' in sys.argv
    qs = questions()
    qs = [qs[0], qs[30]] if smoke else qs
    started = time.monotonic()
    rows = await run(BASELINE_PROMPT, qs, 1 if smoke else REPEATS)
    by_kind: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        by_kind[row['kind']].append(row['correct'])
    total = [row['correct'] for row in rows]
    print(f'{len(rows)} replies in {time.monotonic() - started:.0f}s, {sum(TOKENS):,} tokens '
          f'({sum(TOKENS) // max(len(TOKENS), 1)} per call), errors={sum(bool(r["error"]) for r in rows)}')  # fmt: skip
    print(f'ground-truth accuracy: {sum(total)}/{len(total)}')
    for kind, ok in by_kind.items():
        print(f'  {kind:<11} {sum(ok)}/{len(ok)}')
    if not smoke:
        flips = 0
        by_q: dict[str, set[bool]] = defaultdict(set)
        for row in rows:
            by_q[row['question']].add(row['correct'])
        flips = sum(len(v) > 1 for v in by_q.values())
        print(f'questions whose correctness differs between the {REPEATS} runs: {flips}/40')
        out = Path(__file__).parent.parent / 'results' / 'task_baseline.json'
        out.write_text(json.dumps({'prompt': BASELINE_PROMPT, 'tokens': sum(TOKENS), 'rows': rows}, indent=2))
    else:
        for row in rows:
            print('---', row['question'], row['correct'], repr(row['reply'][-160:]), row['error'])


if __name__ == '__main__':
    asyncio.run(main())
