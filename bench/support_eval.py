"""A strong judge on a realistic task: certify judges of a support agent, and test position bias.

    PYTHONPATH=.:bench <venv>/bin/python bench/support_eval.py

1. The agent (Codex `gpt-5.6-luna`, no reasoning) answers the 40 support questions twice.
2. Three `LLMJudge`s are certified on those 80 real replies, ground truth as the labels:
   weak (no reasoning, sees the question), strong (reasoning effort high, sees the question),
   and strong but default (`include_input=False`). Sequential, batches of 10.
3. Position bias: a `PairwiseJudge`, weak and strong, compares the right answer with a plausible
   wrong one (the refund without the restocking fee, the other side of the shipping threshold,
   the opposite yes/no), each pair both ways round.

Agent replies are cached in `results/support-cache.json`; judge calls are not, since a cache
would hand back the same verdict to a repeated question and fake the stability check. Each stage
is written to `results/support_eval.json` as it finishes.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from pathlib import Path
from typing import Any

from codex_judge import TOKENS, codex_model, codex_text_model
from pydantic_ai import Agent
from pydantic_evals.evaluators import LLMJudge
from support_task import SupportCase, cases, is_correct

from pydantic_evals_admissibility import (
    HumanLabel,
    JudgeCase,
    PairCase,
    PairwiseJudge,
    certify_judge,
    certify_pairwise,
    diagnose,
    first_position_rate,
)

ROOT = Path(__file__).parent.parent
CACHE = ROOT / 'results' / 'support-cache.json'
OUT = ROOT / 'results' / 'support_eval.json'
AGENT_PROMPT = (
    'You are a customer support agent. Answer the customer using only the store policy given. '
    'End your reply with a line "Answer: <answer>", where the answer is yes or no, or a dollar amount.'
)
RUBRIC = "The reply correctly answers the customer's question according to the store policy."
REPEATS = 2


def fmt(case: SupportCase, answer: str) -> str:
    return answer if answer in ('yes', 'no') else f'${answer}'


def plausible_wrong(case: SupportCase) -> str:
    """The mistake a support agent would make, not a random one."""
    p = case.policy
    if case.answer in ('yes', 'no'):
        return 'no' if case.answer == 'yes' else 'yes'
    if case.kind == 'shipping':
        return f'{p.shipping_fee:.2f}' if case.answer == '0.00' else '0.00'
    price = float(case.answer) * 100 / (100 - p.restocking_pct)
    return f'{price:.2f}'  # forgot the restocking fee


async def agent_replies(cs: list[SupportCase]) -> dict[str, list[str]]:
    cache: dict[str, str] = json.loads(CACHE.read_text()) if CACHE.exists() else {}
    agent = Agent(codex_text_model(), instructions=AGENT_PROMPT)
    limit = asyncio.Semaphore(6)

    async def one(c: SupportCase, r: int) -> str:
        key = hashlib.sha256(f'{AGENT_PROMPT}|{c.inputs}|{r}'.encode()).hexdigest()
        if key not in cache:
            async with limit:
                cache[key] = (await agent.run(c.inputs)).output
            CACHE.write_text(json.dumps(cache))
        return cache[key]

    replies = await asyncio.gather(*(one(c, r) for c in cs for r in range(REPEATS)))
    out: dict[str, list[str]] = {}
    for (c, _), reply in zip(((c, r) for c in cs for r in range(REPEATS)), replies):
        out.setdefault(c.name, []).append(reply)
    return out


def save(results: dict[str, Any]) -> None:
    OUT.write_text(json.dumps(results, indent=2, default=str))


async def main() -> None:
    cs = cases()
    by_name = {c.name: c for c in cs}
    results: dict[str, Any] = json.loads(OUT.read_text()) if OUT.exists() else {}

    replies = await agent_replies(cs)
    truth = {n: [is_correct(by_name[n], r) for r in rs] for n, rs in replies.items()}
    flat = [v for vs in truth.values() for v in vs]
    results['agent'] = {'correct': sum(flat), 'replies': len(flat)}
    print(f'agent: {sum(flat)}/{len(flat)} correct', flush=True)
    save(results)

    good = {
        c.name: next((r for r, ok in zip(replies[c.name], truth[c.name]) if ok), f'Answer: {fmt(c, c.answer)}')
        for c in cs
    }
    judge_cases = [JudgeCase(c.name, c.inputs, good[c.name], expected_output=c.answer) for c in cs]
    labels = [HumanLabel(n, r, ok) for n in replies for r, ok in zip(replies[n], truth[n])]

    judges = {
        'weak (no reasoning, sees the question)': LLMJudge(
            rubric=RUBRIC, model=codex_model(effort='none'), include_input=True
        ),
        'strong (reasoning high, sees the question)': LLMJudge(
            rubric=RUBRIC, model=codex_model(effort='high', timeout=300), include_input=True
        ),
        'strong, default include_input=False': LLMJudge(rubric=RUBRIC, model=codex_model(effort='high', timeout=300)),
    }
    for label, judge in judges.items():
        if label in results.get('certificates', {}):
            continue
        before = len(TOKENS)
        cert = await certify_judge(
            judge, judge_cases, human_labels=labels, repeats=REPEATS, max_concurrency=4, batch_size=10
        )
        print(f'\n== {label}: {cert.verdict} ({cert.calls} of {cert.planned} calls)\n{cert.table()}', flush=True)
        for line in diagnose(cert, judge):
            print('   -', line)
        results.setdefault('certificates', {})[label] = {
            **cert.to_dict(), 'calls': cert.calls, 'planned': cert.planned, 'advice': diagnose(cert, judge),
            'tokens': sum(TOKENS[before:]),
        }  # fmt: skip
        save(results)

    pairs = [
        PairCase(c.name, c.inputs, f'Answer: {fmt(c, c.answer)}', f'Answer: {fmt(c, plausible_wrong(c))}') for c in cs
    ]
    for label, effort in (('weak', 'none'), ('strong', 'high')):
        key = f'pairwise {label}'
        if key in results:
            continue
        cert = await certify_pairwise(
            PairwiseJudge(
                rubric='Which reply answers the customer correctly according to the policy?',
                model=codex_model(effort=effort, timeout=300),
            ),
            pairs,
            max_concurrency=4,
        )
        rate = first_position_rate(cert)
        print(f'\n== pairwise {label}: {cert.verdict}\n{cert.table()}\nchose the first answer in {rate}', flush=True)
        results[key] = {**cert.to_dict(), 'first_position_rate': rate}
        save(results)
    print(f'\n{len(TOKENS)} calls, {sum(TOKENS):,} tokens')


if __name__ == '__main__':
    asyncio.run(main())
