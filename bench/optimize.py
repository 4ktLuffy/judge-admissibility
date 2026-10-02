"""One round of prompt optimization, decided three ways, then checked on questions it never saw.

    PYTHONPATH=.:bench <venv>/bin/python bench/optimize.py

1. 80 questions, split by position within each kind into 40 train and 40 held out.
2. The baseline prompt runs on train (2 repeats); the judge grades every reply.
3. Codex proposes 5 prompts from the baseline and the failures the judge flagged: what a
   self-improving loop sees.
4. Each candidate runs on train (2 repeats) and is graded by the judge and by ground truth.
5. Three choices, none of which sees ground truth:
   - naive: the candidate with the best plain-judge score, if it beats the baseline's;
   - gated on the plain judge: `decide(...)` with that judge's certificate (INADMISSIBLE: refused);
   - gated on the reference judge (sees the expected answer): `decide(...)` with its certificate,
     the significance level split across the candidates.
6. Baseline and each chosen prompt run on the 40 held-out questions. Ground truth is used only
   here, as the referee.

Every agent reply and judge verdict is cached in `results/optimize-cache.json`, so a
rerun after a crash does not pay for a call twice. Writes `results/optimize.json`.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any

from codex_judge import TOKENS, codex_model, codex_text_model, run_codex
from pydantic_ai import Agent
from pydantic_evals.evaluators import LLMJudge
from task import Question, is_correct, questions

from pydantic_evals_admissibility import Certificate, Check, GateRules, decide
from pydantic_evals_admissibility._cases import JudgeCase
from pydantic_evals_admissibility._certify import _context, assertion_of

ROOT = Path(__file__).parent.parent
CACHE_PATH = ROOT / 'results' / 'optimize-cache.json'  # published, so the run is reproducible without new calls
BASELINE_PROMPT = 'Answer the question. End your reply with a line "Answer: <answer>".'
RUBRIC = 'The output correctly answers the question.'
REPEATS = 2
CANDIDATES = 5
CONCURRENCY = 6


class Cache:
    def __init__(self) -> None:
        self.data: dict[str, Any] = json.loads(CACHE_PATH.read_text()) if CACHE_PATH.exists() else {}
        self.lock = asyncio.Lock()

    @staticmethod
    def key(*parts: str) -> str:
        return hashlib.sha256('\x1f'.join(parts).encode()).hexdigest()

    async def get_or(self, key: str, make: Any) -> Any:
        if key in self.data:
            return self.data[key]
        value = await make()
        async with self.lock:
            self.data[key] = value
            CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
            tmp = CACHE_PATH.with_suffix('.tmp')
            tmp.write_text(json.dumps(self.data))
            tmp.replace(CACHE_PATH)
        return value


def split() -> tuple[list[Question], list[Question]]:
    by_kind: dict[str, list[Question]] = {}
    for q in questions(n_per_kind=20, seed=7, unique=True):
        by_kind.setdefault(q.kind, []).append(q)
    train = [q for qs in by_kind.values() for q in qs[0::2]]
    test = [q for qs in by_kind.values() for q in qs[1::2]]
    return train, test


async def answers(prompt: str, qs: list[Question], cache: Cache, limit: asyncio.Semaphore) -> dict[str, list[str]]:
    agent = Agent(codex_text_model(), instructions=prompt)

    async def one(q: Question, r: int) -> str:
        async def call() -> str:
            async with limit:
                try:
                    return (await agent.run(q.question)).output
                except Exception as exc:  # recorded as an empty reply, which is never correct
                    return f'[error] {type(exc).__name__}: {exc}'[:300]

        return await cache.get_or(Cache.key('agent', prompt, q.name, q.question, str(r)), call)

    replies = await asyncio.gather(*(one(q, r) for q in qs for r in range(REPEATS)))
    out: dict[str, list[str]] = {}
    for i, q in enumerate(q for q in qs for _ in range(REPEATS)):
        out.setdefault(q.name, []).append(replies[i])
    return out


async def verdicts(
    replies: dict[str, list[str]],
    by_name: dict[str, Question],
    cache: Cache,
    limit: asyncio.Semaphore,
    *,
    reference: bool = False,
) -> dict[str, list[bool]]:
    """The plain judge (question only) or, with `reference`, the judge that also sees the expected answer."""
    judge = LLMJudge(rubric=RUBRIC, model=codex_model(), include_input=True, include_expected_output=reference)

    async def one(name: str, reply: str) -> bool:
        async def call() -> bool | None:
            async with limit:
                try:
                    q = by_name[name]
                    case = JudgeCase(name, q.question, '', expected_output=q.answer if reference else None)
                    raw = await judge.evaluate_async(_context(case, reply))
                    return assertion_of(raw)[0]
                except Exception:
                    return None

        kind = 'judge-reference' if reference else 'judge'
        verdict = await cache.get_or(Cache.key(kind, RUBRIC, by_name[name].question, reply), call)
        return bool(verdict)  # no verdict counts as a fail, as an evaluator failure would

    pairs = [(n, r) for n, rs in replies.items() for r in rs]
    results = await asyncio.gather(*(one(n, r) for n, r in pairs))
    out: dict[str, list[bool]] = {}
    for (n, _), v in zip(pairs, results, strict=True):
        out.setdefault(n, []).append(v)
    return out


def truth(replies: dict[str, list[str]], by_name: dict[str, Question]) -> dict[str, list[bool]]:
    return {n: [is_correct(by_name[n], r) for r in rs] for n, rs in replies.items()}


def rate(scores: dict[str, list[bool]]) -> float:
    flat = [v for vs in scores.values() for v in vs]
    return sum(flat) / len(flat)


async def propose(
    baseline_replies: dict[str, list[str]], judged: dict[str, list[bool]], by_name: dict[str, Question], cache: Cache
) -> list[str]:
    failures = [(by_name[n].question, rs[0]) for n, rs in baseline_replies.items() if not judged[n][0]][:8]
    examples = '\n\n'.join(f'Question: {q}\nReply:\n{r.strip()[-500:]}' for q, r in failures)
    instructions = 'You improve system prompts for an AI assistant. Reply with JSON only: {"prompts": [...]}.'
    prompt = (
        f'The assistant answers short questions (multi-step arithmetic, counting letters, day-of-week arithmetic, '
        f'and string transformations) and must end with a line "Answer: <answer>".\n\n'
        f'Current system prompt:\n{BASELINE_PROMPT}\n\n'
        f'Replies a grader marked as failing:\n\n{examples}\n\n'
        f'Write {CANDIDATES} different improved system prompts that would make the grader pass more replies. '
        f'Each must keep the "Answer: <answer>" final line requirement.'
    )
    schema = ROOT / 'results' / 'scratch' / 'prompts-schema.json'
    schema.write_text(json.dumps({
        'type': 'object', 'additionalProperties': False, 'required': ['prompts'],
        'properties': {'prompts': {'type': 'array', 'items': {'type': 'string'}}},
    }))  # fmt: skip

    async def call() -> list[str]:
        reply = await run_codex(instructions, prompt, model='gpt-5.6-luna', effort='none', schema=schema, timeout=180)
        return json.loads(reply)['prompts'][:CANDIDATES]

    return await cache.get_or(Cache.key('propose', prompt), call)


def certificate_from(path: Path) -> Certificate:
    data = json.loads(path.read_text())
    checks = tuple(
        Check(
            c['name'],
            c['status'],
            c['successes'],
            c['trials'],
            tuple(c['interval']),
            c['threshold'],
            c['detail'],
            c['errors'],
        )
        for c in data['checks']
    )
    return Certificate(data['verdict'], checks, (), judge=data['judge'])


async def main() -> None:
    cache = Cache()
    limit = asyncio.Semaphore(CONCURRENCY)
    train, test = split()
    by_name = {q.name: q for q in train + test}
    started = time.monotonic()

    def log(msg: str) -> None:
        print(f'[{time.monotonic() - started:6.0f}s, {len(TOKENS)} calls] {msg}', flush=True)

    base_train = await answers(BASELINE_PROMPT, train, cache, limit)
    base_judge = await verdicts(base_train, by_name, cache, limit)
    base_ref = await verdicts(base_train, by_name, cache, limit, reference=True)
    base_truth = truth(base_train, by_name)
    log(
        f'baseline on train: plain judge {rate(base_judge):.2f}, reference judge {rate(base_ref):.2f}, '
        f'truth {rate(base_truth):.2f}'
    )

    candidates = await propose(base_train, base_judge, by_name, cache)
    log(f'{len(candidates)} candidates proposed')

    rows: list[dict[str, Any]] = []
    for i, prompt in enumerate(candidates):
        replies = await answers(prompt, train, cache, limit)
        judged = await verdicts(replies, by_name, cache, limit)
        ref = await verdicts(replies, by_name, cache, limit, reference=True)
        true = truth(replies, by_name)
        rows.append({'index': i, 'prompt': prompt, 'judge_train': rate(judged), 'reference_train': rate(ref),
                     'truth_train': rate(true), 'judged': judged, 'ref': ref, 'true': true})  # fmt: skip
        log(f'candidate {i}: plain judge {rate(judged):.2f}, reference judge {rate(ref):.2f}, truth {rate(true):.2f}')

    judge_cert = certificate_from(ROOT / 'results' / 'judge_vs_truth.json')
    ref_cert = certificate_from(ROOT / 'results' / 'judge_vs_truth.reference.json')
    log(f'certificates: plain judge {judge_cert.verdict}, reference judge {ref_cert.verdict}')
    rules = GateRules().for_candidates(len(rows))
    best = max(rows, key=lambda r: r['judge_train'])
    naive = best if best['judge_train'] > rate(base_judge) else None
    gated_judge = [decide(base_judge, r['judged'], certificate=judge_cert, rules=rules) for r in rows]
    gated_ref = [decide(base_ref, r['ref'], certificate=ref_cert, rules=rules) for r in rows]
    promoted = [r for r, g in zip(rows, gated_ref, strict=True) if g.decision == 'PROMOTE']
    gated_pick = max(promoted, key=lambda r: r['reference_train']) if promoted else None
    for r, gj, gr in zip(rows, gated_judge, gated_ref, strict=True):
        log(f'candidate {r["index"]}: gate on plain judge -> {gj.decision}; gate on reference judge -> {gr.summary()}')

    held_out: dict[str, Any] = {}
    to_test = {'baseline': BASELINE_PROMPT}
    if naive is not None:
        to_test['naive pick'] = naive['prompt']
    if gated_pick is not None:
        to_test['gated pick'] = gated_pick['prompt']
    for label, prompt in to_test.items():
        replies = await answers(prompt, test, cache, limit)
        held_out[label] = {'truth': truth(replies, by_name), 'prompt': prompt}
        log(f'held-out truth, {label}: {rate(held_out[label]["truth"]):.2f}')
    held_out_decisions = {
        label: decide(held_out['baseline']['truth'], v['truth']).summary()
        for label, v in held_out.items()
        if label != 'baseline'
    }
    for label, summary in held_out_decisions.items():
        log(f'held-out, {label} vs baseline: {summary}')

    (ROOT / 'results' / 'optimize.json').write_text(json.dumps({
        'baseline_prompt': BASELINE_PROMPT, 'rubric': RUBRIC, 'judge_certificate': judge_cert.verdict,
        'reference_certificate': ref_cert.verdict,
        'baseline_train': {
            'plain_judge': rate(base_judge), 'reference_judge': rate(base_ref), 'truth': rate(base_truth)
        },
        'candidates': [{k: v for k, v in r.items() if k not in ('judged', 'ref', 'true')} for r in rows],
        'naive_pick': None if naive is None else naive['index'],
        'gate_on_judge': [g.decision for g in gated_judge],
        'gate_on_reference': [g.summary() for g in gated_ref],
        'gated_pick': None if gated_pick is None else gated_pick['index'],
        'held_out': {k: {'truth': rate(v['truth']), 'prompt': v['prompt']} for k, v in held_out.items()},
        'held_out_vs_baseline': held_out_decisions,
        'calls': len(TOKENS), 'tokens': sum(TOKENS),
    }, indent=2))  # fmt: skip


if __name__ == '__main__':
    asyncio.run(main())
