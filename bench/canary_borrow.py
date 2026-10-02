"""How often is an answer borrowed from another question actually right? From cached replies only.

    PYTHONPATH=.:bench <venv>/bin/python bench/canary_borrow.py

Why `JudgeCanary(borrow=True)` is off by default. In `bench/self_improving_loop.py` the canary
watched the promoted prompt (optimize.json candidate 2) on the 40 held-out questions, first
reply per question. For those replies, and for each kind of question: over every ordered pair
of two different questions of that kind, is the donor's reply correct for the other question
(`task.is_correct`)? A borrowed answer that is correct is a control a sound judge should pass.

The same rate is shown for other replies from the cache, to show it is a property of the answer
space (seven weekdays, small letter counts) rather than of one prompt. No calls are made.
Writes `results/canary_borrow.json`.
"""

from __future__ import annotations

import json
from collections import defaultdict
from typing import Any

from optimize import BASELINE_PROMPT, CACHE_PATH, ROOT, Cache, split
from task import Question, is_correct, questions

OUT = ROOT / 'results' / 'canary_borrow.json'


def replies_from_cache(prompt: str, qs: list[Question], repeats: tuple[int, ...]) -> dict[str, list[str]]:
    cache = json.loads(CACHE_PATH.read_text())
    out: dict[str, list[str]] = {}
    for q in qs:
        for r in repeats:
            key = Cache.key('agent', prompt, q.name, q.question, str(r))
            if key in cache:
                out.setdefault(q.name, []).append(cache[key])
    return out


def borrowed_correct(qs: list[Question], replies: dict[str, list[str]]) -> dict[str, dict[str, Any]]:
    """Per kind: of all (question, another same-kind question's reply) pairs, how many are correct."""
    by_kind: dict[str, list[Question]] = defaultdict(list)
    for q in qs:
        by_kind[q.kind].append(q)
    out = {}
    for kind, group in by_kind.items():
        correct = total = same_answer = pairs = 0
        for q in group:
            for donor in group:
                if donor.name == q.name:
                    continue
                pairs += 1
                same_answer += donor.answer == q.answer
                for reply in replies.get(donor.name, []):
                    total += 1
                    correct += is_correct(q, reply)
        out[kind] = {
            'correct': correct,
            'borrowed': total,
            'rate': round(correct / total, 4) if total else None,
            'same_expected_answer': f'{same_answer}/{pairs}',
        }
    return out


def main() -> None:
    train, test = split()
    run = json.loads((ROOT / 'results' / 'optimize.json').read_text())
    promoted = run['candidates'][2]['prompt']
    baseline_rows = json.loads((ROOT / 'results' / 'task_baseline.json').read_text())['rows']
    original = questions()  # the 40 of bench/baseline.py
    baseline_replies: dict[str, list[str]] = defaultdict(list)
    for row in baseline_rows:
        baseline_replies[row['question']].append(row['reply'])

    variants = {
        'canary: promoted prompt, held-out questions, first reply (as the canary saw them)': borrowed_correct(
            test, replies_from_cache(promoted, test, (0,))
        ),
        'promoted prompt, held-out questions, both replies': borrowed_correct(
            test, replies_from_cache(promoted, test, (0, 1))
        ),
        'baseline prompt, held-out questions, both replies': borrowed_correct(
            test, replies_from_cache(BASELINE_PROMPT, test, (0, 1))
        ),
        'baseline prompt, training questions, both replies': borrowed_correct(
            train, replies_from_cache(BASELINE_PROMPT, train, (0, 1))
        ),
        'task_baseline.json (the original 40 questions), both replies': borrowed_correct(original, baseline_replies),
    }
    for label, kinds in variants.items():
        print(label)
        for kind, r in kinds.items():
            rate = f'{r["rate"]:.1%}' if r['rate'] is not None else '-'
            same = r['same_expected_answer']
            print(f'  {kind:10} {r["correct"]:3}/{r["borrowed"]:<3} = {rate:>6}   same expected answer {same}')
    OUT.write_text(
        json.dumps(
            {
                'definition': (
                    'per kind, over every ordered pair (question q, another question d of the same kind): '
                    "the share of d's cached replies that task.is_correct accepts as an answer to q"
                ),
                'variants': variants,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
