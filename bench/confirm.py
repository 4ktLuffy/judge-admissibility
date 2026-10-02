"""Stage two: confirm the selected candidate on held-out questions, with the certified judge.

    PYTHONPATH=.:bench <venv>/bin/python bench/confirm.py

Selecting the best of k candidates and testing it on the same questions needs the significance
level split k ways, which at 40 questions misses real gains (see `results/optimize.json`).
Testing the one selected candidate on questions it was not selected on needs no split. This
grades the cached held-out replies of the baseline and the selected candidate with the reference
judge (ADMISSIBLE), decides with a single test, and compares the decision with ground truth.
"""

from __future__ import annotations

import asyncio
import json

from optimize import BASELINE_PROMPT, ROOT, Cache, answers, certificate_from, rate, split, truth, verdicts

from pydantic_evals_admissibility import decide


async def main() -> None:
    run = json.loads((ROOT / 'results' / 'optimize.json').read_text())
    selected = run['candidates'][
        max(range(len(run['candidates'])), key=lambda i: run['candidates'][i]['reference_train'])
    ]
    cache, limit = Cache(), asyncio.Semaphore(6)
    _, test = split()
    by_name = {q.name: q for q in test}
    ref_cert = certificate_from(ROOT / 'results' / 'judge_vs_truth.reference.json')
    out = {}
    for label, prompt in (('baseline', BASELINE_PROMPT), ('selected', selected['prompt'])):
        replies = await answers(prompt, test, cache, limit)  # cached from the optimizer run
        out[label] = {
            'judge': await verdicts(replies, by_name, cache, limit, reference=True),
            'truth': truth(replies, by_name),
        }
        print(f'{label}: reference judge {rate(out[label]["judge"]):.2f}, truth {rate(out[label]["truth"]):.2f}')
    on_judge = decide(out['baseline']['judge'], out['selected']['judge'], certificate=ref_cert)
    on_truth = decide(out['baseline']['truth'], out['selected']['truth'])
    print('confirmation on the certified judge:', on_judge.summary())
    print('the same decision on ground truth:  ', on_truth.summary())
    agree = sum(
        j == t
        for label in out
        for name in out[label]['judge']
        for j, t in zip(out[label]['judge'][name], out[label]['truth'][name], strict=True)
    )
    total = sum(len(v) for label in out for v in out[label]['judge'].values())
    print(f'reference judge agrees with ground truth on {agree}/{total} held-out replies')
    (ROOT / 'results' / 'confirm.json').write_text(json.dumps({
        'selected_index': selected['index'], 'on_judge': on_judge.summary(), 'on_truth': on_truth.summary(),
        'decision': on_judge.decision, 'judge_truth_agreement': [agree, total],
        'rates': {k: {'judge': rate(v['judge']), 'truth': rate(v['truth'])} for k, v in out.items()},
    }, indent=2))  # fmt: skip


if __name__ == '__main__':
    asyncio.run(main())
