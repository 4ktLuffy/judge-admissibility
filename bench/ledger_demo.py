"""Which questions did the optimizer spend, and why confirming on spent questions promotes noise.

    PYTHONPATH=.:bench <venv>/bin/python bench/ledger_demo.py [--trials 1000]

No model is called. Two parts:

1. Replay `bench/optimize.py` into a `FeedbackLedger`. The proposer was shown up to 8 baseline training
   replies the plain judge failed (`propose`: the first reply of each question, in order); then all 5
   candidates were scored on all 40 training questions and the best was selected on those scores
   (`confirm.py` picks by `reference_train`). Both are feedback, so both are recorded. The ledger then
   checks `bench/confirm.py`'s confirmation set (the 40 held-out questions) and, for contrast, what a
   confirmation on the training questions would have got.
2. A simulation of why it matters: baseline and K candidates are all the same (no real gain), an
   optimizer picks the candidate that scored best on a dev set, and the pick is "confirmed" with a single
   `decide` on that same dev set, on the dev set with the level split K ways, or on fresh cases.

Writes `results/ledger_demo.json`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

from pydantic_evals_admissibility import Decision, FeedbackLedger, GateRules, decide

ROOT = Path(__file__).parent.parent


def draw(rates: list[float], repeats: int, rng: random.Random) -> dict[str, list[bool]]:
    return {f'c{i}': [rng.random() < p for _ in range(repeats)] for i, p in enumerate(rates)}


def peeking_trial(
    k: int, rng: random.Random, *, cases: int = 40, repeats: int = 2, resamples: int = 2000
) -> dict[str, Decision]:
    """One optimizer run with no real gain: pick the best of `k` identical candidates on a dev set, then confirm."""
    rules = GateRules(resamples=resamples, seed=rng.randrange(2**31))
    dev = [rng.uniform(0.1, 0.9) for _ in range(cases)]
    baseline = draw(dev, repeats, rng)
    scored = [draw(dev, repeats, rng) for _ in range(k)]
    best = max(scored, key=lambda s: sum(map(sum, s.values())))
    fresh = [rng.uniform(0.1, 0.9) for _ in range(cases)]
    return {
        'exposed': decide(baseline, best, rules=rules).decision,
        'exposed, level split k ways': decide(baseline, best, rules=rules.for_candidates(k)).decision,
        'fresh': decide(draw(fresh, repeats, rng), draw(fresh, repeats, rng), rules=rules).decision,
    }


def simulate(k: int, trials: int, *, seed: int = 0, **kwargs: Any) -> dict[str, float]:
    """The share of trials each way of confirming promoted a candidate that is no better."""
    rng = random.Random(seed)
    counts: dict[str, int] = {}
    for _ in range(trials):
        for way, decision in peeking_trial(k, rng, **kwargs).items():
            counts[way] = counts.get(way, 0) + (decision == 'PROMOTE')
    return {way: n / trials for way, n in counts.items()}


def replay() -> dict[str, Any]:
    from judge_bridge import judged, replies
    from optimize import BASELINE_PROMPT, CANDIDATES, split

    run = json.loads((ROOT / 'results' / 'optimize.json').read_text())
    train, test = split()
    by_name = {q.name: q for q in train + test}
    base = replies(BASELINE_PROMPT, train)
    plain = judged('judge', {'baseline': base}, by_name)['baseline']
    shown = [n for n in base if not plain[n][0]][:8]  # exactly as `optimize.propose` picks its examples

    ledger = FeedbackLedger()
    ledger.record(shown, by='proposer: failing baseline replies shown to codex')
    ledger.record([q.name for q in train], by=f'selection: best of {CANDIDATES} candidates on train scores')

    selected = json.loads((ROOT / 'results' / 'confirm.json').read_text())['selected_index']
    held_out = [q.name for q in test]
    texts = {'baseline': base, 'candidate': replies(run['candidates'][selected]['prompt'], train)}
    reference = judged('judge-reference', texts, by_name)
    on_train = ledger.confirm(reference['baseline'], reference['candidate'])
    print(f'proposer saw {len(shown)} questions: {", ".join(shown)}')
    print(f'held-out confirmation set: {len(ledger.fresh(held_out))} of {len(held_out)} fresh')
    ledger.check_confirmation(held_out)  # raises if confirm.py had used a spent question
    print(f'confirming on train instead: {on_train.summary()}')
    return {
        'proposer_saw': shown,
        'train_exposed': len(ledger.exposed(q.name for q in train)),
        'held_out_fresh': len(ledger.fresh(held_out)),
        'held_out_total': len(held_out),
        'overlap_train_held_out': len({q.name for q in train} & set(held_out)),
        'confirmation_on_train': on_train.summary(),
        'ledger': ledger.to_dict(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=1000)
    args = parser.parse_args()
    out: dict[str, Any] = {'replay': replay(), 'simulation': {}}
    budget = GateRules().level / 2
    print(f'\nno real gain, 40 cases x 2 repeats, {args.trials} trials; false-promotion budget {budget:.3f}')
    for k in (5, 20):
        rates = simulate(k, args.trials, seed=k)
        out['simulation'][f'k={k}'] = rates
        print(f'k={k:>2}: ' + ', '.join(f'{way} {rate:.3f}' for way, rate in rates.items()))
    out['simulation']['setup'] = {'cases': 40, 'repeats': 2, 'trials': args.trials, 'budget': budget, 'resamples': 2000}
    (ROOT / 'results' / 'ledger_demo.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
