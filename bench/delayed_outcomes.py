"""Delayed outcomes: correct a lenient judge's pass rate with what happened later. Simulated, then real.

    PYTHONPATH=.:bench <venv>/bin/python bench/delayed_outcomes.py

1. A simulated support desk: 2000 tickets a day, the agent truly succeeds on 60%, the judge has
   sensitivity 0.95 and specificity 0.7, so it passes about 69%. Outcomes (reopened or not)
   arrive after a day: today's pass rate is corrected with yesterday's outcomes. 500 days, three
   ways outcomes arrive: for 30% of tickets at random; for 30% of passed tickets only (only a
   passed ticket can be reopened); for 45% of passed and 10% of failed. Each is corrected two
   ways: with the package's adjustment for who got an outcome, and naively, from observed cases
   as if they were all there were.
2. Real verdicts: `results/support_eval.json` holds each certified judge's verdicts on 80 real
   agent replies (two per question) with ground truth (`human:1` / `human:0`). Ground truth is
   treated as the delayed outcome. Calibrate on the first reply to each question, correct the
   judge's pass rate on the second, and compare with the second replies' truth.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

from pydantic_evals_admissibility._outcomes import outcome_calibration, recalibrated_pass_rate

ROOT = Path(__file__).parent.parent
OUT = ROOT / 'results' / 'delayed_outcomes.json'
TICKETS, DAYS, TRUTH, SENS, SPEC = 2000, 500, 0.6, 0.95, 0.7
SCENARIOS = {
    'uniform 30%': (0.3, 0.3),
    'passed only (30% of passed)': (0.3, 0.0),
    'verdict-dependent (45% passed, 10% failed)': (0.45, 0.10),
}
RESAMPLES = 1000


def day(rng: random.Random, observe: tuple[float, float]) -> tuple[dict[str, bool], dict[str, bool | None], float]:
    verdicts: dict[str, bool] = {}
    outcomes: dict[str, bool | None] = {}
    successes = 0
    for i in range(TICKETS):
        ok = rng.random() < TRUTH
        successes += ok
        passed = rng.random() < (SENS if ok else 1 - SPEC)
        verdicts[f't{i}'] = passed
        outcomes[f't{i}'] = ok if rng.random() < observe[0 if passed else 1] else None
    return verdicts, outcomes, successes / TICKETS


def summarise(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    done = [r for r in rows if r[key] is not None]
    errors = [r[key] - r['truth'] for r in done]
    covered = [r[f'{key}_interval'][0] <= r['truth'] <= r[f'{key}_interval'][1] for r in done if r[f'{key}_interval']]
    widths = [r[f'{key}_interval'][1] - r[f'{key}_interval'][0] for r in done if r[f'{key}_interval']]
    return {
        'days_estimated': len(done),
        'mean': sum(r[key] for r in done) / len(done) if done else None,
        'bias': sum(errors) / len(errors) if errors else None,
        'mean_abs_error': sum(map(abs, errors)) / len(errors) if errors else None,
        'coverage': sum(covered) / len(covered) if covered else None,
        'mean_width': sum(widths) / len(widths) if widths else None,
    }


def simulate() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, observe in SCENARIOS.items():
        rng = random.Random(0)
        started = time.perf_counter()
        yesterday = day(rng, observe)
        rows, flagged = [], 0
        for d in range(DAYS):
            today = day(rng, observe)
            verdicts, outcomes, _ = yesterday
            report = outcome_calibration(verdicts, outcomes, resamples=0)
            flagged += report.biased
            adjusted = recalibrated_pass_rate(today[0], report, resamples=RESAMPLES, seed=d)
            observed = {c: v for c, v in verdicts.items() if outcomes[c] is not None}
            naive_report = outcome_calibration(observed, {c: outcomes[c] for c in observed}, resamples=0)
            naive = recalibrated_pass_rate(today[0], naive_report, resamples=RESAMPLES, seed=d)
            rows.append({
                'truth': today[2],
                'raw': adjusted.raw.rate,
                'adjusted': adjusted.corrected,
                'adjusted_interval': adjusted.interval,
                'naive': naive.corrected,
                'naive_interval': naive.interval,
            })  # fmt: skip
            yesterday = today
        raw = [r['raw'] - r['truth'] for r in rows]
        out[name] = {
            'observe_rate': {'passed': observe[0], 'failed': observe[1]},
            'truth_mean': sum(r['truth'] for r in rows) / DAYS,
            'raw_mean': sum(r['raw'] for r in rows) / DAYS,
            'raw_bias': sum(raw) / DAYS,
            'flagged_biased_days': flagged,
            'adjusted': summarise(rows, 'adjusted'),
            'naive': summarise(rows, 'naive'),
            'seconds': round(time.perf_counter() - started, 1),
        }
    return out


def replay() -> dict[str, Any]:
    data = json.loads((ROOT / 'results' / 'support_eval.json').read_text())
    out: dict[str, Any] = {}
    for name, cert in data['certificates'].items():
        seen: dict[str, int] = {}
        reply: dict[str, tuple[int, bool | None, bool]] = {}  # id -> (reply index, verdict, truth)
        for j in cert['judgments']:
            if j['role'].startswith('human:'):
                k = seen.get(j['case'], 0)
                seen[j['case']] = k + 1
                reply[f'{j["case"]}#{k}'] = (k, j['passed'], j['role'] == 'human:1')
        errored = sum(v is None for _, v, _ in reply.values())
        judged = {i: r for i, r in reply.items() if r[1] is not None}
        everything = outcome_calibration(
            {i: bool(v) for i, (_, v, _) in judged.items()},
            {i: t for i, (_, _, t) in judged.items()},
            slices={i: i.split('-')[0] for i in judged},
        )
        first = {i: r for i, r in judged.items() if r[0] == 0}
        second = {i: r for i, r in judged.items() if r[0] == 1}
        calibration = outcome_calibration(
            {i: bool(v) for i, (_, v, _) in first.items()}, {i: t for i, (_, _, t) in first.items()}
        )
        estimate = recalibrated_pass_rate({i: bool(v) for i, (_, v, _) in second.items()}, calibration)
        truth = sum(t for _, _, t in second.values()) / len(second) if second else None
        out[name] = {
            'replies': len(reply),
            'errored': errored,
            'all_replies': everything.to_dict(),
            'split': {
                'calibrated_on_first_replies': calibration.to_dict(),
                'second_replies': len(second),
                'estimate': estimate.to_dict(),
                'truth': truth,
            },
        }
        print(f'== {name}\n{everything.table()}')
        print(
            f'  split: calibrate on {calibration.cases} first replies, apply to {len(second)} second replies: '
            f'judge passes {estimate.raw.rate:.3f}, corrected {_f(estimate.corrected)} {_iv(estimate.interval)}, '
            f'truth {_f(truth)}' + ''.join(f'\n  ! {w}' for w in estimate.warnings) + '\n'
        )
    return out


def _f(x: float | None) -> str:
    return '-' if x is None else f'{x:.3f}'


def _iv(iv: tuple[float, float] | None) -> str:
    return '-' if iv is None else f'[{iv[0]:.3f}, {iv[1]:.3f}]'


def main() -> None:
    sim = simulate()
    print(f'Simulated desk: {TICKETS} tickets/day, {DAYS} days, truth {TRUTH}, sensitivity {SENS}, specificity {SPEC}')
    print(
        f'{"outcomes arrive for":<44} {"truth":>6} {"raw":>6} {"method":<9} {"est":>6} {"bias":>7} '
        f'{"cover":>6} {"width":>6} days'
    )
    for name, s in sim.items():
        for method in ('adjusted', 'naive'):
            m = s[method]
            bias = '-' if m['bias'] is None else f'{m["bias"]:+.3f}'
            print(
                f'{name if method == "adjusted" else "":<44} {s["truth_mean"]:>6.3f} {s["raw_mean"]:>6.3f} {method:<9} '
                f'{_f(m["mean"]):>6} {bias:>7} '
                f'{_f(m["coverage"]):>6} {_f(m["mean_width"]):>6} {m["days_estimated"]}'
            )
        print(f'{"":<44} flagged as biased on {s["flagged_biased_days"]} of {DAYS} days ({s["seconds"]}s)')
    print()
    real = replay()
    OUT.write_text(
        json.dumps(
            {
                'simulation': {
                    'tickets_per_day': TICKETS,
                    'days': DAYS,
                    'truth': TRUTH,
                    'sensitivity': SENS,
                    'specificity': SPEC,
                    'resamples': RESAMPLES,
                    'scenarios': sim,
                },
                'support_eval_replay': real,
            },
            indent=2,
        )
    )


if __name__ == '__main__':
    main()
