"""Prediction-powered inference for an agent's pass rate: simulated, then replayed on real verdicts.

    PYTHONPATH=.:bench <venv>/bin/python bench/ppi.py

No model is called.

1. Simulation: populations of N = 5000 outputs, each truly passing with probability 0.6, judged by
   a judge of a given sensitivity and specificity. People label a uniform random audit of n in
   {50, 100, 200}, 2000 populations per setting. Compared on the pass rate of each population:
   plain PPI and PPI++ (`interval`: CLT, exact where the CLT one has zero width), PPI's exact
   interval, people alone (Wilson, and exact Clopper-Pearson), the judge alone (Wilson on all N),
   and Rogan-Gladen calibrated on the audit (`recalibrated_pass_rate`, 200 bootstrap draws).
   Reported: bias, coverage, mean width.
2. Replay: `results/support_eval.json` holds each certified Codex judge's verdicts on real agent
   replies with ground truth (roles `human:1` / `human:0`): 80 replies, or 20 for the
   `include_input=False` judge, whose certification stopped early. Ground truth plays the audit:
   2000 random audits of 20 and 40 replies (5 and 10 of the 20). Same comparisons, plus people
   alone with the finite population correction, so that PPI's correction does not flatter it.
"""

from __future__ import annotations

import json
import random
import time
from pathlib import Path
from statistics import NormalDist
from typing import Any

from pydantic_evals_admissibility._outcomes import outcome_calibration, recalibrated_pass_rate
from pydantic_evals_admissibility._ppi import _estimate, audit_sample, ppi_pass_rate
from pydantic_evals_admissibility._stats import clopper_pearson, wilson

ROOT = Path(__file__).parent.parent
SOURCE = ROOT / 'results' / 'support_eval.json'
OUT = ROOT / 'results' / 'ppi.json'
N, TRUTH, TRIALS, ALPHA = 5000, 0.6, 2000, 0.05
AUDITS = (50, 100, 200)
JUDGES = {
    'near-perfect (sens 0.98, spec 0.98)': (0.98, 0.98),
    'good (0.95, 0.90)': (0.95, 0.90),
    'lenient (0.95, 0.70)': (0.95, 0.70),
    'mediocre (0.80, 0.80)': (0.80, 0.80),
    'useless (0.70, 0.30): verdict independent of truth': (0.70, 0.30),
}
RG_RESAMPLES = 200
Z = NormalDist().inv_cdf(1 - ALPHA / 2)
NAN = float('nan')
Row = dict[str, tuple[float, tuple[float, float]] | None]


def methods(judge_passes: int, total: int, pairs: list[tuple[bool, bool]], seed: int, rg: bool = True) -> Row:
    """Every estimator on one audit: (estimate, interval), or None where it is undefined."""
    ppi = _estimate(judge_passes, total, pairs, ALPHA, tuned=False)
    tuned = _estimate(judge_passes, total, pairs, ALPHA, tuned=True)
    n, k = len(pairs), sum(h for h, _ in pairs)
    mean = k / n
    se = (max(0.0, 1 - n / total) * mean * (1 - mean) / (n - 1)) ** 0.5  # sample variance, n - 1
    row: Row = {
        'ppi': (ppi.estimate, ppi.interval),
        'ppi++': (tuned.estimate, tuned.interval),
        'ppi exact': (ppi.estimate, ppi.exact_interval),
        'human only (Wilson)': (mean, wilson(k, n, Z)),
        'human only (Clopper-Pearson)': (mean, clopper_pearson(k, n, ALPHA)),
        'human only (CLT, finite population)': (mean, (max(0.0, mean - Z * se), min(1.0, mean + Z * se))),
        'judge only (Wilson on N)': (judge_passes / total, wilson(judge_passes, total, Z)),
    }
    if rg:
        audit_verdicts = {f'a{i}': j for i, (_, j) in enumerate(pairs)}
        audit_truth = {f'a{i}': h for i, (h, _) in enumerate(pairs)}
        calibration = outcome_calibration(audit_verdicts, audit_truth, resamples=0)
        traffic = [True] * judge_passes + [False] * (total - judge_passes)
        corrected = recalibrated_pass_rate(traffic, calibration, alpha=ALPHA, resamples=RG_RESAMPLES, seed=seed)
        row['Rogan-Gladen (audit as calibration)'] = (
            (corrected.corrected, corrected.interval)
            if corrected.corrected is not None and corrected.interval is not None
            else None
        )
    return row


class Tally:
    def __init__(self) -> None:
        self.rows: dict[str, list[tuple[float, float, float, float]]] = {}
        self.undefined: dict[str, int] = {}
        self.trials = 0

    def add(self, row: Row, truth: float) -> None:
        self.trials += 1
        for name, value in row.items():
            if value is None:
                self.undefined[name] = self.undefined.get(name, 0) + 1
                continue
            estimate, (low, high) = value
            self.rows.setdefault(name, []).append((estimate, low, high, truth))

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for name, rows in self.rows.items():
            k = len(rows)
            errors = [e - t for e, _, _, t in rows]
            out[name] = {
                'defined': k,
                'bias': round(sum(errors) / k, 4),
                'rmse': round((sum(e * e for e in errors) / k) ** 0.5, 4),
                'coverage': round(sum(lo <= t <= hi for _, lo, hi, t in rows) / k, 4),
                'mean_width': round(sum(hi - lo for _, lo, hi, _ in rows) / k, 4),
                'zero_width': round(sum(hi == lo for _, lo, hi, _ in rows) / k, 4),
            }
        for name, count in self.undefined.items():
            out.setdefault(name, {})['undefined'] = count
        return out


def population(rng: random.Random, sens: float, spec: float) -> tuple[list[int], float]:
    """Cell counts (human pass & judge pass, human pass & judge fail, human fail & judge pass, both fail)."""
    passing = rng.binomialvariate(N, TRUTH)
    tp = rng.binomialvariate(passing, sens)
    fp = rng.binomialvariate(N - passing, 1 - spec)
    return [tp, passing - tp, fp, N - passing - fp], passing / N


def draw(rng: random.Random, cells: list[int], n: int) -> list[tuple[bool, bool]]:
    """A uniform random audit without replacement, as (human, judge) pairs."""
    bounds = [sum(cells[: i + 1]) for i in range(4)]
    kinds = ((True, True), (True, False), (False, True), (False, False))
    return [kinds[next(c for c, b in enumerate(bounds) if i < b)] for i in rng.sample(range(N), n)]


def simulate() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for label, (sens, spec) in JUDGES.items():
        for n in AUDITS:
            rng, tally, started = random.Random(n), Tally(), time.perf_counter()
            for trial in range(TRIALS):
                cells, truth = population(rng, sens, spec)
                tally.add(methods(cells[0] + cells[2], N, draw(rng, cells, n), seed=trial), truth)
            out.setdefault(label, {})[f'n={n}'] = summary = tally.summary()
            print(
                f'{label:<52} n={n:<4} '
                + '  '.join(
                    f'{name.split(" (")[0]} {s.get("coverage", 0):.3f}/{s.get("mean_width", 0):.3f}'
                    for name, s in summary.items()
                    if name
                    in ('ppi', 'ppi++', 'ppi exact', 'human only (Wilson)', 'Rogan-Gladen (audit as calibration)')
                )
                + f'  ({time.perf_counter() - started:.0f}s)',
                flush=True,
            )
    return out


def replay() -> dict[str, Any]:
    data = json.loads(SOURCE.read_text())
    out: dict[str, Any] = {}
    for label, certificate in data['certificates'].items():
        verdicts: dict[str, bool] = {}
        labels: dict[str, bool] = {}
        seen: dict[str, int] = {}
        for j in certificate['judgments']:
            if not j['role'].startswith('human:') or j['error'] is not None:
                continue
            i = seen[j['case']] = seen.get(j['case'], -1) + 1
            name = f'{j["case"]}#{i}'
            verdicts[name], labels[name] = bool(j['passed']), j['role'] == 'human:1'
        total = len(verdicts)
        truth = sum(labels.values()) / total
        result: dict[str, Any] = {
            'replies': total,
            'true_pass_rate': truth,
            'judge_pass_rate': sum(verdicts.values()) / total,
            'false_fails': sum(labels[c] and not verdicts[c] for c in verdicts),
            'false_passes': sum(verdicts[c] and not labels[c] for c in verdicts),
        }
        for n in (20, 40) if total >= 80 else (total // 4, total // 2):
            tally, ratios, lams = Tally(), [], []
            for seed in range(TRIALS):
                sample = audit_sample(verdicts, n, seed)
                public = ppi_pass_rate(verdicts, {c: labels[c] for c in sample.cases}, sample=sample)
                if public.width_ratio is not None:
                    ratios.append(public.width_ratio)
                lams.append(public.lam)
                pairs = [(labels[c], verdicts[c]) for c in sample.cases]
                tally.add(methods(sum(verdicts.values()), total, pairs, seed), truth)
            result[f'audit {n}'] = {
                **tally.summary(),
                'ppi++ width / Wilson width (mean)': round(sum(ratios) / len(ratios), 4) if ratios else None,
                'ppi++ lam (mean)': round(sum(lams) / len(lams), 4),
            }
        out[label] = result
        print(f'\n== {label}: {total} replies, true {truth:.3f}, judge {result["judge_pass_rate"]:.3f}')
        for key, summary in result.items():
            if key.startswith('audit'):
                for name, s in summary.items():
                    if isinstance(s, dict):
                        print(
                            f'  {key:<9} {name:<38} bias {s.get("bias", NAN):+.3f} '
                            f'coverage {s.get("coverage", NAN):.3f} width {s.get("mean_width", NAN):.3f}'
                            + (f'  undefined {s["undefined"]}' if 'undefined' in s else '')
                        )
    return out


def main() -> None:
    results: dict[str, Any] = {
        'setup': {
            'population': N,
            'true_pass_rate': TRUTH,
            'trials': TRIALS,
            'alpha': ALPHA,
            'audits': AUDITS,
            'rogan_gladen_resamples': RG_RESAMPLES,
            'truth_for_coverage': 'the realized pass rate of each finite population',
        }
    }
    results['replay'] = replay()
    results['simulation'] = simulate()
    OUT.write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
