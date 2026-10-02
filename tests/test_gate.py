"""The gate's own controls: it must not promote noise, and it must tell better from worse."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

from pydantic_evals_admissibility import Certificate, Check, GateRules, decide

CASES = 40
FAST = GateRules(resamples=2000)


def simulate(difficulty: list[float], shift: float, seed: int, repeats: int = 2) -> dict[str, list[bool]]:
    rng = random.Random(seed)
    return {
        f'c{i}': [rng.random() < min(1.0, max(0.0, p + shift)) for _ in range(repeats)]
        for i, p in enumerate(difficulty)
    }


def difficulties(seed: int = 0) -> list[float]:
    rng = random.Random(seed)
    return [rng.uniform(0.1, 0.9) for _ in range(CASES)]


def decisions(shift: float, trials: int) -> list[str]:
    d = difficulties()
    return [decide(simulate(d, 0.0, 2 * t), simulate(d, shift, 2 * t + 1), rules=FAST).decision for t in range(trials)]


def test_identical_versions_are_almost_never_promoted() -> None:
    """A/A: the candidate is the baseline. A 95% interval allows about 2.5% false promotions."""
    out = decisions(0.0, trials=200)
    assert out.count('PROMOTE') / len(out) <= 0.05, out.count('PROMOTE')
    assert out.count('REJECT') / len(out) <= 0.05, out.count('REJECT')


def test_a_clearly_better_version_is_promoted() -> None:
    out = decisions(+0.3, trials=50)
    assert out.count('PROMOTE') / len(out) >= 0.9


def test_a_better_version_is_never_rejected() -> None:
    """Missing a small gain is acceptable (INCONCLUSIVE); calling a better version worse is not."""
    for shift in (0.1, 0.3):
        assert decisions(shift, trials=50).count('REJECT') <= 1


def test_a_clearly_worse_version_is_rejected() -> None:
    out = decisions(-0.3, trials=50)
    assert out.count('REJECT') / len(out) >= 0.9


def test_better_on_average_but_breaking_many_cases_is_rejected() -> None:
    # Cases 0-29 fail on the baseline and pass on the candidate; cases 30-35 break; 36-39 unchanged.
    baseline = {f'c{i}': [i >= 30] for i in range(CASES)}
    candidate = {f'c{i}': [i < 30 or i >= 36] for i in range(CASES)}
    result = decide(baseline, candidate, rules=GateRules(resamples=2000, max_regressions=0.1))
    assert result.p_better is not None and result.p_better < 0.025, result.summary()  # clearly better on average
    assert result.decision == 'REJECT' and result.regressed == 6, result.summary()  # but 6 > 10% of 40


def test_scores_from_an_uncertified_judge_are_refused() -> None:
    check = Check('acceptance', 'FAIL', 0, 40, (0.0, 0.09), 0.7)
    bad = Certificate('INADMISSIBLE', (check,), ())
    result = decide({'c': [True]}, {'c': [True]}, certificate=bad)
    assert result.decision == 'REFUSED' and 'acceptance' in result.reason


def test_mismatched_cases_are_an_error() -> None:
    with pytest.raises(ValueError):
        decide({'a': [True]}, {'b': [True]})


def test_unequal_repeats_per_case_are_an_error() -> None:
    """1 baseline run against 100 candidate runs makes the gain asymmetric under no difference.

    Simulated with equal true quality, the sign-flip test then acted on most comparisons.
    """
    rng = random.Random(0)
    d = difficulties()
    baseline = {f'c{i}': [rng.random() < p] for i, p in enumerate(d)}
    candidate = {f'c{i}': [rng.random() < p for _ in range(100)] for i, p in enumerate(d)}
    with pytest.raises(ValueError, match='same number of outcomes'):
        decide(baseline, candidate, rules=FAST)


def test_no_cases_are_an_error() -> None:
    with pytest.raises(ValueError, match='no cases'):
        decide({}, {})


def test_real_aa_the_two_baseline_runs_are_not_told_apart() -> None:
    """The same prompt, run twice on Codex: 11 of 40 questions changed. The gate must not act on it."""
    path = Path(__file__).parent.parent / 'results' / 'task_baseline.json'
    if not path.exists():
        pytest.skip('run bench/baseline.py first')
    rows = json.loads(path.read_text())['rows']
    runs = [{r['question']: [r['correct']] for r in rows if r['repeat'] == k} for k in (0, 1)]
    result = decide(runs[0], runs[1])
    assert result.decision == 'INCONCLUSIVE', result.summary()


def test_choosing_among_five_identical_candidates_rarely_promotes_any() -> None:
    """Five candidates, none better: the chance of promoting at least one must stay within 5%."""
    d = difficulties()
    plain = corrected = 0
    for t in range(200):
        base = simulate(d, 0.0, 1000 + 6 * t)
        cands = [simulate(d, 0.0, 1001 + 6 * t + j) for j in range(5)]
        plain += any(decide(base, c, rules=FAST).decision == 'PROMOTE' for c in cands)
        corrected += any(decide(base, c, rules=FAST.for_candidates(5)).decision == 'PROMOTE' for c in cands)
    assert corrected / 200 <= 0.05, (plain, corrected)
    assert plain > corrected  # the correction is doing something


def test_more_cases_detect_smaller_gains() -> None:
    from pydantic_evals_admissibility import detectable_gain

    small = {f'c{i}': [i % 2 == 0, i % 3 == 0] for i in range(20)}
    large = {f'c{i}': [i % 2 == 0, i % 3 == 0] for i in range(200)}
    g_small, g_large = detectable_gain(small, trials=60), detectable_gain(large, trials=60)
    assert g_large is not None
    assert g_small is None or g_large < g_small, (g_small, g_large)
