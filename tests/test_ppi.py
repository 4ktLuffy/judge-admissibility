"""`ppi_pass_rate`: the agent's pass rate from a judge on everything and people on a uniform random audit."""

from __future__ import annotations

import random

import pytest

from pydantic_evals_admissibility._ppi import _estimate, audit_sample, ppi_pass_rate


def traffic(n: int, truth: float, sens: float, spec: float, seed: int) -> tuple[dict[str, bool], dict[str, bool]]:
    """Judge verdicts and true labels for `n` outputs."""
    rng = random.Random(seed)
    verdicts, labels = {}, {}
    for i in range(n):
        ok = rng.random() < truth
        labels[f'o{i}'] = ok
        verdicts[f'o{i}'] = rng.random() < (sens if ok else 1 - spec)
    return verdicts, labels


def run(verdicts: dict[str, bool], labels: dict[str, bool], n: int, seed: int, tuned: bool = True):  # type: ignore[no-untyped-def]
    sample = audit_sample(verdicts, n, seed)
    return ppi_pass_rate(verdicts, {c: labels[c] for c in sample.cases}, sample=sample, tuned=tuned)


def test_unbiased_where_the_judge_alone_is_not() -> None:
    """A lenient judge (sensitivity 0.95, specificity 0.7) overstates a 0.6 pass rate by about 9 points."""
    errors: dict[str, list[float]] = {'ppi': [], 'ppi++': [], 'judge': []}
    for trial in range(300):
        verdicts, labels = traffic(2000, 0.6, 0.95, 0.7, seed=trial)
        truth = sum(labels.values()) / len(labels)
        plain, tuned = run(verdicts, labels, 100, trial, tuned=False), run(verdicts, labels, 100, trial)
        errors['ppi'].append(plain.estimate - truth)
        errors['ppi++'].append(tuned.estimate - truth)
        errors['judge'].append(plain.judge_only - truth)
    bias = {name: sum(e) / len(e) for name, e in errors.items()}
    assert bias['judge'] > 0.07
    assert abs(bias['ppi']) < 0.01 and abs(bias['ppi++']) < 0.01


def test_a_perfect_judge_shrinks_the_interval_to_its_own_rate() -> None:
    verdicts, labels = traffic(1000, 0.6, 1.0, 1.0, seed=1)
    estimate = run(verdicts, labels, 50, 0)
    truth = sum(labels.values()) / len(labels)
    assert estimate.estimate == pytest.approx(truth) == estimate.judge_only
    assert (estimate.false_fails, estimate.false_passes) == (0, 0) and estimate.lam == 1.0
    assert 'zero width' in estimate.warnings[0]  # the CLT interval would be a point at the judge's rate
    assert estimate.interval_method == 'exact' and estimate.interval == estimate.exact_interval
    low, high = estimate.exact_interval  # the finite-sample one is honest about 50 labels
    assert low < truth < high and high - low < estimate.human_only_interval[1] - estimate.human_only_interval[0]


def test_a_judge_constant_on_the_audit_keeps_its_weight() -> None:
    """Every audited output passed, by people and judge: lam is undefined, and 0 would discard a perfect judge."""
    estimate = _estimate(900, 1000, [(True, True)] * 20, 0.05, tuned=True)
    assert estimate.lam == 1.0 and estimate.estimate == pytest.approx(0.9)


def test_a_useless_judge_costs_plain_ppi_and_ppi_plus_plus_recovers() -> None:
    """A verdict independent of the truth: the rectifier is noisier than the labels alone."""
    plain_ratio, tuned_ratio, lams = [], [], []
    for trial in range(200):
        verdicts, labels = traffic(2000, 0.6, 0.7, 0.3, seed=100 + trial)
        plain, tuned = run(verdicts, labels, 100, trial, tuned=False), run(verdicts, labels, 100, trial)
        assert plain.width_ratio is not None and tuned.width_ratio is not None
        plain_ratio.append(plain.width_ratio)
        tuned_ratio.append(tuned.width_ratio)
        lams.append(tuned.lam)
    assert sum(plain_ratio) / 200 > 1.2  # plain PPI is wider than people alone
    assert sum(tuned_ratio) / 200 < 1.02  # PPI++ is not
    assert sum(lams) / 200 < 0.15


def test_tuned_ppi_is_the_post_stratified_estimate_and_not_rogan_gladen() -> None:
    """Before clipping, PPI++ is q * PPV + (1 - q) * (1 - NPV); Rogan-Gladen divides by sens + spec - 1 instead."""
    pairs = [(True, True)] * 30 + [(False, True)] * 10 + [(True, False)] * 5 + [(False, False)] * 15
    estimate = _estimate(3500, 5000, pairs, 0.05, tuned=True)
    q, ppv, false_omission = 0.7, 30 / 40, 5 / 20
    assert 0 < estimate.lam < 1
    assert estimate.estimate == pytest.approx(q * ppv + (1 - q) * false_omission)
    sens, spec = 30 / 35, 15 / 25
    rogan_gladen = (q + spec - 1) / (sens + spec - 1)
    assert abs(rogan_gladen - estimate.estimate) > 0.02


def test_small_audits_of_a_good_judge_fall_back_to_the_exact_interval() -> None:
    """A judge wrong on 2% of outputs, 30 audited: half the audits see no mistake, and the CLT interval is a point.

    Measured with this seed, before the fallback: the CLT interval covered 453 of 1000 times, the exact one
    1000 times. With it, `interval` covered 999.
    """
    rng = random.Random(3)
    clt = exact = fallback = 0
    trials = 1000
    for _ in range(trials):
        cells = [0, 0, 0, 0]  # (human, judge): pass/pass, pass/fail, fail/pass, fail/fail
        for _ in range(5000):
            ok = rng.random() < 0.6
            wrong = rng.random() < 0.02
            cells[(1 if wrong else 0) if ok else (2 if wrong else 3)] += 1
        truth = (cells[0] + cells[1]) / 5000
        kinds = ((True, True), (True, False), (False, True), (False, False))
        bounds = [sum(cells[: i + 1]) for i in range(4)]
        pairs = [kinds[next(c for c, b in enumerate(bounds) if i < b)] for i in rng.sample(range(5000), 30)]
        e = _estimate(cells[0] + cells[2], 5000, pairs, 0.05, tuned=False)
        clt += e.interval[0] <= truth <= e.interval[1]
        exact += e.exact_interval[0] <= truth <= e.exact_interval[1]
        fallback += e.interval_method == 'exact'
    assert 0.4 < fallback / trials < 0.65
    assert clt / trials >= 0.95
    assert exact / trials >= 0.95


def test_an_audit_that_was_not_drawn_at_random_is_refused() -> None:
    verdicts, labels = traffic(200, 0.6, 0.9, 0.8, seed=4)
    sample = audit_sample(verdicts, 20, seed=0)
    failures = {c: labels[c] for c in verdicts if not verdicts[c]}  # a review queue of the judge's failures
    with pytest.raises(ValueError, match='not drawn'):
        ppi_pass_rate(verdicts, failures, sample=sample)
    with pytest.raises(ValueError, match='no label'):  # dropping drawn cases that were hard to label
        ppi_pass_rate(verdicts, {c: labels[c] for c in sample.cases[:15]}, sample=sample)
    swapped = sample.__class__((*sample.cases[:-1], next(c for c in verdicts if c not in sample.cases)),
                               sample.population, sample.seed, sample.digest)  # fmt: skip
    with pytest.raises(ValueError, match='edited'):
        ppi_pass_rate(verdicts, {c: labels[c] for c in swapped.cases}, sample=swapped)
    other = {**verdicts, 'extra': True}
    with pytest.raises(ValueError, match='different set of outputs'):
        ppi_pass_rate(other, {c: labels[c] for c in sample.cases}, sample=sample)


def test_audit_sample_is_uniform_and_reproducible() -> None:
    names = [f'o{i}' for i in range(10)]
    assert audit_sample(names, 4, 7) == audit_sample(reversed(names), 4, 7)  # order of the names does not matter
    counts = dict.fromkeys(names, 0)
    for seed in range(2000):
        for case in audit_sample(names, 3, seed).cases:
            counts[case] += 1
    assert all(500 < c < 700 for c in counts.values())  # 600 expected each


def test_validation() -> None:
    names = ['a', 'b', 'c']
    with pytest.raises(ValueError, match='unique'):
        audit_sample(['a', 'a', 'b'], 2)
    with pytest.raises(ValueError, match='between 2 and 3'):
        audit_sample(names, 1)
    with pytest.raises(ValueError, match='between 2 and 3'):
        audit_sample(names, 4)
    sample = audit_sample(names, 2)
    verdicts = dict.fromkeys(names, True)
    with pytest.raises(ValueError, match='alpha'):
        ppi_pass_rate(verdicts, dict.fromkeys(sample.cases, True), sample=sample, alpha=0)


def test_summary_and_to_dict() -> None:
    verdicts, labels = traffic(500, 0.6, 0.9, 0.8, seed=5)
    estimate = run(verdicts, labels, 60, 0)
    assert estimate.summary().startswith('PPI++: ')
    data = estimate.to_dict()
    assert data['n_audit'] == 60 and data['n_total'] == 500 and len(data['interval']) == 2
    assert data['false_fails'] + data['false_passes'] <= 60


def _two_false_fails() -> tuple[dict[str, bool], dict[str, bool], object]:
    """100 outputs, the judge passes 50; the two audited outputs are both ones people pass and the judge fails."""
    names = [f'o{i}' for i in range(100)]
    sample = audit_sample(names, 2, seed=0)
    others = [c for c in names if c not in sample.cases]
    verdicts = {c: c in others[:50] for c in names}
    return verdicts, dict.fromkeys(sample.cases, True), sample


@pytest.mark.parametrize('tuned', [True, False])
def test_a_degenerate_audit_gives_an_ordered_interval_inside_zero_one(tuned: bool) -> None:
    """Regression: estimate 1.0 with interval (1.5, 1.0), a negative width ratio, and no warning."""
    verdicts, labels, sample = _two_false_fails()
    estimate = ppi_pass_rate(verdicts, labels, sample=sample, tuned=tuned)  # type: ignore[arg-type]
    for low, high in (estimate.interval, estimate.exact_interval):
        assert 0.0 <= low <= estimate.estimate <= high <= 1.0
    assert estimate.width_ratio is not None and estimate.width_ratio >= 0
    assert estimate.interval == estimate.exact_interval and estimate.interval_method == 'exact'
    assert estimate.raw_estimate == pytest.approx(1.5)
    assert any('outside [0, 1]' in w for w in estimate.warnings)
    assert any('exact' in w for w in estimate.warnings)


def test_every_zero_variance_audit_falls_back_to_the_exact_interval() -> None:
    """Each way the residual can be constant on a non-census audit: no disagreement, all one kind of mistake,
    and PPI++ choosing lam = 0 where people agree with each other."""
    audits = {
        'agree': [(True, True)] * 10 + [(False, False)] * 10,
        'all false fails': [(True, False)] * 5,
        'all false passes': [(False, True)] * 5,
        'people constant, judge varies': [(True, True)] * 3 + [(True, False)] * 3,
        'judge constant': [(True, True)] * 3 + [(False, True)] * 3,
    }
    for judge_passes in (0, 1, 50, 99, 100):
        for name, pairs in audits.items():
            for tuned in (True, False):
                e = _estimate(judge_passes, 100, pairs, 0.05, tuned)
                assert 0.0 <= e.interval[0] <= e.estimate <= e.interval[1] <= 1.0, (name, judge_passes, tuned)
                assert 0.0 <= e.exact_interval[0] <= e.estimate <= e.exact_interval[1] <= 1.0
                assert e.width_ratio is None or e.width_ratio >= 0
                if e.interval[1] - e.interval[0] == 0:
                    assert e.interval_method == 'census', (name, judge_passes, tuned)
                if e.interval_method == 'exact':
                    assert e.interval == e.exact_interval and e.warnings


def test_a_census_is_exact_and_not_a_fallback() -> None:
    pairs = [(True, True)] * 6 + [(True, False)] * 2 + [(False, False)] * 2
    e = _estimate(6, 10, pairs, 0.05, tuned=True)
    assert e.interval_method == 'census' and e.estimate == pytest.approx(0.8)
    assert e.interval == pytest.approx((0.8, 0.8)) == e.exact_interval


def test_plain_ppi_before_clipping_is_unbiased_and_clipping_and_ppi_plus_plus_are_not() -> None:
    """Four outputs that truly pass, the judge passes two, every audit of two: the reviewer's counterexample.

    The audit of the judge's two failures gives plain PPI 1.5, clipped to 1; PPI++ sets lam = 0 on the four
    mixed audits. Both average 11/12, not 1. Unclipped plain PPI averages exactly 1.
    """
    from itertools import combinations

    judge = [True, True, False, False]
    raw, clipped, tuned = [], [], []
    for audit in combinations(range(4), 2):
        pairs = [(True, judge[i]) for i in audit]
        plain = _estimate(2, 4, pairs, 0.05, tuned=False)
        raw.append(plain.raw_estimate)
        clipped.append(plain.estimate)
        tuned.append(_estimate(2, 4, pairs, 0.05, tuned=True).estimate)
    assert sum(raw) / len(raw) == pytest.approx(1.0)
    assert sum(clipped) / len(clipped) == pytest.approx(11 / 12)
    assert sum(tuned) / len(tuned) == pytest.approx(11 / 12)
