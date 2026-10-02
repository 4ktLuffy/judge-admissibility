"""Delayed outcomes: the judge against what happened, and its pass rate corrected by what that showed."""

from __future__ import annotations

import random

import pytest

from pydantic_evals_admissibility._outcomes import outcome_calibration, recalibrated_pass_rate


def simulate(
    n: int, truth: float, sens: float, spec: float, seed: int, observe: tuple[float, float] = (1.0, 1.0)
) -> tuple[dict[str, bool], dict[str, bool | None], float]:
    """Verdicts, outcomes (None until observed, with a chance per verdict: passed, failed), and the true rate."""
    rng = random.Random(seed)
    verdicts, outcomes = {}, {}
    successes = 0
    for i in range(n):
        ok = rng.random() < truth
        successes += ok
        passed = rng.random() < (sens if ok else 1 - spec)
        verdicts[f't{i}'] = passed
        outcomes[f't{i}'] = ok if rng.random() < observe[0 if passed else 1] else None
    return verdicts, outcomes, successes / n


def test_a_perfect_judge() -> None:
    verdicts, outcomes, truth = simulate(400, 0.6, 1.0, 1.0, seed=1)
    report = outcome_calibration(verdicts, outcomes)
    assert report.sensitivity.rate == report.specificity.rate == report.agreement.rate == 1.0
    assert report.ppv.rate == report.npv.rate == 1.0 and not report.biased and not report.warnings
    assert report.sensitivity.interval[0] > 0.98  # Wilson, case as the unit
    estimate = recalibrated_pass_rate(verdicts, report)
    assert estimate.corrected == pytest.approx(truth) == estimate.raw.rate


def test_a_lenient_judge_overstates_and_the_correction_fixes_it() -> None:
    """Sensitivity 0.95, specificity 0.7, true success 0.6: the judge passes about 0.69."""
    covered, raw_error, corrected_error = 0, [], []
    for day in range(40):
        calibration_v, calibration_o, _ = simulate(1000, 0.6, 0.95, 0.7, seed=2 * day)
        traffic, _, truth = simulate(1000, 0.6, 0.95, 0.7, seed=2 * day + 1)
        report = outcome_calibration(calibration_v, calibration_o, resamples=300)
        estimate = recalibrated_pass_rate(traffic, report, resamples=300)
        assert estimate.corrected is not None and estimate.interval is not None
        raw_error.append(estimate.raw.rate - truth)  # type: ignore[operator]
        corrected_error.append(estimate.corrected - truth)
        covered += estimate.interval[0] <= truth <= estimate.interval[1]
    assert sum(raw_error) / 40 > 0.07  # the raw pass rate overstates success by about 9 points
    assert abs(sum(corrected_error) / 40) < 0.01
    assert covered >= 36  # a 95% interval, 40 days


def test_pending_outcomes_are_reported_and_not_counted() -> None:
    verdicts = {'a': True, 'b': True, 'c': False, 'd': False, 'e': True}
    outcomes = {'a': True, 'b': None, 'c': False}  # 'b' explicitly pending; 'd' and 'e' not yet arrived
    report = outcome_calibration(verdicts, outcomes, resamples=50)
    assert (report.cases, report.observed, report.pending) == (5, 2, 3)
    assert report.sensitivity.trials == 1 and report.specificity.trials == 1
    assert '3 of 5 cases have no outcome yet' in report.warnings[0]
    with pytest.raises(ValueError):
        outcome_calibration(verdicts, {'zz': True})


def test_outcomes_observed_only_for_passed_verdicts_are_flagged_and_not_corrected() -> None:
    """Only passed tickets can be reopened: nothing is known about the failed ones."""
    verdicts, outcomes, _ = simulate(2000, 0.6, 0.95, 0.7, seed=3, observe=(0.3, 0.0))
    report = outcome_calibration(verdicts, outcomes, resamples=200)
    assert report.biased and report.observed_failed.successes == 0
    assert report.sensitivity.rate == 1.0  # an artifact: every observed success was passed
    assert any('no outcomes observed for failed verdicts' in w for w in report.warnings)
    assert report.adjusted_sensitivity.value is None
    estimate = recalibrated_pass_rate(verdicts, report)
    assert estimate.corrected is None and any('cannot correct' in w for w in estimate.warnings)


def test_verdict_dependent_observation_is_flagged_with_the_gap_and_adjusted() -> None:
    verdicts, outcomes, _ = simulate(4000, 0.6, 0.95, 0.7, seed=4, observe=(0.45, 0.1))
    report = outcome_calibration(verdicts, outcomes, resamples=300)
    assert report.biased
    gap = report.observation_gap
    assert gap.value == pytest.approx(0.35, abs=0.04) and gap.interval is not None and gap.interval[0] > 0
    assert any('45% of passed and 10% of failed' in w for w in report.warnings)
    # Observed failures are mostly passed ones, so specificity on observed cases is far too low.
    assert report.specificity.rate is not None and report.specificity.rate < 0.45
    assert report.adjusted_specificity.value == pytest.approx(0.7, abs=0.05)
    assert report.adjusted_sensitivity.value == pytest.approx(0.95, abs=0.02)


def test_uniform_observation_is_not_flagged() -> None:
    verdicts, outcomes, _ = simulate(2000, 0.6, 0.95, 0.7, seed=5, observe=(0.3, 0.3))
    report = outcome_calibration(verdicts, outcomes, resamples=100)
    assert not report.biased


def test_slices_break_the_report_down() -> None:
    verdicts = {f'r{i}': True for i in range(10)} | {f's{i}': i < 5 for i in range(10)}
    outcomes = {**{f'r{i}': True for i in range(10)}, **{f's{i}': True for i in range(10)}}
    slices = {c: c[0] for c in verdicts}
    report = outcome_calibration(verdicts, outcomes, slices=slices, resamples=50)
    assert report.slices['r'].sensitivity.rate == 1.0 and report.slices['s'].sensitivity.rate == 0.5
    assert 'slices' in report.to_dict() and 's ' in report.table()


def test_a_judge_no_better_than_chance_cannot_be_corrected() -> None:
    verdicts, outcomes, _ = simulate(1000, 0.6, 0.5, 0.5, seed=7)
    estimate = recalibrated_pass_rate(verdicts, outcome_calibration(verdicts, outcomes, resamples=100), resamples=300)
    # The point estimate may exist by luck, but the interval and the warning must say the judge is near chance.
    assert estimate.dropped > 0 and estimate.interval is not None
    assert estimate.interval[1] - estimate.interval[0] > 0.5 and any('chance' in w for w in estimate.warnings)
