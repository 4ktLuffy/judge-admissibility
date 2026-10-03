"""`certify_confidence_gate` and `calibrate_gate`: a judge that decides only when sure, certified honestly."""

from __future__ import annotations

import math
import random

import pytest

from pydantic_evals_admissibility._confidence_gate import (
    ConfidenceGate,
    ConfidenceRequirements,
    GateCase,
    _tally,  # pyright: ignore[reportPrivateUsage]
    calibrate_gate,
    certify_confidence_gate,
    choose_gate,
    confidence_grid,
    gate_cases,
)
from pydantic_evals_admissibility._stats import clopper_pearson

GATE = ConfidenceGate(0.2, 0.8)


def test_from_confidence_matches_the_plan() -> None:
    """#9723: at cutoff 0.5, confidence 0.6 decides at p >= 0.8 or p <= 0.2, equality accepted."""
    assert ConfidenceGate.from_confidence(0.6) == GATE
    assert GATE.decide(0.8) == 'pass' and GATE.decide(0.2) == 'fail' and GATE.decide(0.5) == 'defer'
    ungated = ConfidenceGate.from_confidence(0.0)
    assert ungated == ConfidenceGate(0.5, 0.5)
    assert ungated.decide(0.5) == 'pass' and ungated.decide(0.4999) == 'fail'
    assert ConfidenceGate.from_confidence(0.5, cutoff=0.6) == ConfidenceGate(0.3, 0.8)
    with pytest.raises(ValueError):
        ConfidenceGate(0.8, 0.2)
    assert len(confidence_grid()) == 20


def test_counts_and_exact_intervals() -> None:
    probabilities = {'a': 0.95, 'b': 0.9, 'c': 0.85, 'd': 0.1, 'e': 0.05, 'f': 0.5, 'g': 0.6, 'h': 0.3}
    labels = {'a': True, 'b': True, 'c': False, 'd': False, 'e': True, 'f': True, 'g': False, 'h': False}
    report = certify_confidence_gate(gate_cases(probabilities, labels), GATE)
    assert (report.cases, report.decided, report.deferred) == (8, 5, 3)
    assert (report.auto_correct, report.false_passes, report.false_fails) == (3, 1, 1)
    accuracy = report.check('auto_accuracy')
    coverage = report.check('coverage')
    assert accuracy is not None and coverage is not None
    assert accuracy.interval == clopper_pearson(3, 5) and coverage.interval == clopper_pearson(5, 8)
    assert report.per_case['f'] == 'defer' and report.end_to_end is None
    assert report.status == 'UNVALIDATED'  # 5 decided cases, fewer than min_trials


def test_a_good_gate_passes_and_a_bad_one_fails() -> None:
    good = [GateCase(f'p{i}', 0.95, True) for i in range(40)] + [GateCase(f'f{i}', 0.05, False) for i in range(40)]
    assert certify_confidence_gate(good, GATE).status == 'PASS'
    # Confident and wrong half the time: shown below a 0.9 bar.
    bad = [GateCase(f'p{i}', 0.95, i % 2 == 0) for i in range(80)]
    report = certify_confidence_gate(bad, GATE)
    assert report.status == 'FAIL' and 'auto_accuracy' in report.reason


def test_repeats_count_once_and_measure_stability() -> None:
    cases = [GateCase(f'c{i}', 0.95, True) for i in range(12)] + [GateCase(f'n{i}', 0.05, False) for i in range(12)]
    repeats = [GateCase('c0', 0.5, True), GateCase('c1', 0.97, True)]
    report = certify_confidence_gate(cases + repeats, GATE)
    assert report.cases == 24 and report.decided == 24  # c0's deferral on repeat is not a new case
    assert report.repeated_cases == 2 and report.stability == 0.5
    with pytest.raises(ValueError, match='labelled both'):
        certify_confidence_gate([GateCase('x', 0.9, True), GateCase('x', 0.9, False)], GATE)


def test_everything_deferred() -> None:
    cases = [GateCase(f'c{i}', 0.5, i % 2 == 0) for i in range(30)]
    report = certify_confidence_gate(cases, GATE)
    accuracy = report.check('auto_accuracy')
    assert accuracy is not None and accuracy.status == 'UNVALIDATED' and accuracy.interval == (0.0, 1.0)
    assert report.decided == 0 and report.coverage == 0 and report.accuracy is None
    assert report.status == 'UNVALIDATED'


def test_nothing_above_the_high_threshold() -> None:
    cases = [GateCase(f'c{i}', 0.1, False) for i in range(20)] + [GateCase(f'd{i}', 0.6, True) for i in range(20)]
    report = certify_confidence_gate(cases, GATE)
    assert (report.decided, report.false_passes, report.auto_correct) == (20, 0, 20)


def test_one_class_is_never_a_pass() -> None:
    """All labels pass and the judge passes all: a constant judge would do as well."""
    cases = [GateCase(f'c{i}', 0.99, True) for i in range(60)]
    report = certify_confidence_gate(cases, GATE)
    accuracy = report.check('auto_accuracy')
    assert accuracy is not None and accuracy.status == 'PASS'
    assert report.status == 'UNVALIDATED' and 'same' in report.reason


def test_end_to_end_with_a_fallback() -> None:
    sure = [GateCase(f's{i}', 0.95 if i % 2 else 0.05, bool(i % 2)) for i in range(60)]
    # The fallback agrees with people on every deferred case but u0.
    unsure = [GateCase(f'u{i}', 0.5, bool(i % 2), fallback=bool(i % 2) != (i == 0)) for i in range(10)]
    report = certify_confidence_gate(sure + unsure, GATE, requirements=ConfidenceRequirements(min_end_to_end=0.8))
    assert report.end_to_end_correct == 69 and report.end_to_end == 69 / 70
    check = report.check('end_to_end')
    assert check is not None and check.interval == clopper_pearson(69, 70) and check.status == 'PASS'
    assert report.status == 'PASS'
    with pytest.raises(ValueError, match='no fallback verdict'):
        certify_confidence_gate(sure + unsure + [GateCase('u99', 0.5, True)], GATE)
    without = certify_confidence_gate(
        [GateCase(c.name, c.probability, c.label) for c in sure + unsure],
        GATE,
        requirements=ConfidenceRequirements(min_end_to_end=0.8),
    )
    assert without.end_to_end is None and without.status == 'UNVALIDATED'


def test_thresholds_chosen_on_the_same_cases_do_not_pass() -> None:
    cases = [GateCase(f'p{i}', 0.95, True) for i in range(40)] + [GateCase(f'f{i}', 0.05, False) for i in range(40)]
    report = certify_confidence_gate(cases, GATE, chosen_on=[c.name for c in cases[:10]])
    assert report.status == 'UNVALIDATED' and report.chosen_on == 10 and 'optimistic' in report.reason


def test_choose_gate_returns_none_when_nothing_qualifies() -> None:
    cases = [GateCase(f'c{i}', 0.99, i % 2 == 0) for i in range(100)]
    assert choose_gate(cases, min_accuracy=0.9) is None
    selection = calibrate_gate(cases)
    assert selection.chosen is None and selection.status == 'UNVALIDATED'


def test_the_split_never_puts_a_case_on_both_sides() -> None:
    rng = random.Random(1)
    cases = [GateCase(f'c{i}', rng.random(), rng.random() < 0.6) for i in range(100)]
    cases += [GateCase('c0', cases[0].probability, cases[0].label)]
    selection = calibrate_gate(cases, requirements=ConfidenceRequirements(min_accuracy=0.5))
    assert not set(selection.selection_cases) & set(selection.confirmation_cases)
    assert len(selection.selection_cases) + len(selection.confirmation_cases) == 100


# Selection bias. A judge whose probability is a noisy sigmoid of the truth: accuracy rises with
# the confidence demanded, from 0.79 ungated to 0.99 at the top of the grid; 0.9 is crossed at
# about confidence 0.55, so the choice is near the bar, where optimism matters.


def _draw(n: int, rng: random.Random, sep: float = 1.2) -> list[GateCase]:
    out: list[GateCase] = []
    for i in range(n):
        label = rng.random() < 0.7
        score = sep * (1 if label else -1) + rng.gauss(0, 1.5)
        out.append(GateCase(f'c{i}', 1 / (1 + math.exp(-score)), label))
    return out


def _true_accuracy(sep: float = 1.2) -> dict[ConfidenceGate, float]:
    population = _draw(100_000, random.Random(12345), sep)
    out: dict[ConfidenceGate, float] = {}
    for gate in confidence_grid():
        decided, correct, _, _ = _tally(gate, population)
        out[gate] = correct / decided
    return out


def test_in_sample_thresholds_overstate_accuracy_and_held_out_ones_do_not() -> None:
    """390 cases, as in #9641. The optimism shrinks with more cases (bench/confidence_gate.py measures it)."""
    truth = _true_accuracy()
    in_sample: list[float] = []
    held_out: list[float] = []
    for trial in range(150):
        cases = _draw(390, random.Random(trial))
        selection = calibrate_gate(cases, seed=trial)
        if selection.chosen is not None and selection.confirmed is not None:
            assert selection.confirmed.accuracy is not None
            held_out.append(selection.confirmed.accuracy - truth[selection.chosen])
        # The widest gate whose exact lower bound on these labels is >= 0.9, then reported on them.
        gate = choose_gate(cases, min_accuracy=0.9)
        if gate is not None:
            report = certify_confidence_gate(cases, gate)
            assert report.accuracy is not None
            in_sample.append(report.accuracy - truth[gate])
    bias_in = sum(in_sample) / len(in_sample)
    bias_out = sum(held_out) / len(held_out)
    assert bias_in > 0.01 and sum(b > 0 for b in in_sample) / len(in_sample) > 0.9, bias_in
    assert len(held_out) >= 30 and abs(bias_out) < 0.012, (len(held_out), bias_out)


@pytest.mark.parametrize('method', ['split', 'fixed_sequence'])
def test_honest_selection_does_not_pass_a_gate_below_the_bar(method: str) -> None:
    """Negative control: every gate's true accuracy is below the 0.97 bar, so (almost) no PASS may come out."""
    truth = _true_accuracy(sep=0.9)
    assert max(truth.values()) < 0.97
    requirements = ConfidenceRequirements(min_accuracy=0.97)
    passes = 0
    for trial in range(40):
        selection = calibrate_gate(
            _draw(390, random.Random(1000 + trial), sep=0.9),
            requirements=requirements,
            method=method,  # type: ignore[arg-type]
            seed=trial,
        )
        passes += selection.status == 'PASS'
    assert passes <= 2


def test_fixed_sequence_finds_a_gate_when_one_is_clearly_good() -> None:
    selection = calibrate_gate(
        _draw(390, random.Random(7)), requirements=ConfidenceRequirements(min_accuracy=0.85), method='fixed_sequence'
    )
    assert selection.status == 'PASS' and selection.chosen is not None
    assert selection.tested and all(t[4] for t in selection.tested[:-1])


# G1 (review): replies to one question are not independent evidence. 40 questions, each with two
# agent replies and one plausible wrong answer; the judge is right on all but three outputs, each
# in a different question. That is the shape of the real run in bench/confidence_gate.py.


def _questions(errors: int = 3) -> list[GateCase]:
    out: list[GateCase] = []
    for q in range(40):
        for kind, label in (('reply0', True), ('reply1', True), ('wrong', False)):
            wrong_verdict = kind == 'wrong' and q < errors  # the judge passes a wrong answer
            p = 0.99 if label or wrong_verdict else 0.01
            out.append(GateCase(f'q{q}#{kind}', p, label, group=f'q{q}'))
    return out


def test_grouped_outputs_count_once_per_group() -> None:
    cases = _questions()
    by_group = certify_confidence_gate(cases, GATE)
    accuracy = by_group.check('auto_accuracy')
    assert accuracy is not None and accuracy.trials == 40 and accuracy.successes == 37
    assert accuracy.interval == clopper_pearson(37, 40)
    assert by_group.units == 40 and by_group.cases == 120 and by_group.output_accuracy == 117 / 120
    assert by_group.status == 'UNVALIDATED'  # 37/40: lower bound 0.80, not shown above 0.9
    # Counting outputs, as the first version of the benchmark did, PASSes on 120 "independent" cases.
    by_output = certify_confidence_gate(cases, GATE, unit='case')
    check = by_output.check('auto_accuracy')
    assert check is not None and check.trials == 120 and by_output.status == 'PASS'


def test_ungrouped_cases_are_their_own_groups() -> None:
    cases = [GateCase(f'p{i}', 0.95, True) for i in range(40)] + [GateCase(f'f{i}', 0.05, False) for i in range(40)]
    a, b = certify_confidence_gate(cases, GATE), certify_confidence_gate(cases, GATE, unit='case')
    assert a.to_dict() | {'unit': None} == b.to_dict() | {'unit': None}


def test_a_split_never_shares_a_group() -> None:
    cases = _questions()
    for seed in range(10):
        selection = calibrate_gate(cases, seed=seed, candidates=confidence_grid(top=0.9))
        left = {n.split('#')[0] for n in selection.selection_cases}
        right = {n.split('#')[0] for n in selection.confirmation_cases}
        assert not left & right and len(left) + len(right) == 40


def test_a_case_in_a_group_the_thresholds_were_chosen_on_is_tainted() -> None:
    cases = [c for c in _questions(errors=0)]
    report = certify_confidence_gate(cases, GATE, chosen_on=['q0#reply0'])
    assert report.chosen_on == 3 and report.status == 'UNVALIDATED'


def test_a_group_is_right_end_to_end_only_if_every_output_is() -> None:
    cases = [
        GateCase(f'q{q}#{k}', 0.5, k == 'a', fallback=(k == 'a') != (q == 0 and k == 'b'), group=f'q{q}')
        for q in range(20)
        for k in ('a', 'b')
    ]
    report = certify_confidence_gate(cases, GATE)
    assert report.end_to_end_correct == 19 and report.units == 20 and report.fully_decided == 0
