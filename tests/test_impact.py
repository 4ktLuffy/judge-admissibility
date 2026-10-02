"""Judge-change impact: re-deciding past comparisons under a new judge finds the decisions that flip, says which
way, and flags the shipped ones the new judge would reject."""

from __future__ import annotations

import pytest

from pydantic_evals_admissibility import ComparisonRecord, GateRules, decision_impact

FAST = GateRules(resamples=2000)
N = 40


def outcomes(passing: int) -> dict[str, list[bool]]:
    """N cases, one outcome each; the first `passing` pass."""
    return {f'c{i}': [i < passing] for i in range(N)}


def record(old: tuple[int, int], new: tuple[int, int], taken: str | None = None) -> ComparisonRecord:
    """Baseline and candidate pass counts under the old and the new judge."""
    return ComparisonRecord(
        old={'baseline': outcomes(old[0]), 'candidate': outcomes(old[1])},
        new={'baseline': outcomes(new[0]), 'candidate': outcomes(new[1])},
        taken=taken,  # type: ignore[arg-type]
    )


def test_flips_are_found_with_their_direction() -> None:
    report = decision_impact(
        {
            'steady promote': record((10, 30), (12, 32)),
            'promote to inconclusive': record((10, 30), (10, 12)),
            'inconclusive to promote': record((20, 21), (5, 25)),
            'steady inconclusive': record((20, 20), (22, 22)),
        },
        rules=FAST,
    )
    r = report.records
    assert (r['steady promote'].old, r['steady promote'].new) == ('PROMOTE', 'PROMOTE')
    assert not r['steady promote'].flipped and r['steady promote'].concern == 'none'
    assert r['promote to inconclusive'].direction == 'PROMOTE->INCONCLUSIVE'
    assert r['promote to inconclusive'].concern == 'unsupported'  # shipped, and the new judge cannot back it
    assert r['inconclusive to promote'].direction == 'INCONCLUSIVE->PROMOTE'
    assert r['inconclusive to promote'].concern == 'missed'
    assert r['promote to inconclusive'].gain_old == pytest.approx(0.5)
    assert r['promote to inconclusive'].gain_new == pytest.approx(0.05)
    assert report.flipped == ['promote to inconclusive', 'inconclusive to promote']
    counts = report.counts()
    assert counts['flipped'] == 2 and counts['flip PROMOTE->INCONCLUSIVE'] == 1
    assert counts['dangerous'] == 0
    assert '2 of 4 decisions flip' in report.table()


def test_a_shipped_release_the_new_judge_rejects_is_dangerous() -> None:
    report = decision_impact(
        {
            # The gate said INCONCLUSIVE, but a naive optimizer shipped it on the raw score.
            'shipped on raw score': record((20, 22), (30, 10), taken='PROMOTE'),
            'gate promoted': record((10, 30), (30, 10)),
            'held back': record((20, 22), (30, 10), taken='INCONCLUSIVE'),
        },
        rules=FAST,
    )
    assert report.records['shipped on raw score'].old == 'INCONCLUSIVE'
    assert report.records['shipped on raw score'].direction == 'INCONCLUSIVE->REJECT'
    assert report.records['gate promoted'].direction == 'PROMOTE->REJECT'
    assert report.dangerous == ['shipped on raw score', 'gate promoted']
    assert report.records['held back'].concern == 'none'  # flipped, but nothing was shipped
    assert report.to_dict()['counts']['dangerous'] == 2  # type: ignore[index]


def test_rules_are_the_ones_the_decisions_were_made_with() -> None:
    history = {'marginal': record((14, 22), (14, 22))}
    assert decision_impact(history, rules=FAST).records['marginal'].old == 'PROMOTE'
    assert decision_impact(history, rules=FAST.for_candidates(20)).records['marginal'].old == 'INCONCLUSIVE'


def test_an_empty_history_is_an_error() -> None:
    with pytest.raises(ValueError, match='no past comparisons'):
        decision_impact({})
