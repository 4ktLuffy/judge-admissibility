"""Rubric disagreement discovery: contested items are ranked by how evenly raters split, splits are attributed to
a dispute between raters or noise within one, and contested items are grouped by the words of their reasons."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from pydantic_evals_admissibility import Certificate, Judgment
from pydantic_evals_admissibility._disagree import (
    Cluster,
    Rating,
    clarify_disagreements,
    find_disagreements,
    normalized,
    ratings_from_certificate,
    tokens,
)

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))

IMPERSONAL = 'The message is impersonal and does not address the user in second person.'
FRIENDLY = 'The message is friendly, clear and suitable for display.'
TERSE = 'The message is terse and cold, with no greeting.'
WARM = 'The message is warm and polite, a good greeting.'


def disputed() -> list[Rating]:
    """Two style disputes (person, warmth), one noisy rater on one item, and one item everyone passes."""
    ratings = []
    for item in ('We interpret it as May.', 'Conflicting dates.'):
        ratings += [Rating(item, 'strict', False, IMPERSONAL) for _ in range(3)]
        ratings += [Rating(item, 'lenient', True, FRIENDLY) for _ in range(3)]
    for item in ('OK.', 'Done.'):
        ratings += [Rating(item, 'strict', False, TERSE), Rating(item, 'lenient', True, WARM)]
    ratings += [Rating('You asked for May.', 'strict', p, 'It names the month plainly.') for p in (True, False, True)]
    ratings += [Rating('You asked for May.', 'lenient', True, 'It names the month plainly.')]
    ratings += [Rating('You asked for June.', r, True, FRIENDLY) for r in ('strict', 'lenient')]
    ratings += [Rating('Errored.', 'strict', None), Rating('Errored.', 'lenient', True, FRIENDLY)]
    return ratings


def test_tokens_keep_hyphenated_words_and_drop_stop_words() -> None:
    assert tokens('It is NOT in a second‑person, user-friendly style.') == ['second-person', 'user-friendly', 'style']
    assert normalized('a  b\n c') == 'a b c' and normalized({'x': 1}) == "{'x': 1}"


def test_items_are_ranked_and_each_split_is_attributed() -> None:
    report = find_disagreements(disputed())
    assert report.raters == ('strict', 'lenient')
    assert report.items == 6  # 'Errored.' has one completed verdict, below min_votes
    order = [i.item for i in report.contested]
    assert set(order[:4]) == {'We interpret it as May.', 'Conflicting dates.', 'OK.', 'Done.'}
    assert order[4] == 'You asked for May.' and 'You asked for June.' not in order
    first = report.contested[0]
    assert first.split == 1.0 and first.between and not first.within and first.kind == 'between raters'
    noisy = report.contested[4]
    assert noisy.within and not noisy.between and noisy.kind == 'within a rater' and noisy.split == 0.0
    assert noisy.by_rater == {'strict': (2, 3), 'lenient': (1, 1)}


def test_repeats_do_not_outvote_a_rater_asked_once() -> None:
    ratings = [Rating('x', f'judge {k}', False, IMPERSONAL) for k in range(3) for _ in range(4)]
    ratings.append(Rating('x', 'author', True))
    (item,) = find_disagreements(ratings).contested
    assert item.split == 0.5  # three raters to one, not twelve verdicts to one
    assert item.minority == ('author',)


def test_contested_items_cluster_by_their_reasons_with_both_readings_quoted() -> None:
    report = find_disagreements(disputed())
    between = [c for c in report.clusters if c.between]
    assert len(between) == 2
    person = next(c for c in between if any(i.item == 'Conflicting dates.' for i in c.items))
    assert {i.item for i in person.items} == {'We interpret it as May.', 'Conflicting dates.'}
    assert 'impersonal' in person.fail.terms and 'second' in person.fail.terms
    assert 'friendly' in person.passing.terms
    assert person.fail.quotes[0] == ('strict', IMPERSONAL) and person.passing.quotes[0] == ('lenient', FRIENDLY)
    warmth = next(c for c in between if c is not person)
    assert {i.item for i in warmth.items} == {'OK.', 'Done.'} and 'terse' in warmth.fail.terms
    # The noisy item is no rubric dispute: it sorts after the disputes.
    assert report.clusters[-1].between == 0


def test_a_rater_in_the_minority_everywhere_is_flagged() -> None:
    ratings = []
    for k in range(5):
        item = f'answer {k}'
        ratings += [Rating(item, r, True, 'Correct according to the policy.') for r in ('a', 'b', 'truth')]
        ratings.append(Rating(item, 'blind', False, 'The question is not provided, so it cannot be verified.'))
    report = find_disagreements(ratings)
    assert report.outliers() == {'a': (0, 5), 'b': (0, 5), 'truth': (0, 5), 'blind': (5, 5)}
    assert 'blind: 5/5' in report.table()


async def test_clarify_is_called_only_for_disputes_and_at_most_max_calls() -> None:
    seen: list[Cluster] = []

    def clarify(rubric: str, cluster: Cluster) -> str:
        seen.append(cluster)
        return '  Address the user as "you".  '

    report = await clarify_disagreements(find_disagreements(disputed()), 'Be friendly.', clarify, max_calls=1)
    assert len(seen) == 1 and seen[0].between
    assert report.clusters[0].clarification == 'Address the user as "you".'
    assert all(c.clarification is None for c in report.clusters[1:])

    async def async_clarify(rubric: str, cluster: Cluster) -> str:
        return 'x'

    every = await clarify_disagreements(find_disagreements(disputed()), 'r', async_clarify, max_calls=9)
    assert [c.clarification for c in every.clusters] == ['x', 'x', None]  # never the noise-only cluster
    data = every.to_dict()
    assert data['clusters'][0]['clarification'] == 'x' and data['contested'][0]['between']
    json.dumps(data)


def test_a_whitespace_copy_is_the_same_item_and_a_saved_certificate_reads_back() -> None:
    judgments = (
        Judgment('a', 'reference#0', 'No timeframe from your request.', False, 'not second person'),
        Judgment('a', 'reference#1', 'No timeframe from your request.', False, 'not second person'),
        Judgment('a', 'must_hold:whitespace', 'No  timeframe from  your request.\n', True, 'refers to "your request"'),
        Judgment('b', 'reference#0', 'Done.', None, error='timeout'),
    )
    cert = Certificate('UNVALIDATED', (), judgments)
    ratings = ratings_from_certificate(cert.to_dict(), rater='judge')
    assert ratings == ratings_from_certificate(cert, rater='judge')
    report = find_disagreements(ratings)
    (item,) = report.contested
    assert item.item == 'No timeframe from your request.' and item.by_rater == {'judge': (1, 3)} and item.within
    skip_b = ratings_from_certificate(cert, rater='judge', item=lambda j: None if j.case == 'b' else j.case)
    assert {r.item for r in skip_b} == {'a'}


def test_the_bench_finds_the_rubric_dispute_on_pydantics_example_dataset() -> None:
    from disagreements import pydantic_ratings  # noqa: E402

    rubric, ratings = pydantic_ratings()
    report = find_disagreements(ratings)
    assert 'second-person or friendly' in rubric
    disputed_items = {i.item for i in report.contested if i.between}
    assert 'Conflicting time instructions: 2025 and 2020 cannot both apply.' in disputed_items
    assert report.outliers()['dataset author (expected_output)'][0] >= 3
    top = report.clusters[0]
    assert 'impersonal' in top.fail.terms
