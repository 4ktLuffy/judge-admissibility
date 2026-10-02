"""The ground truth has to be right before anything is measured against it."""

from __future__ import annotations

import sys
from collections import Counter
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from task import WEEKDAYS, Question, final_answer, is_correct, questions  # noqa: E402


def test_forty_unique_questions_in_four_kinds() -> None:
    qs = questions()
    assert len(qs) == 40 and len({q.name for q in qs}) == 40
    assert Counter(q.kind for q in qs) == {'arithmetic': 10, 'letters': 10, 'dates': 10, 'strings': 10}
    assert questions() == qs  # seeded: the same set every time


def test_answers_recomputed_independently() -> None:
    for q in questions():
        if q.kind == 'letters':
            letter, phrase = q.question.split('"')[1], q.question.split('"')[3]
            assert q.answer == str(sum(1 for c in phrase if c == letter))
        elif q.kind == 'dates':
            start = date.fromisoformat(q.question[:10])
            days = int(q.question.split(' days later')[0].split()[-1])
            assert WEEKDAYS[start.weekday()] in q.question
            assert q.answer == WEEKDAYS[(start + timedelta(days=days)).weekday()]
        elif q.kind == 'strings':
            word = q.question.split('"')[1]
            assert q.answer.lower() == word[::-1]
            assert all(c.isupper() == (c.lower() in 'aeiou') for c in q.answer)
        else:
            assert int(q.answer) == int(q.answer)  # arithmetic is checked in the next test


def test_arithmetic_by_hand() -> None:
    q = Question('t', 'arithmetic', '', '')
    # 30 boxes x 20 = 600; sell 100 -> 500; ship 2 boxes (40) -> 460; return 15 -> 475
    assert is_correct(Question('t', 'arithmetic', '', '475'), 'Work...\nAnswer: 475')
    assert not is_correct(q, 'no answer line')


def test_final_answer_parsing() -> None:
    assert final_answer('blah\nAnswer: 42') == '42'
    assert final_answer('**Answer:** Tuesday.') == 'Tuesday'
    assert final_answer('Answer: 1\nmore\nAnswer: 2') == '2'
    assert final_answer('The answer is 42') is None
    assert is_correct(Question('a', 'arithmetic', '', '1475'), 'Answer: 1,475')
    assert is_correct(Question('d', 'dates', '', 'Tuesday'), 'answer: tuesday')
    assert not is_correct(Question('s', 'strings', '', 'yrrEbwArts'), 'Answer: yrrebwarts')


def test_answers_with_units_or_extra_words_still_count() -> None:
    assert is_correct(Question('a', 'arithmetic', '', '200'), 'Answer: 200 pens')
    assert is_correct(Question('l', 'letters', '', '3'), 'Answer: 3 times')
    assert is_correct(Question('d', 'dates', '', 'Monday'), 'Answer: It is a Monday.')


def test_ambiguous_answer_lines_are_wrong() -> None:
    assert not is_correct(Question('a', 'arithmetic', '', '200'), 'Answer: 200 or 201')
    assert not is_correct(Question('d', 'dates', '', 'Monday'), 'Answer: Monday or Tuesday')
    assert not is_correct(Question('a', 'arithmetic', '', '200'), 'Answer: 2000')


def test_unicode_minus_is_a_minus() -> None:
    """Found by the certified judge: it passed "\u221227 pens" for -27 while this checker failed it."""
    assert is_correct(Question('a', 'arithmetic', '', '-27'), 'Answer: \u221227 pens')
    assert not is_correct(Question('a', 'arithmetic', '', '-27'), 'Answer: 27 pens')
