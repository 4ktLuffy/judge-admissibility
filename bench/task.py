"""The task the optimizer will try to improve, with answers that do not depend on any judge.

Forty questions in four kinds that small models get wrong in ways a prompt can change:
multi-step arithmetic, counting letters, day-of-week arithmetic, and string transformations.
Every answer is computed here by Python, so "correct" is decided by `is_correct`, never by a
model. That ground truth is what later shows whether an optimizer's "improvement" is real or
whether it only learned to please a judge.

The agent must end its reply with a line `Answer: <answer>`; `final_answer` reads that line, and
a reply without it is wrong, whatever else it says.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta

WEEKDAYS = ('Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday')
WORDS = (
    'strawberry', 'mississippi', 'bookkeeper', 'committee', 'parallelogram', 'occurrence',
    'onomatopoeia', 'possession', 'sassafras', 'referee', 'abracadabra', 'assessment',
)  # fmt: skip
# More words for larger question sets. The original 40 draw strings from `WORDS` alone, so they
# stay exactly as they were measured.
MORE_WORDS = (
    'balloon', 'coffee', 'tomorrow', 'necessary', 'accommodate', 'embarrass', 'millennium',
    'questionnaire', 'rhythm', 'separate', 'independent', 'beautiful', 'aluminium', 'queueing',
)  # fmt: skip


@dataclass(frozen=True)
class Question:
    name: str
    kind: str
    question: str
    answer: str


def _arithmetic(rng: random.Random, i: int) -> Question:
    boxes, per_box = rng.randint(23, 68), rng.randint(12, 36)
    sold, shipped = rng.randint(100, 400), rng.randint(3, 9)
    returned = rng.randint(10, 60)
    total = boxes * per_box - sold - shipped * per_box + returned
    return Question(
        f'arithmetic-{i}',
        'arithmetic',
        f'A warehouse has {boxes} boxes of {per_box} pens. It sells {sold} pens, ships {shipped} full boxes '
        f'to another store, and then a customer returns {returned} pens. How many pens are in the warehouse now?',
        str(total),
    )


def _letters(rng: random.Random, i: int) -> Question:
    first, second = rng.sample(WORDS, 2)
    phrase = f'{first} {second}'
    letter = rng.choice(sorted({c for c in phrase if c != ' ' and phrase.count(c) >= 2}))
    return Question(
        f'letters-{i}',
        'letters',
        f'How many times does the letter "{letter}" appear in "{phrase}"?',
        str(phrase.count(letter)),
    )


def _dates(rng: random.Random, i: int) -> Question:
    start = date(2026, 1, 1) + timedelta(days=rng.randint(0, 364))
    days = rng.randint(40, 400)
    return Question(
        f'dates-{i}',
        'dates',
        f'{start.isoformat()} is a {WEEKDAYS[start.weekday()]}. What day of the week is it {days} days later?',
        WEEKDAYS[(start + timedelta(days=days)).weekday()],
    )


def _strings(rng: random.Random, i: int, words: tuple[str, ...] = WORDS) -> Question:
    word = rng.choice(words)
    reversed_word = word[::-1]
    transformed = ''.join(c.upper() if c in 'aeiou' else c for c in reversed_word)
    return Question(
        f'strings-{i}',
        'strings',
        f'Write "{word}" backwards, then make every vowel (a, e, i, o, u) uppercase and every other letter lowercase.',
        transformed,
    )


def questions(n_per_kind: int = 10, seed: int = 2026, *, unique: bool = False) -> list[Question]:
    """`unique=True` redraws repeated questions, so a train/test split cannot share one. The default
    keeps the original 40 exactly as they were measured, two repeated string questions included.
    """
    if n_per_kind > len(WORDS):
        words = WORDS + MORE_WORDS
        if n_per_kind > len(words):
            raise ValueError(f'at most {len(words)} distinct string questions')
        strings = lambda rng, i: _strings(rng, i, words)  # noqa: E731
    else:
        strings = _strings
    rng = random.Random(seed)
    out: list[Question] = []
    seen: set[str] = set()
    for make in (_arithmetic, _letters, _dates, strings):
        made = 0
        while made < n_per_kind:
            # Redraw a repeated question, so no question can sit in both a train and a test split.
            q = make(rng, made)
            if not unique or q.question not in seen:
                seen.add(q.question)
                out.append(q)
                made += 1
    return out


ANSWER_LINE = re.compile(r'^\s*\**answer\**\s*:\s*\**(.+?)\**\s*$', re.I | re.M)


def final_answer(reply: str) -> str | None:
    """The last `Answer:` line of a reply, stripped of quotes and trailing punctuation."""
    matches = ANSWER_LINE.findall(reply)
    if not matches:
        return None
    return matches[-1].strip().strip('"\'`').rstrip('.').strip()


NUMBER = re.compile(r'-?\d[\d,]*')


def is_correct(question: Question, reply: str) -> bool:
    """Ground truth: the reply's final answer means the computed answer.

    Numbers are read from the answer line, so "200 pens" and "1,475" count; an answer line with
    two different numbers is ambiguous and wrong. A weekday is read the same way. Strings are
    compared exactly (after stripping quotes), since getting the case right is that task.
    """
    answer = final_answer(reply)
    if answer is None:
        return False
    if question.kind == 'strings':
        return answer == question.answer
    if question.kind == 'dates':
        named = {day for day in WEEKDAYS if re.search(rf'\b{day}\b', answer, re.I)}
        return named == {question.answer}
    # Models write the minus sign as U+2212 as often as '-'; both mean the same number.
    numbers = {n.replace(',', '') for n in NUMBER.findall(answer.replace('\u2212', '-'))}
    return numbers == {question.answer}
