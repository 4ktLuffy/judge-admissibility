"""The support task's ground truth, recomputed independently, and its checker's edge cases."""

from __future__ import annotations

import re
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'bench'))
from support_task import SupportCase, cases, is_correct  # noqa: E402


def test_forty_unique_cases_four_kinds() -> None:
    cs = cases()
    assert len(cs) == 40 and len({c.name for c in cs}) == 40
    assert Counter(c.kind for c in cs) == {'return': 10, 'shipping': 10, 'refund': 10, 'warranty': 10}
    assert cases() == cs


def test_answers_recomputed_from_the_policy_text() -> None:
    """Read the facts back out of the text the agent sees, not from the generator's fields."""
    for c in cases():
        text = c.policy.text()
        if c.kind == 'return':
            window = int(re.search(r'within (\d+) days', text).group(1))  # type: ignore[union-attr]
            d1, d2 = (datetime.strptime(d, '%B %d, %Y') for d in re.findall(r'(\w+ \d+, \d{4})', c.question))
            assert c.answer == ('yes' if (d2 - d1).days < window else 'no')
        elif c.kind == 'shipping':
            over = int(re.search(r'orders of \$(\d+)', text).group(1))  # type: ignore[union-attr]
            fee = float(re.search(r'flat \$(\d+\.\d+)', text).group(1))  # type: ignore[union-attr]
            total = float(re.search(r'\$(\d+\.\d+)', c.question).group(1))  # type: ignore[union-attr]
            assert float(c.answer) == (0.0 if total >= over else fee)
        elif c.kind == 'refund':
            pct = int(re.search(r'(\d+)% restocking', text).group(1))  # type: ignore[union-attr]
            price = float(re.search(r'\$(\d+\.\d+)', c.question).group(1))  # type: ignore[union-attr]
            assert abs(float(c.answer) - price * (100 - pct) / 100) < 0.006
        else:
            months = int(re.search(r'for (\d+) months', text).group(1))  # type: ignore[union-attr]
            age = int(re.search(r'(\d+) months after', c.question).group(1))  # type: ignore[union-attr]
            assert c.answer == ('yes' if age < months else 'no')


def test_checker_reads_meaning_not_format() -> None:
    c = cases()
    yes = next(x for x in c if x.answer == 'yes')
    money = next(x for x in c if x.kind == 'refund')
    free = next((x for x in c if x.kind == 'shipping' and x.answer == '0.00'), None)
    assert is_correct(yes, 'Sure thing!\nAnswer: Yes, it is covered.')
    assert not is_correct(yes, 'Answer: No')
    assert not is_correct(yes, 'Yes it is.')  # no answer line
    assert is_correct(money, f'Answer: ${money.answer}')
    assert not is_correct(money, f'Answer: ${float(money.answer) + 1:.2f}')
    assert not is_correct(money, f'Answer: ${money.answer} or ${float(money.answer) + 1:.2f}')
    if free is not None:
        assert is_correct(free, 'Answer: Free') and is_correct(free, 'Answer: $0')


def test_no_case_sits_on_an_ambiguous_edge() -> None:
    for c in cases():
        if c.kind == 'warranty':
            age = int(re.search(r'(\d+) months after', c.question).group(1))  # type: ignore[union-attr]
            assert age != c.policy.warranty_months
        if c.kind == 'return':
            d1, d2 = (datetime.strptime(d, '%B %d, %Y') for d in re.findall(r'(\w+ \d+, \d{4})', c.question))
            assert abs((d2 - d1).days - c.policy.return_days) > 1


def test_inputs_carry_policy_and_question() -> None:
    c: SupportCase = cases()[0]
    assert c.policy.store in c.inputs and c.question in c.inputs
