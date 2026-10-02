"""A support agent answering from a policy, with answers that code can check.

Closer to what pydantic-evals users grade than arithmetic: each case is a short fictional store
policy, with distracting details, and a customer question that needs the policy plus a little
reasoning (dates, thresholds, percentages). The agent replies in free text, the way a support
agent does, and ends with `Answer: <answer>`. The right answer is computed here, so ground truth
never depends on a model.

Questions avoid the edges where a policy could be read two ways: day counts are never within one
day of the return window, and warranty ages never equal the warranty length.
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass
from datetime import date, timedelta

STORES = ('Northwind Outfitters', 'Bluebell Home', 'Copperline Audio', 'Juniper Kitchenware', 'Atlas Cycle Co.',
          'Harbor Lane Books', 'Pinecrest Toys', 'Meridian Optics', 'Saffron Pantry', 'Kestrel Tech')  # fmt: skip
PRODUCTS = ('blender', 'headphones', 'desk lamp', 'backpack', 'kettle', 'bike light', 'speaker', 'tent')


@dataclass(frozen=True)
class Policy:
    store: str
    return_days: int
    restocking_pct: int
    free_shipping_over: int
    shipping_fee: float
    warranty_months: int

    def text(self) -> str:
        return (
            f'{self.store} customer policy.\n'
            f'Returns: items can be returned within {self.return_days} days of delivery. Opened items are '
            f'refunded minus a {self.restocking_pct}% restocking fee; unopened items are refunded in full. '
            f'Gift cards and final-sale items cannot be returned.\n'
            f'Shipping: orders of ${self.free_shipping_over} or more ship free; smaller orders pay a flat '
            f'${self.shipping_fee:.2f}. Express shipping is available at checkout for an extra $12.\n'
            f'Warranty: every product is covered against defects for {self.warranty_months} months from '
            f'purchase. Accidental damage is not covered.\n'
            f'Support hours: Monday to Friday, 9am to 6pm. Loyalty members earn 2 points per dollar.'
        )


@dataclass(frozen=True)
class SupportCase:
    name: str
    kind: str
    policy: Policy
    question: str
    answer: str  # 'yes', 'no', or a dollar amount like '42.50'

    @property
    def inputs(self) -> str:
        return f'Policy:\n{self.policy.text()}\n\nCustomer: {self.question}'


def _policy(rng: random.Random, store: str) -> Policy:
    return Policy(
        store=store,
        return_days=rng.choice((14, 30, 45, 60)),
        restocking_pct=rng.choice((10, 15, 20)),
        free_shipping_over=rng.choice((35, 50, 75)),
        shipping_fee=rng.choice((4.99, 6.99, 8.5)),
        warranty_months=rng.choice((6, 12, 24)),
    )


def _return(rng: random.Random, p: Policy, i: int) -> SupportCase:
    received = date(2026, 1, 1) + timedelta(days=rng.randint(0, 200))
    eligible = i % 2 == 0  # exactly balanced, so "always no" is not a good strategy
    elapsed = rng.randint(3, p.return_days - 2) if eligible else rng.randint(p.return_days + 2, p.return_days + 40)
    today = received + timedelta(days=elapsed)
    q = (f'My order was delivered on {received:%B %-d, %Y}. Today is {today:%B %-d, %Y}. '
         f'Can I still return it? (It is unopened.)')  # fmt: skip
    return SupportCase(f'return-{i}', 'return', p, q, 'yes' if eligible else 'no')


def _shipping(rng: random.Random, p: Policy, i: int) -> SupportCase:
    while True:
        total = round(rng.uniform(10, 120), 2)
        if abs(total - p.free_shipping_over) >= 1:
            break
    q = f'My cart comes to ${total:.2f} with standard shipping. How much will I pay for shipping?'
    fee = 0.0 if total >= p.free_shipping_over else p.shipping_fee
    return SupportCase(f'shipping-{i}', 'shipping', p, q, f'{fee:.2f}')


def _refund(rng: random.Random, p: Policy, i: int) -> SupportCase:
    price = round(rng.uniform(15, 300), 2)
    product = rng.choice(PRODUCTS)
    q = f'I opened my {product} (paid ${price:.2f}) and want to return it within the window. How much will I get back?'
    return SupportCase(f'refund-{i}', 'refund', p, q, f'{price * (100 - p.restocking_pct) / 100:.2f}')


def _warranty(rng: random.Random, p: Policy, i: int) -> SupportCase:
    covered = i % 2 == 0  # exactly balanced
    months = (
        rng.randint(1, p.warranty_months - 1) if covered else rng.randint(p.warranty_months + 1, p.warranty_months + 18)
    )
    product = rng.choice(PRODUCTS)
    q = f'My {product} stopped working on its own {months} months after I bought it. Is it covered by the warranty?'
    return SupportCase(f'warranty-{i}', 'warranty', p, q, 'yes' if covered else 'no')


def cases(n_per_kind: int = 10, seed: int = 11) -> list[SupportCase]:
    rng = random.Random(seed)
    out: list[SupportCase] = []
    for make in (_return, _shipping, _refund, _warranty):
        for i in range(n_per_kind):
            out.append(make(rng, _policy(rng, rng.choice(STORES)), i))
    return out


_ANSWER = re.compile(r'^\s*\**answer\**\s*:\s*\**(.+?)\**\s*$', re.I | re.M)
_MONEY = re.compile(r'\$?\s*(\d[\d,]*(?:\.\d+)?)')


def is_correct(case: SupportCase, reply: str) -> bool:
    """The reply's last `Answer:` line means the computed answer.

    Yes/no answers are read from the first word ("Yes, it is covered" counts); amounts from the
    one number on the line, within a cent ("$0", "free" and "$0.00" are all 0.00 for shipping).
    """
    lines = _ANSWER.findall(reply)
    if not lines:
        return False
    line = lines[-1].strip().strip('"\'`').rstrip('.')
    if case.answer in ('yes', 'no'):
        first = re.match(r'\W*(yes|no)\b', line, re.I)
        return bool(first) and first.group(1).lower() == case.answer
    if case.kind == 'shipping' and re.search(r'\bfree\b', line, re.I) and not _MONEY.search(line):
        return case.answer == '0.00'
    amounts = {float(m.replace(',', '')) for m in _MONEY.findall(line)}
    return len(amounts) == 1 and abs(amounts.pop() - float(case.answer)) < 0.011
