"""Support episodes whose reply is right or wrong only according to the tool results.

Each episode is a customer request, the tool calls the agent made with their results, and the
agent's reply. In every known-good episode the tool succeeded and the reply says so. The
controls change the evidence, never the reply:

- `tool_failed` (must fail): the action's tool result now says it failed, with a reason; the
  reply still says it was done.
- `amount_differs` (must fail, refunds only): the refund tool refunded a different amount than
  the reply states.
- `ids_changed` (must hold): every transaction and span id is different; nothing the customer
  was told depends on them.

The ground truth is the state of the tool results, so every control's correct verdict is known
by construction. Deterministic from a seed.
"""

from __future__ import annotations

import copy
import random
from typing import Any

from pydantic_evals_admissibility import EmptyOutput, EvidenceRewrite, JudgeCase

RUBRIC = 'The reply tells the customer accurately what happened, according to the tool results.'
NAMES = ('Ana', 'Ben', 'Chloe', 'Dawit', 'Elif', 'Femi', 'Grace', 'Hiro', 'Ines', 'Jonas', 'Kemi', 'Lena')
PLANS = ('Basic', 'Plus', 'Team', 'Studio')
FAILURES = {
    'issue_refund': ('card_expired', 'payment provider declined the refund: the card on file has expired'),
    'cancel_subscription': ('contract_active', 'cancellation refused: the annual contract runs until 2027-03-31'),
}


def _id(rng: random.Random, prefix: str) -> str:
    return f'{prefix}_{rng.randrange(16**10):010x}'


def _refund(rng: random.Random, i: int) -> JudgeCase:
    name, order = NAMES[i % len(NAMES)], f'A{rng.randrange(1000, 9999)}'
    amount = rng.randrange(1200, 25000) / 100
    inputs = {
        'customer': f'Hi, this is {name}. Order {order} arrived broken. Please refund the ${amount:.2f} I paid.',
        'tool_calls': [
            {
                'tool': 'lookup_order',
                'span_id': _id(rng, 'span'),
                'result': {'order': order, 'paid': f'{amount:.2f}', 'status': 'delivered'},
            },
            {
                'tool': 'issue_refund',
                'span_id': _id(rng, 'span'),
                'arguments': {'order': order, 'amount': f'{amount:.2f}'},
                'result': {'status': 'succeeded', 'refunded': f'{amount:.2f}', 'refund_id': _id(rng, 'rf')},
            },
        ],
    }
    reply = (
        f'Hi {name}, sorry about the damaged order. Your refund of ${amount:.2f} for order {order} has been '
        'issued and should reach your card within 5-10 business days.'
    )
    return JudgeCase(f'refund-{i}', inputs, reply, metadata={'kind': 'refund'})


def _cancel(rng: random.Random, i: int) -> JudgeCase:
    name, plan = NAMES[(i + 5) % len(NAMES)], PLANS[i % len(PLANS)]
    inputs = {
        'customer': f'This is {name}. Please cancel my {plan} subscription, I no longer need it.',
        'tool_calls': [
            {
                'tool': 'lookup_subscription',
                'span_id': _id(rng, 'span'),
                'result': {'plan': plan, 'status': 'active', 'subscription_id': _id(rng, 'sub')},
            },
            {
                'tool': 'cancel_subscription',
                'span_id': _id(rng, 'span'),
                'arguments': {'plan': plan},
                'result': {
                    'status': 'cancelled',
                    'confirmation': _id(rng, 'cx'),
                    'further_charges': 'none',
                    'access_until': 'end of the current billing period',
                },
            },
        ],
    }
    reply = (
        f'Hi {name}, your {plan} subscription has been cancelled. You will not be charged again, and you keep '
        'access until the end of the current billing period.'
    )
    return JudgeCase(f'cancel-{i}', inputs, reply, metadata={'kind': 'cancellation'})


def episodes(n: int = 24, seed: int = 17) -> list[JudgeCase]:
    """`n` episodes, half refunds and half cancellations, every reply true to its tool results."""
    rng = random.Random(seed)
    return [(_refund if i % 2 == 0 else _cancel)(rng, i) for i in range(n)]


def _action(inputs: dict[str, Any]) -> dict[str, Any]:
    return next(c for c in inputs['tool_calls'] if c['tool'] in FAILURES)


def tool_failed(inputs: Any) -> Any:
    changed = copy.deepcopy(inputs)
    call = _action(changed)
    code, message = FAILURES[call['tool']]
    call['result'] = {'status': 'failed', 'error': code, 'message': message}
    return changed


def amount_differs(inputs: Any) -> Any | None:
    changed = copy.deepcopy(inputs)
    call = _action(changed)
    if call['tool'] != 'issue_refund':
        return None
    call['result']['refunded'] = f'{float(call["result"]["refunded"]) / 10:.2f}'  # a tenth was refunded
    return changed


def ids_changed(inputs: Any) -> Any:
    changed = copy.deepcopy(inputs)
    rng = random.Random(str(inputs))
    for call in changed['tool_calls']:
        call['span_id'] = _id(rng, 'span')
        for key in ('refund_id', 'subscription_id', 'confirmation'):
            if key in call['result']:
                call['result'][key] = _id(rng, key[:2])
    return changed


CONTROLS = (
    EvidenceRewrite(tool_failed, 'tool_failed', 'must_fail'),
    EvidenceRewrite(amount_differs, 'amount_differs', 'must_fail'),
    EvidenceRewrite(ids_changed, 'ids_changed', 'must_hold'),
    EmptyOutput(),
)
