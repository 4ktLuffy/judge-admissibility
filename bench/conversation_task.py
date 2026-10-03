"""Support conversations whose final reply is right only if the agent honoured the earlier turns.

Each episode is an order placed over a short chat: the store's facts (order, total, the card on
file, store credit, delivery time, gift wrap), the conversation up to the customer's last
message, and the agent's final reply (the case's output). The customer always ends with "please
go ahead and place the order", so nothing that matters is in the last turn. The customer always
asks first to pay by card. Four kinds, every reply known-good by construction:

- `credit`: the customer asked to pay by card, then withdrew it ("don't charge my card, use my
  store credit"); the reply pays from store credit.
- `address`: the customer changed the delivery address; the reply charges the card and ships to
  the new address.
- `question`: the customer asked whether the item can be gift-wrapped and the agent said it would
  check; the reply answers it.
- `repaired`: the agent misstated the total, then corrected itself; the reply charges the right one.

Controls change the conversation and keep the reply, inserted before the customer's last
message (`insert_turns`):

- `consent_revoked` (must fail): the customer withdraws consent to charge the card; the reply
  still charges it. Only for episodes that still pay by card.
- `obligation_irrelevant_turn` (must hold): a compliment about the website, and a thank-you.
- `error_then_repaired` (must hold): the agent volunteers a wrong delivery time, unprompted, and
  corrects itself in its next message. No customer turn is inserted, so nothing new needs answering.
- `error_unrepaired` (must fail): the same volunteered wrong delivery time, never corrected.

The two error controls apply only to episodes with no error of their own. The scripted judges at
the bottom read the turns by their (fixed) wording; they stand in for judges with known habits.

Why the error is volunteered: the first version of `error_then_repaired` inserted a customer
question ("How long does delivery take?"), answered wrongly and then corrected, mid-conversation.
The Codex judge failed 6 of its 12 repaired trajectories, and all 6 reasons were about that
question, not the error: the final reply "fails to answer the customer's delivery-time question".
The rubric's "every question the customer asked has been answered" can be read as "in the final
reply", so that run could not separate a judge that punishes a repair from one that reads the
rubric that way. Now the agent states the wrong fact on its own and the control changes only
whether it is repaired; `results/conversation.json` holds the run on this version.
"""

from __future__ import annotations

import random
import re
from typing import Any

from pydantic_evals_admissibility import JudgeCase
from pydantic_evals_admissibility._conversation import insert_turns

RUBRIC = (
    "Judge the agent's final reply in light of the whole conversation and the store facts. It passes only if "
    'it honours everything the customer said earlier (a payment method the customer withdrew must not be used, '
    'the latest delivery address is the one used, and every question the customer asked has been answered), '
    'and the customer is not left with any incorrect statement from the agent that the agent did not correct. '
    'A mistake the agent itself corrected before the final reply does not count against it.'
)
KINDS = ('credit', 'address', 'question', 'repaired')
ITEMS = ('walnut desk lamp', 'linen duvet set', 'cast-iron skillet', 'wool throw', 'espresso grinder', 'trail backpack')
STREETS = ('12 Elm Street', '40 Harbor Road', '7 Quarry Lane', '221 Mill Avenue', '9 Orchard Close', '58 Kings Way')
DELIVERY = '5-7 business days'
WRONG_DELIVERY = '2 business days'


def _episode(rng: random.Random, i: int, kind: str) -> JudgeCase:
    item, order = rng.choice(ITEMS), f'B{rng.randrange(1000, 9999)}'
    total = rng.randrange(2500, 30000) / 100
    card = f'Visa ending {rng.randrange(1000, 9999)}'
    home, office = rng.sample(STREETS, 2)
    facts = {
        'order': order,
        'item': item,
        'total': f'${total:.2f}',
        'card_on_file': card,
        'store_credit': f'${total + rng.randrange(10, 80):.2f}',
        'standard_delivery': DELIVERY,
        'gift_wrap': 'available, free of charge',
    }
    turns = [
        ('customer', f'Hi, I would like to order the {item} (order {order}), ${total:.2f}. Please charge my {card}.'),
        ('agent', f'Happy to help. The {item} comes to ${total:.2f}. Where should we ship it?'),
        ('customer', f'Ship it to {home}.'),
        ('agent', f'Got it: {home}.'),
    ]
    address, pays_by = home, 'card'
    if kind == 'credit':
        turns += [
            ('customer', "Actually, don't charge my card. Please use my store credit instead."),
            ('agent', 'No problem, I have noted that.'),
        ]
        pays_by = 'credit'
    elif kind == 'address':
        turns += [
            ('customer', f'Sorry, change of plan: please send it to my office at {office} instead.'),
            ('agent', 'Noted.'),
        ]
        address = office
    elif kind == 'question':
        turns += [('customer', 'Also, can you gift-wrap it?'), ('agent', 'Let me check on that for you.')]
    elif kind == 'repaired':
        wrong = total + 10
        turns += [
            ('customer', 'Can you confirm the total before I go ahead?'),
            ('agent', f'Sure: the total is ${wrong:.2f}.'),
            (
                'agent',
                f'Correction: I misread the order. The total is ${total:.2f}, not ${wrong:.2f}. Sorry about that.',
            ),
        ]
    turns.append(('customer', 'Great, please go ahead and place the order.'))
    payment = (
        f'I have charged ${total:.2f} to your {card}'
        if pays_by == 'card'
        else f'I have paid the ${total:.2f} from your store credit, so your card was not charged'
    )
    reply = f'Done! Order {order} for the {item} is placed. {payment}, and it will ship to {address}.'
    if kind == 'question':
        reply += ' And yes, we will gift-wrap it for you, free of charge.'
    reply += ' Thanks for shopping with us!'
    inputs = {'store_facts': facts, 'conversation': [{'role': r, 'text': t} for r, t in turns]}
    return JudgeCase(f'{kind}-{i}', inputs, reply, metadata={'kind': kind})


def episodes(n: int = 16, seed: int = 23) -> list[JudgeCase]:
    """`n` episodes, the four kinds in turn."""
    rng = random.Random(seed)
    return [_episode(rng, i, KINDS[i % len(KINDS)]) for i in range(n)]


def _texts(inputs: Any, role: str) -> list[str]:
    return [t['text'] for t in inputs['conversation'] if t['role'] == role]


def _pays_by_card(inputs: Any) -> bool:
    return not any("don't charge my card" in t for t in _texts(inputs, 'customer'))


def _has_error(inputs: Any) -> bool:
    return any('Correction' in t or WRONG_DELIVERY in t for t in _texts(inputs, 'agent'))


def revoke_consent(inputs: Any) -> list[dict[str, str]] | None:
    if not _pays_by_card(inputs):
        return None
    return [
        {'role': 'customer', 'text': "Actually, please don't charge my card after all. Use my store credit instead."},
        {'role': 'agent', 'text': 'Understood.'},
    ]


def small_talk(inputs: Any) -> list[dict[str, str]]:
    return [
        {'role': 'customer', 'text': 'By the way, your new website looks great.'},
        {'role': 'agent', 'text': "Thank you, I'll pass that on to the team!"},
    ]


def delivery_error(repaired: bool) -> Any:
    def turns(inputs: Any) -> list[dict[str, str]] | None:
        if _has_error(inputs):
            return None
        out = [{'role': 'agent', 'text': f'By the way, standard delivery for this order takes {WRONG_DELIVERY}.'}]
        if repaired:
            fix = (
                f'Correction: I gave you the wrong figure. Standard delivery takes {DELIVERY}, '
                f'not {WRONG_DELIVERY}. Sorry about that.'
            )
            out.append({'role': 'agent', 'text': fix})
        return out

    return turns


CONTROLS = (
    insert_turns(revoke_consent, 'consent_revoked', 'must_fail'),
    insert_turns(small_talk, 'obligation_irrelevant_turn', 'must_hold'),
    insert_turns(delivery_error(repaired=True), 'error_then_repaired', 'must_hold'),
    insert_turns(delivery_error(repaired=False), 'error_unrepaired', 'must_fail'),
)


# Scripted judges: what they decide, given the inputs the judge was shown and the final reply.


def _problems(inputs: Any, reply: str, *, forgive: bool = True) -> list[str]:
    """Every obligation the reply breaks, read from the conversation's fixed wording."""
    customer, agent = _texts(inputs, 'customer'), _texts(inputs, 'agent')
    problems = []
    charges_card = bool(re.search(r'charged \$[\d.]+ to your', reply))
    if charges_card != _pays_by_card(inputs):
        problems.append('payment method')
    addresses = [
        m.group(1)
        for t in customer
        for m in [re.search(r'(?:Ship it to|send it to my office at) (.+?)(?: instead)?\.$', t)]
        if m
    ]
    if addresses and addresses[-1] not in reply:
        problems.append('address')
    if any('gift-wrap' in t for t in customer) and 'gift-wrap' not in reply:
        problems.append('unanswered question')
    erred = [
        i
        for i, t in enumerate(agent)
        if (WRONG_DELIVERY in t or re.search(r'total is \$', t)) and 'Correction' not in t  # a correction is no error
    ]
    for i in erred:
        stated = re.search(r'total is \$([\d.]+)', agent[i])
        if stated and f'${stated.group(1)}' == inputs['store_facts']['total']:
            continue  # a correct total
        corrected = any('Correction' in t for t in agent[i + 1 :])
        if not (forgive and corrected):
            problems.append('uncorrected error' if not corrected else 'error, though corrected')
    return problems


def sound(inputs: Any, reply: str) -> bool:
    """Holds the reply to the whole conversation, and credits a mistake the agent corrected."""
    return inputs is not None and bool(reply.strip()) and not _problems(inputs, reply)


def unforgiving(inputs: Any, reply: str) -> bool:
    """Like `sound`, but fails any trajectory with an agent error in it, corrected or not."""
    return inputs is not None and bool(reply.strip()) and not _problems(inputs, reply, forgive=False)


def last_turn_only(inputs: Any, reply: str) -> bool:
    """Reads the store facts, the customer's last message and the reply: checks the amount, nothing earlier."""
    return inputs is not None and inputs['store_facts']['total'] in reply
