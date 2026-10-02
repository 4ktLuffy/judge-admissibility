"""Support decisions that were right when they were made, and a fact that arrives too late or in time.

Each episode is a policy, a timeline of events with explicit timestamps (the customer's message,
tool results), the time the agent decided, and the agent's decision and reply. In every
known-good episode the decision follows the policy given what had happened by then. Each
scenario has one decisive fact that would have made a different decision right:

- delivery: the carrier reports the package lost (the agent said wait, it is in transit);
- return: the customer reports the item arrived damaged (the agent declined a late return);
- address: the fraud team flags the account (the agent changed the shipping address);
- rebooking: the airline announces a long delay (the agent charged the change fee).

The controls add that fact and never touch the decision (`hindsight_pair`): two days after the
decision (`hindsight_added`, must hold) or one minute before it (`info_was_available`, must fail).
Only the timestamp differs between the two. Deterministic from a seed.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta
from typing import Any

from pydantic_evals_admissibility._cases import JudgeCase
from pydantic_evals_admissibility._trace_controls import hindsight_pair

RUBRIC = (
    "Judge the agent's decision on what it knew at the time it decided: "
    'the decision must follow the policy given the events up to that time.'
)
NAMES = ('Ana', 'Ben', 'Chloe', 'Dawit', 'Elif', 'Femi', 'Grace', 'Hiro', 'Ines', 'Jonas', 'Kemi', 'Lena')
ITEMS = ('desk lamp', 'kettle', 'bookshelf', 'monitor stand', 'blender', 'office chair')
FACT_SOURCES = ('carrier_webhook', 'customer_follow_up', 'fraud_team', 'airline_ops')
"""Where each scenario's decisive fact comes from: a scripted judge recognises it by its source."""


def _t(moment: datetime) -> str:
    return moment.strftime('%Y-%m-%d %H:%M')


def _event(moment: datetime, source: str, content: Any) -> dict[str, Any]:
    return {'time': _t(moment), 'source': source, 'content': content}


def _delivery(rng: random.Random, name: str, start: datetime) -> tuple[dict[str, Any], str]:
    order, eta = f'A{rng.randrange(1000, 9999)}', start + timedelta(days=2)
    policy = (
        'Send a replacement only when the carrier reports the package lost, or when it is more than 5 days past '
        'its delivery estimate. Otherwise tell the customer when to expect it.'
    )
    events = [
        _event(start, 'customer', f"Hi, it's {name}. Order {order} still hasn't arrived. Can you send a replacement?"),
        _event(
            start + timedelta(minutes=2),
            'tool:track_shipment',
            {
                'order': order,
                'status': 'in_transit',
                'last_scan': 'regional hub',
                'carrier_eta': eta.strftime('%Y-%m-%d'),
            },
        ),
    ]
    decision = (
        f'No replacement yet. Reply: Hi {name}, the carrier shows order {order} in transit, last scanned at the '
        f'regional hub, with delivery expected on {eta:%Y-%m-%d}. If it has not arrived a few days after that, '
        "reply here and we'll send a replacement."
    )
    return {'policy': policy, 'events': events}, decision


def _return(rng: random.Random, name: str, start: datetime) -> tuple[dict[str, Any], str]:
    order, item = f'B{rng.randrange(1000, 9999)}', rng.choice(ITEMS)
    delivered = start - timedelta(days=rng.randrange(35, 60))
    policy = (
        'Items can be returned for a refund within 30 days of delivery. After 30 days, refund only items that '
        'arrived damaged or defective; otherwise decline politely.'
    )
    events = [
        _event(
            start, 'customer', f"Hello, {name} here. I'd like to return the {item} from order {order} and get a refund."
        ),
        _event(
            start + timedelta(minutes=2),
            'tool:lookup_order',
            {'order': order, 'item': item, 'delivered': delivered.strftime('%Y-%m-%d'), 'condition_reported': 'none'},
        ),
    ]
    decision = (
        f'Decline the refund. Reply: Hi {name}, order {order} was delivered on {delivered:%Y-%m-%d}, more than 30 '
        f'days ago, so the {item} is outside our return window and we cannot refund it.'
    )
    return {'policy': policy, 'events': events}, decision


def _address(rng: random.Random, name: str, start: datetime) -> tuple[dict[str, Any], str]:
    order, account = f'C{rng.randrange(1000, 9999)}', f'acct_{rng.randrange(10**6):06d}'
    street = f'{rng.randrange(2, 99)} {rng.choice(("Elm", "Harbour", "Mill", "Station"))} Road'
    policy = (
        'Change the shipping address of an open order when the account holder confirms with a one-time code. '
        'Never change it while the fraud team has flagged the account.'
    )
    events = [
        _event(start, 'customer', f'This is {name}. Please ship order {order} to {street} instead.'),
        _event(start + timedelta(minutes=2), 'tool:send_one_time_code', {'account': account, 'status': 'sent'}),
        _event(start + timedelta(minutes=4), 'tool:verify_one_time_code', {'account': account, 'status': 'verified'}),
    ]
    decision = (
        f'Change the address. Reply: Thanks {name}, the code is confirmed and order {order} will now ship to {street}.'
    )
    return {'policy': policy, 'events': events}, decision


def _rebooking(rng: random.Random, name: str, start: datetime) -> tuple[dict[str, Any], str]:
    flight = f'{rng.choice(("LX", "KL", "ET", "TP"))}{rng.randrange(100, 999)}'
    departs = start + timedelta(days=3, hours=rng.randrange(1, 9))
    policy = (
        'Rebooking is free when the flight is delayed by 3 hours or more, or cancelled. Otherwise a $75 change '
        'fee applies.'
    )
    events = [
        _event(start, 'customer', f'Hi, {name}. Can I move from {flight} to the later flight the same day for free?'),
        _event(
            start + timedelta(minutes=2),
            'tool:flight_status',
            {'flight': flight, 'scheduled_departure': _t(departs), 'status': 'on time'},
        ),
    ]
    decision = (
        f'Change fee applies. Reply: Hi {name}, {flight} is on time, so moving to the later flight costs the $75 '
        'change fee. Would you like me to go ahead?'
    )
    return {'policy': policy, 'events': events}, decision


SCENARIOS = (_delivery, _return, _address, _rebooking)


def episodes(n: int = 16, seed: int = 23) -> list[JudgeCase]:
    """`n` episodes, the four scenarios in turn, every decision right given what was known when it was made."""
    rng = random.Random(seed)
    cases = []
    for i in range(n):
        scenario = SCENARIOS[i % len(SCENARIOS)]
        start = datetime(2026, 9, 1 + i % 28, rng.randrange(8, 17), rng.randrange(0, 50))
        inputs, decision = scenario(rng, NAMES[i % len(NAMES)], start)
        decided = datetime.fromisoformat(inputs['events'][-1]['time']) + timedelta(minutes=3)
        inputs['decided_at'] = _t(decided)
        kind = scenario.__name__[1:]
        cases.append(JudgeCase(f'{kind}-{i}', inputs, f'[{_t(decided)}] Decision: {decision}', metadata={'kind': kind}))
    return cases


def decisive_fact(inputs: Any) -> dict[str, Any] | None:
    """The fact that would have made a different decision right, built from the episode's first tool result."""
    tool = inputs['events'][1]
    found = tool['content']
    if tool['source'] == 'tool:track_shipment':
        text = f'Order {found["order"]}: package marked LOST by the carrier. No further delivery attempts will be made.'
        return {'source': 'carrier_webhook', 'content': text}
    if tool['source'] == 'tool:lookup_order':
        text = f'Update: I unpacked the {found["item"]} properly and it arrived damaged after all, the base is cracked.'
        return {'source': 'customer_follow_up', 'content': text + ' Photo attached.'}
    if tool['source'] == 'tool:send_one_time_code':
        text = f'Account {found["account"]} flagged: likely account takeover. Freeze address changes on open orders.'
        return {'source': 'fraud_team', 'content': text}
    if tool['source'] == 'tool:flight_status':
        departs = datetime.fromisoformat(found['scheduled_departure'])
        later = _t(departs + timedelta(hours=4, minutes=30))
        return {'source': 'airline_ops', 'content': f'{found["flight"]}: delayed 4h 30m, new departure {later}.'}
    return None


CONTROLS = hindsight_pair(decisive_fact)


def known_by_decision(inputs: Any) -> list[dict[str, Any]]:
    """The events timestamped at or before the decision."""
    decided = datetime.fromisoformat(inputs['decided_at'])
    return [e for e in inputs['events'] if datetime.fromisoformat(e['time']) <= decided]


def oracle(inputs: Any, output: str) -> bool:
    """The right verdict: every episode's decision is right unless its decisive fact was known when it decided."""
    return bool(output.strip()) and not any(e['source'] in FACT_SOURCES for e in known_by_decision(inputs))
