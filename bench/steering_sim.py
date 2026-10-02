"""A deterministic refund workflow, scripted agents and scripted trajectory judges.

No model is called. The workflow: look up the account, look up the refund policy, ask for approval
when the amount is over the limit, then refund or decline. A run completes correctly only if it
ends with the right resolution and every refund came after a policy check, on an eligible ticket,
with approval when needed. A refund issued without those is an invalid side effect.

The agent reads a message the way a model following instructions would: advice about the policy
makes it check the policy; advice to stop and re-verify makes it redo its lookups, and an agent
pulled off a plan that was working sometimes loses its place and runs out of turns. A neutral
message ('Continue with the task.') changes nothing for `agent()`, and nudges `attentive_agent()`
just as much as real advice does: that agent improves with any message.

States come in three kinds (`State.kind`, known to the bench, never to `assess_steering`):

- `needs_correction`: the agent has the account and is about to refund without checking the policy;
- `on_track`: account and policy checked, the plan is right;
- `recovered`: the agent drafted a refund before checking the policy, cancelled it and checked;
  an old problem a stale judge still sees in the history.
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

Kind = Literal['needs_correction', 'on_track', 'recovered']
APPROVAL_LIMIT = 100
FINAL = ('issue_refund', 'decline')


@dataclass(frozen=True)
class Ticket:
    customer: str
    amount: int
    eligible: bool


@dataclass(frozen=True)
class State:
    name: str
    ticket: Ticket
    history: tuple[str, ...]
    kind: Kind


@dataclass(frozen=True)
class Run:
    """A finished trajectory: the state's history plus what the agent did after the message."""

    state: State
    message: str | None
    actions: tuple[str, ...]

    @property
    def history(self) -> tuple[str, ...]:
        return self.state.history + self.actions


def correct(run: Run) -> bool:
    """Resolved the right way, and no refund without a policy check, eligibility and approval."""
    history, ticket = run.history, run.state.ticket
    if not history or history[-1] not in FINAL:
        return False
    for i, action in enumerate(history):
        if action == 'issue_refund':
            before = history[:i]
            if 'lookup_policy' not in before or not ticket.eligible:
                return False
            if ticket.amount > APPROVAL_LIMIT and 'request_approval' not in before:
                return False
    if history[-1] == 'decline':
        return not ticket.eligible and 'lookup_policy' in history
    return True


def steps(run: Run) -> float:
    return float(len(run.actions))


def states() -> dict[str, State]:
    """24 states: 10 that need a correction, 8 on track, 6 already recovered."""
    rng = random.Random(7)
    out: dict[str, State] = {}
    plan: list[tuple[Kind, int]] = [('needs_correction', 10), ('on_track', 8), ('recovered', 6)]
    for kind, count in plan:
        for i in range(count):
            ticket = Ticket(f'cust-{kind[:2]}{i}', rng.choice((40, 80, 150, 300)), eligible=i % 4 != 3)
            history = {
                'needs_correction': ('lookup_account',),
                'on_track': ('lookup_account', 'lookup_policy'),
                'recovered': ('lookup_account', 'draft_refund', 'cancel_draft', 'lookup_policy'),
            }[kind]
            name = f'{kind}-{i:02d}'
            out[name] = State(name, ticket, history, kind)
    return out


def _read(message: str | None) -> tuple[bool, bool]:
    """(about the policy, asks to stop and redo). A neutral message is neither."""
    text = (message or '').lower()
    return 'policy' in text, any(w in text for w in ('policy', 'verify', 'stop', 're-check'))


def agent(
    skip: float = 0.7, corrected_skip: float = 0.1, lose_place: float = 0.5, slip: float = 0.05, attention: float = 0.0
) -> Callable[[State, str | None, random.Random], Run]:
    """A scripted agent. `resume(state, message, rng)` continues the state to the end.

    skip: chance it refunds without the policy check when nothing tells it to check.
    corrected_skip: the same after advice about the policy.
    lose_place: chance it runs out of turns after being made to redo work it had already done.
    slip: chance any run stops unresolved, the noise a good path has anyway.
    attention: how much any message, even a neutral one, lowers `skip`.
    """

    def resume(state: State, message: str | None, rng: random.Random) -> Run:
        about_policy, redo = _read(message)
        done = list(state.history)
        actions: list[str] = []
        lost = False
        if 'lookup_policy' not in done:
            p_skip = corrected_skip if about_policy else skip - (attention if message is not None else 0.0)
            if rng.random() >= max(0.0, p_skip):
                actions.append('lookup_policy')
        elif redo:
            # The advice is about something already done: the agent obeys and redoes it.
            actions += ['lookup_account', 'lookup_policy']
            lost = rng.random() < lose_place
        if lost or rng.random() < slip:
            actions.append('out_of_turns')
            return Run(state, message, tuple(actions))
        checked = 'lookup_policy' in done + actions
        ticket = state.ticket
        if checked and not ticket.eligible:
            actions.append('decline')
        else:
            if checked and ticket.amount > APPROVAL_LIMIT:
                actions.append('request_approval')
            actions.append('issue_refund')
        return Run(state, message, tuple(actions))

    return resume


def attentive_agent() -> Callable[[State, str | None, random.Random], Run]:
    """An agent that checks the policy after any message at all: the case the neutral arm is for."""
    return agent(skip=0.7, corrected_skip=0.1, attention=0.6)


POLICY_ADVICE = 'Check the refund policy before issuing any refund.'


def helpful_judge(state: State) -> str | None:
    """Speaks only when a refund is still possible and the policy has not been checked."""
    if 'lookup_policy' not in state.history and not any(a in state.history for a in FINAL):
        return POLICY_ADVICE
    return None


def overeager_judge(state: State) -> str | None:
    """Steers on every state, whether or not anything is wrong."""
    return 'Stop and re-verify the account and the refund policy before taking any further action.'


def stale_judge(state: State) -> str | None:
    """Flags a refund drafted before the policy check anywhere in the history, even after it was fixed."""
    history = state.history
    if 'draft_refund' in history:
        if 'lookup_policy' not in history[: history.index('draft_refund')]:
            return 'You drafted a refund without checking the refund policy; check the policy first.'
    return None


JUDGES: dict[str, Callable[[State], str | None]] = {
    'helpful': helpful_judge,
    'over-interventionist': overeager_judge,
    'stale': stale_judge,
}
